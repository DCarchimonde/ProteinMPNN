import argparse
import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint
import json
import numpy as np
import random
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from sklearn.metrics import classification_report
from torch.optim.lr_scheduler import CosineAnnealingLR

# --- 系统路径 ---
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    print(f"[System] Random Seed locked to: {seed}")

try:
    from model_utils import ProteinMPNN, gather_nodes, cat_neighbors_nodes
    from nmethyl.utils.nmethyl_config import (
        EXTENDED_AA_ALPHABET, NATURAL_AA_ALPHABET,
        NMETHYL_TO_NATURAL_MAPPING, EXTENDED_AA_TO_INDEX
    )
except ImportError as e:
    print(f"Error: Missing project files. {e}")
    sys.exit(1)

# =============================================================================
# 1. 模型定义 (强制 k=48)
# =============================================================================

class DecoupledProteinMPNN(ProteinMPNN):
    def __init__(self, hidden_dim=128, **kwargs):
        super().__init__(
            num_letters=21, 
            hidden_dim=hidden_dim, 
            vocab=21, 
            k_neighbors=48,  # <--- 这里的 48 是关键！
            **kwargs
        )
        self.W_s = nn.Embedding(len(EXTENDED_AA_ALPHABET), hidden_dim)
        self.W_out_base = nn.Linear(hidden_dim, len(NATURAL_AA_ALPHABET))
        self.W_out_methyl = nn.Linear(hidden_dim, 2)

    def forward(self, X, S, mask, chain_M, residue_idx, chain_encoding_all):
        E, E_idx = self.features(X, mask, residue_idx, chain_encoding_all)
        h_V = torch.zeros((E.shape[0], E.shape[1], E.shape[-1]), device=E.device)
        h_E = self.W_e(E)

        mask_attend = gather_nodes(mask.unsqueeze(-1),  E_idx).squeeze(-1)
        mask_attend = mask.unsqueeze(-1) * mask_attend
        for layer in self.encoder_layers:
            h_V, h_E = torch.utils.checkpoint.checkpoint(layer, h_V, h_E, E_idx, mask, mask_attend)

        h_S = self.W_s(S)
        h_ES = cat_neighbors_nodes(h_S, h_E, E_idx)
        h_EX_encoder = cat_neighbors_nodes(torch.zeros_like(h_S), h_E, E_idx)
        h_EXV_encoder = cat_neighbors_nodes(h_V, h_EX_encoder, E_idx)

        chain_M = chain_M * mask
        decoding_order = torch.argsort((chain_M + 0.0001) * (torch.abs(torch.randn(chain_M.shape, device=X.device))))
        mask_size = E_idx.shape[1]
        permutation_matrix_reverse = torch.nn.functional.one_hot(decoding_order, num_classes=mask_size).float()
        order_mask_backward = torch.einsum('ij, biq, bjp->bqp', (1 - torch.triu(torch.ones(mask_size, mask_size, device=X.device))), permutation_matrix_reverse, permutation_matrix_reverse)
        mask_attend = torch.gather(order_mask_backward, 2, E_idx).unsqueeze(-1)
        mask_1D = mask.view([mask.size(0), mask.size(1), 1, 1])
        mask_bw = mask_1D * mask_attend
        mask_fw = mask_1D * (1. - mask_attend)
        h_EXV_encoder_fw = mask_fw * h_EXV_encoder
        
        for layer in self.decoder_layers:
            h_ESV = cat_neighbors_nodes(h_V, h_ES, E_idx)
            h_ESV = mask_bw * h_ESV + h_EXV_encoder_fw
            h_V = torch.utils.checkpoint.checkpoint(layer, h_V, h_ESV, mask)
        
        logits_base = self.W_out_base(h_V)
        logits_methyl = self.W_out_methyl(h_V)
        return logits_base, logits_methyl

# =============================================================================
# 2. 增强版权重加载器 (Parent Initialization)
# =============================================================================

def advanced_load_weights(model, pretrained_path, device):
    print(f"\n>>> [v10 Advanced Loader] Opening {pretrained_path}...")
    if not os.path.exists(pretrained_path):
        print("❌ Error: File not found.")
        sys.exit(1)
        
    ckpt = torch.load(pretrained_path, map_location=device)
    state_dict = ckpt.get('model_state_dict', ckpt)
    model_state = model.state_dict()
    
    # A. 骨架加载 (Backbone)
    print("[Loader] Matching backbone keys...")
    load_dict = {}
    for k, v in state_dict.items():
        clean_k = k.replace("module.", "")
        if clean_k in model_state and v.shape == model_state[clean_k].shape:
            load_dict[clean_k] = v
    
    print(f"  ✅ Loaded {len(load_dict)} backbone keys.")
    model.load_state_dict(load_dict, strict=False)

    # B. Embedding 初始化 (包含亲本迁移)
    print("[Loader] Initializing Embeddings...")
    ws_file = state_dict.get('W_s.weight', state_dict.get('module.W_s.weight'))
    
    if ws_file is not None:
        STD_ALPHABET = 'ACDEFGHIKLMNPQRSTVWYX'
        
        # 1. 迁移标准氨基酸 (A->A)
        with torch.no_grad():
            mapped_std = 0
            for i, char in enumerate(STD_ALPHABET):
                if char in EXTENDED_AA_TO_INDEX:
                    target_idx = EXTENDED_AA_TO_INDEX[char]
                    model.W_s.weight.data[target_idx].copy_(ws_file[i])
                    mapped_std += 1
            print(f"  ✅ Mapped {mapped_std} standard embeddings.")

        # 2. [关键] 亲本初始化 (C -> c)
        # 将 N-甲基化氨基酸初始化为其对应的天然氨基酸权重
        # 这样模型一开始就把 'c' 当作 'C' 看待，而不是随机噪声
        print("[Loader] Performing Parent Initialization (Natural -> N-methyl)...")
        with torch.no_grad():
            mapped_methyl = 0
            # 遍历映射表：N-Me (relative idx) -> Natural (relative idx)
            for nme_idx, nat_idx in NMETHYL_TO_NATURAL_MAPPING.items():
                # 这里的 key 是 0..18 (relative)，我们需要找到它们在 Extended 中的真实位置
                # 假设 nme 字母在 EXTENDED_AA_ALPHABET 的后半段
                # 更好的方法是直接用字母查找
                
                # 获取天然氨基酸的 Embedding (已经从预训练加载好了)
                natural_char = NATURAL_AA_ALPHABET[nat_idx]
                natural_emb = model.W_s.weight.data[EXTENDED_AA_TO_INDEX[natural_char]]
                
                # 找到对应的 N-甲基化字符
                # 我们需要反查：已知 nme_idx (in METHYL_AA_ALPHABET), 找 char
                from nmethyl.utils.nmethyl_config import METHYL_AA_ALPHABET
                if nme_idx < len(METHYL_AA_ALPHABET):
                    nme_char = METHYL_AA_ALPHABET[nme_idx]
                    if nme_char in EXTENDED_AA_TO_INDEX:
                        target_idx = EXTENDED_AA_TO_INDEX[nme_char]
                        # 复制权重：把 'C' 的特征给 'c'
                        model.W_s.weight.data[target_idx].copy_(natural_emb)
                        mapped_methyl += 1
                        
            print(f"  ✅ Initialized {mapped_methyl} N-methylated embeddings from parents.")

    # C. Head 初始化
    w_out = state_dict.get('W_out.weight', state_dict.get('module.W_out.weight'))
    b_out = state_dict.get('W_out.bias', state_dict.get('module.W_out.bias'))
    if w_out is not None:
        with torch.no_grad():
            model.W_out_base.weight.data.copy_(w_out[:20])
            model.W_out_base.bias.data.copy_(b_out[:20])
        print("  ✅ Base Head Initialized.")

    print(">>> Weight loading complete.\n")

# =============================================================================
# 3. 训练辅助 (保持不变)
# =============================================================================

class JSONLDataset(Dataset):
    def __init__(self, jsonl_file, augment=False):
        self.data = []
        self.augment = augment
        with open(jsonl_file, 'r') as f:
            for line in f: self.data.append(json.loads(line))
    def __len__(self): return len(self.data)
    def __getitem__(self, idx): return self.data[idx]

def get_weighted_sampler(dataset, oversample_weight=20.0):
    weights = []
    offset = len(NATURAL_AA_ALPHABET)
    methyl_indices = {k + offset for k in NMETHYL_TO_NATURAL_MAPPING.keys()}
    for item in dataset.data:
        has_methyl = False
        for key in item:
            if key.startswith('seq_chain_'):
                for char in item[key]:
                    if EXTENDED_AA_TO_INDEX.get(char, -1) in methyl_indices:
                        has_methyl = True; break
            if has_methyl: break
        weights.append(oversample_weight if has_methyl else 1.0)
    return WeightedRandomSampler(weights, len(weights), replacement=True)

def collate_fn(batch): return batch

def featurize_batch(batch, device):
    alphabet = EXTENDED_AA_ALPHABET
    B = len(batch)
    batch = [b for b in batch if 'seq' in b and len(b['seq']) > 0]
    if not batch: return None
    lengths = [len(b['seq']) for b in batch]
    L_max = max(lengths)
    X = np.zeros([B, L_max, 4, 3])
    S = np.zeros([B, L_max], dtype=np.int32)
    mask = np.zeros([B, L_max], dtype=np.float32)
    chain_M = np.zeros([B, L_max], dtype=np.float32)
    residue_idx = -100 * np.ones([B, L_max], dtype=np.int32)
    chain_encoding_all = np.zeros([B, L_max], dtype=np.int32)
    
    for i, b in enumerate(batch):
        all_chains = b.get('masked_list', []) + b.get('visible_list', [])
        l_p = 0
        for c_i, c_id in enumerate(all_chains):
            seq = b.get(f'seq_chain_{c_id}', '')
            N = np.array(b.get(f'N_chain_{c_id}', []))
            CA = np.array(b.get(f'CA_chain_{c_id}', []))
            C = np.array(b.get(f'C_chain_{c_id}', []))
            O = np.array(b.get(f'O_chain_{c_id}', []))
            l = min(len(seq), len(CA))
            if l == 0: continue
            X[i, l_p:l_p+l, 0, :] = N[:l]; X[i, l_p:l_p+l, 1, :] = CA[:l]
            X[i, l_p:l_p+l, 2, :] = C[:l]; X[i, l_p:l_p+l, 3, :] = O[:l]
            indices = []
            for aa in seq[:l]:
                idx = EXTENDED_AA_TO_INDEX.get(aa, EXTENDED_AA_TO_INDEX['X'])
                indices.append(idx)
            S[i, l_p:l_p+l] = indices
            if c_id in b.get('masked_list', []): chain_M[i, l_p:l_p+l] = 1.0
            residue_idx[i, l_p:l_p+l] = np.arange(l) + c_i * 100
            chain_encoding_all[i, l_p:l_p+l] = c_i
            l_p += l
            
    isnan = np.isnan(X); mask = np.isfinite(np.sum(X, (2,3))).astype(np.float32); X[isnan] = 0.
    return [torch.from_numpy(t).to(dtype=torch.long if t.dtype==np.int32 else torch.float32, device=device) for t in [X, S, mask, chain_M, residue_idx, chain_encoding_all]]

def calculate_final_loss(logits_base, logits_methyl, targets, mask, methyl_weight=4.0):
    mask_flat = mask.contiguous().view(-1).bool()
    if mask_flat.sum() == 0: return torch.tensor(0.0, device=logits_base.device), 0
    targets_flat = targets.contiguous().view(-1)[mask_flat]
    x_idx = EXTENDED_AA_TO_INDEX.get('X', -1)
    valid_pos = targets_flat != x_idx
    if valid_pos.sum() == 0: return torch.tensor(0.0, device=logits_base.device), 0
    targets_flat = targets_flat[valid_pos]
    logits_base = logits_base.contiguous().view(-1, 20)[mask_flat][valid_pos]
    logits_methyl = logits_methyl.contiguous().view(-1, 2)[mask_flat][valid_pos]
    
    base_targets = targets_flat.clone()
    offset = len(NATURAL_AA_ALPHABET)
    for methyl_rel, natural_idx in NMETHYL_TO_NATURAL_MAPPING.items():
        base_targets[base_targets == (methyl_rel + offset)] = natural_idx
    base_targets[base_targets >= len(NATURAL_AA_ALPHABET)] = -100
    
    methyl_targets = (targets_flat >= len(NATURAL_AA_ALPHABET)).long()
    loss_base = F.cross_entropy(logits_base, base_targets, label_smoothing=0.1, ignore_index=-100)
    loss_methyl = F.cross_entropy(logits_methyl, methyl_targets, weight=torch.tensor([1.0, 5.0], device=logits_base.device))
    return loss_base + methyl_weight * loss_methyl, base_targets.numel()

def final_evaluation(model, loader, device):
    model.eval()
    print("\n=== Final Evaluation ===")
    all_preds, all_targets = [], []
    nat_to_me_abs = {v: k + len(NATURAL_AA_ALPHABET) for k, v in NMETHYL_TO_NATURAL_MAPPING.items()}
    with torch.no_grad():
        for batch in loader:
            f = featurize_batch(batch, device)
            if f is None: continue
            lb, lm = model(*f)
            base = torch.argmax(lb, -1)
            is_me = torch.argmax(lm, -1)
            final = base.clone()
            for n_idx, m_idx in nat_to_me_abs.items():
                mask_update = (is_me == 1) & (base == n_idx)
                final[mask_update] = m_idx
            tgts = f[1].cpu().numpy().flatten()
            preds = final.cpu().numpy().flatten()
            mask = f[2].cpu().numpy().flatten().astype(bool)
            valid = mask & (tgts != EXTENDED_AA_TO_INDEX.get('X', -1))
            all_preds.extend(preds[valid])
            all_targets.extend(tgts[valid])
    all_preds, all_targets = np.array(all_preds), np.array(all_targets)
    if len(all_targets) == 0: print("No valid targets."); return
    acc = np.mean(all_preds == all_targets)
    natural_mask = all_targets < len(NATURAL_AA_ALPHABET)
    natural_acc = np.mean(all_targets[natural_mask] == all_preds[natural_mask]) if natural_mask.sum() > 0 else 0
    labels = sorted(np.unique(np.concatenate([all_targets, all_preds])).astype(int))
    names = [EXTENDED_AA_ALPHABET[i] for i in labels if i < len(EXTENDED_AA_ALPHABET)]
    print(classification_report(all_targets, all_preds, labels=labels, target_names=names, zero_division=0))
    print(f"Overall Accuracy: {acc:.4f}")
    print(f"Natural AA Recovery: {natural_acc:.4f}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrained_weights", type=str, required=True)
    parser.add_argument("--nmethyl_data", type=str, required=True)
    parser.add_argument("--test_data", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="./final_run_v10")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--methyl_loss_weight", type=float, default=4.0)
    parser.add_argument("--sampler_weight", type=float, default=20.0)
    parser.add_argument("--test_only", action="store_true")
    args = parser.parse_args()
    set_seed(42)
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = DecoupledProteinMPNN().to(device)

    if args.test_only:
        model_path = os.path.join(args.output_dir, "best_model.pt")
        print(f"Loading {model_path}...")
        model.load_state_dict(torch.load(model_path, map_location=device)['model_state_dict'])
        test_ds = JSONLDataset(args.test_data)
        test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)
        final_evaluation(model, test_loader, device)
        return

    # 加载与初始化 (Transfer Learning)
    advanced_load_weights(model, args.pretrained_weights, device)

    train_ds = JSONLDataset(args.nmethyl_data, augment=True)
    sampler = get_weighted_sampler(train_ds, oversample_weight=args.sampler_weight)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, sampler=sampler, collate_fn=collate_fn)
    test_ds = JSONLDataset(args.test_data)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)

    optimizer = torch.optim.AdamW([
        {'params': [p for n, p in model.named_parameters() if 'W_out' not in n and 'W_s' not in n], 'lr': 1e-4 * 0.1},
        {'params': [p for n, p in model.named_parameters() if 'W_out' in n or 'W_s' in n], 'lr': 1e-4}
    ], weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs * len(train_loader))

    print("Starting Training (v10 - Transfer Learning Complete)...")
    best_loss = float('inf')
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss, n_steps = 0, 0
        for batch in train_loader:
            f = featurize_batch(batch, device)
            if f is None: continue
            optimizer.zero_grad()
            lb, lm = model(*f)
            loss, valid = calculate_final_loss(lb, lm, f[1], f[2], methyl_weight=args.methyl_loss_weight)
            if valid > 0 and torch.isfinite(loss):
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                total_loss += loss.item()
                n_steps += 1
        avg_loss = total_loss / n_steps if n_steps > 0 else 0
        if epoch % 10 == 0: print(f"Epoch {epoch}: Loss = {avg_loss:.4f}")
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save({'model_state_dict': model.state_dict()}, os.path.join(args.output_dir, "best_model.pt"))

    model.load_state_dict(torch.load(os.path.join(args.output_dir, "best_model.pt"))['model_state_dict'])
    final_evaluation(model, test_loader, device)

if __name__ == "__main__":
    main()
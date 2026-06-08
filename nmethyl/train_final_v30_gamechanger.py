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
import copy
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torch.optim.lr_scheduler import OneCycleLR

# =============================================================================
# 0. 配置区域
# =============================================================================

NATURAL_RESIDUE_MAP = {
    'ALA': 'A', 'ARG': 'R', 'ASN': 'N', 'ASP': 'D', 'CYS': 'C',
    'GLN': 'Q', 'GLU': 'E', 'GLY': 'G', 'HIS': 'H', 'ILE': 'I',
    'LEU': 'L', 'LYS': 'K', 'MET': 'M', 'PHE': 'F', 'PRO': 'P',
    'SER': 'S', 'THR': 'T', 'TRP': 'W', 'TYR': 'Y', 'VAL': 'V',
}

NMETHYL_RESIDUE_MAP = {
    'MAA': 'a', 'SAR': 'g', 'MLE': 'l', 'IML': 'i', 'MVA': 'v',
    'MME': 'm', 'MEA': 'f', 'YNM': 'y', 'E9M': 'w', '5JP': 's',
    'SER': 's', 'NZC': 't', 'NCY': 'c', 'ZCA': 'n', 'GNC': 'q',
    'SOQ': 'd', 'EME': 'e', 'NMK': 'k', 'MMO': 'r', 'E9V': 'h',
}

NATURAL_AA_ALPHABET = "ACDEFGHIKLMNPQRSTVWY"
METHYL_AA_ALPHABET = "".join(sorted(list(set(NMETHYL_RESIDUE_MAP.values()))))
EXTENDED_AA_ALPHABET = NATURAL_AA_ALPHABET + METHYL_AA_ALPHABET + "X"

EXTENDED_AA_TO_INDEX = {aa: i for i, aa in enumerate(EXTENDED_AA_ALPHABET)}

NMETHYL_TO_NATURAL_MAPPING = {
    i: EXTENDED_AA_TO_INDEX[char.upper()]
    for i, char in enumerate(METHYL_AA_ALPHABET)
}

print(f"[Config] Loaded {len(NATURAL_AA_ALPHABET)} Natural + {len(METHYL_AA_ALPHABET)} Methyl Types.")
print(f"[Config] Total Vocab Size: {len(EXTENDED_AA_ALPHABET)}")

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def set_strict_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)
    print(f"[System] 🔒 Random Seed STRICTLY locked to: {seed}")

try:
    from model_utils import ProteinMPNN, gather_nodes, cat_neighbors_nodes
except ImportError as e:
    print(f"Error: Missing project files. {e}")
    sys.exit(1)

# =============================================================================
# 1. 模型定义 (回到 V14 经典架构 - 简单最美)
# =============================================================================

class DecoupledProteinMPNN(ProteinMPNN):
    def __init__(self, hidden_dim=128, augment_eps=0.1, **kwargs):
        vocab_size = len(EXTENDED_AA_ALPHABET)
        
        super().__init__(
            num_letters=vocab_size, 
            hidden_dim=hidden_dim, 
            vocab=vocab_size, 
            k_neighbors=48, 
            augment_eps=augment_eps, 
            **kwargs
        )
        self.W_s = nn.Embedding(vocab_size, hidden_dim)
        
        # Base Head
        self.W_out_base = nn.Linear(hidden_dim, len(NATURAL_AA_ALPHABET))
        
        # Methyl Head (MLP)
        self.W_out_methyl = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 2)
        )

    def forward(self, X, S, mask, chain_M, residue_idx, chain_encoding_all):
        E, E_idx = self.features(X, mask, residue_idx, chain_encoding_all)
        h_V = torch.zeros((E.shape[0], E.shape[1], E.shape[-1]), device=E.device)
        h_E = self.W_e(E)

        mask_attend = gather_nodes(mask.unsqueeze(-1),  E_idx).squeeze(-1)
        mask_attend = mask.unsqueeze(-1) * mask_attend
        for layer in self.encoder_layers:
            h_V, h_E = torch.utils.checkpoint.checkpoint(layer, h_V, h_E, E_idx, mask, mask_attend)

        # 这里的 S 已经被外面替换成了 S_blind (全X)，所以 Embedding 也是未知的
        h_S = self.W_s(S)
        h_ES = cat_neighbors_nodes(h_S, h_E, E_idx)
        h_EX_encoder = cat_neighbors_nodes(torch.zeros_like(h_S), h_E, E_idx)
        h_EXV_encoder = cat_neighbors_nodes(h_V, h_EX_encoder, E_idx)

        chain_M = chain_M * mask
        # 训练时依然加噪声，增加鲁棒性
        if self.training:
            decoding_order = torch.argsort((chain_M + 0.0001) * (torch.abs(torch.randn(chain_M.shape, device=X.device))))
        else:
            decoding_order = torch.argsort(chain_M + 0.0001)

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
# 2. 权重加载
# =============================================================================
def advanced_load_weights(model, pretrained_path, device):
    print(f"\n>>> [Loader] Opening {pretrained_path}...")
    ckpt = torch.load(pretrained_path, map_location=device)
    state_dict = ckpt.get('model_state_dict', ckpt)
    model_state = model.state_dict()
    
    load_dict = {}
    for k, v in state_dict.items():
        clean_k = k.replace("module.", "")
        if clean_k in model_state and v.shape == model_state[clean_k].shape:
            load_dict[clean_k] = v
    model.load_state_dict(load_dict, strict=False)
    
    ws_file = state_dict.get('W_s.weight', state_dict.get('module.W_s.weight'))
    if ws_file is not None:
        STD = 'ACDEFGHIKLMNPQRSTVWYX'
        with torch.no_grad():
            for i, char in enumerate(STD):
                if char in EXTENDED_AA_TO_INDEX:
                    target_idx = EXTENDED_AA_TO_INDEX[char]
                    if i < ws_file.shape[0]:
                        model.W_s.weight.data[target_idx].copy_(ws_file[i])
            for nme_rel_idx, nat_idx in NMETHYL_TO_NATURAL_MAPPING.items():
                nme_char = METHYL_AA_ALPHABET[nme_rel_idx]
                target_idx = EXTENDED_AA_TO_INDEX[nme_char]
                parent_emb = model.W_s.weight.data[nat_idx]
                noise = torch.randn_like(parent_emb) * 0.01 
                model.W_s.weight.data[target_idx].copy_(parent_emb + noise)

    w_out = state_dict.get('W_out.weight', state_dict.get('module.W_out.weight'))
    b_out = state_dict.get('W_out.bias', state_dict.get('module.W_out.bias'))
    if w_out is not None:
        with torch.no_grad():
            model.W_out_base.weight.data.copy_(w_out[:20])
            model.W_out_base.bias.data.copy_(b_out[:20])
    
    print("✅ Weights Loaded.")

# =============================================================================
# 3. 损失函数
# =============================================================================
def calculate_loss(logits_base, logits_methyl, targets, mask, methyl_weight=2.0):
    mask_flat = mask.contiguous().view(-1).bool()
    if mask_flat.sum() == 0: return torch.tensor(0.0, device=logits_base.device, requires_grad=True), 0
    
    targets_flat = targets.contiguous().view(-1)[mask_flat]
    x_idx = EXTENDED_AA_TO_INDEX.get('X', -1)
    valid_pos = targets_flat != x_idx
    if valid_pos.sum() == 0: return torch.tensor(0.0, device=logits_base.device, requires_grad=True), 0
    
    targets_flat = targets_flat[valid_pos]
    
    logits_base_flat = logits_base.contiguous().view(-1, 20)[mask_flat][valid_pos]
    logits_methyl_flat = logits_methyl.contiguous().view(-1, 2)[mask_flat][valid_pos]
    
    base_targets = targets_flat.clone()
    offset = len(NATURAL_AA_ALPHABET)
    for nme_rel_idx, nat_idx in NMETHYL_TO_NATURAL_MAPPING.items():
        actual_nme_idx = offset + nme_rel_idx
        base_targets[base_targets == actual_nme_idx] = nat_idx
    base_targets[base_targets >= len(NATURAL_AA_ALPHABET)] = -100
    
    methyl_targets = (targets_flat >= len(NATURAL_AA_ALPHABET)).long()
    
    loss_base = F.cross_entropy(logits_base_flat, base_targets, ignore_index=-100)
    loss_methyl = F.cross_entropy(logits_methyl_flat, methyl_targets, weight=torch.tensor([1.0, methyl_weight], device=logits_base.device))
    
    return loss_base + methyl_weight * loss_methyl, base_targets.numel()

# =============================================================================
# 4. 数据处理
# =============================================================================
class JSONLDataset(Dataset):
    def __init__(self, jsonl_file):
        self.data = []
        with open(jsonl_file, 'r') as f:
            for line in f: self.data.append(json.loads(line))
    def __len__(self): return len(self.data)
    def __getitem__(self, idx): return self.data[idx]

def get_weighted_sampler(dataset, oversample_weight=10.0):
    weights = []
    offset = len(NATURAL_AA_ALPHABET)
    methyl_indices = {i + offset for i in range(len(METHYL_AA_ALPHABET))}
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
    B = len(batch)
    batch = [b for b in batch if 'seq_chain_A' in b and len(b['seq_chain_A']) > 0]
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

# =============================================================================
# 5. 验证与报告
# =============================================================================
def validate(model, loader, device):
    model.eval()
    all_preds_combined, all_targets_combined = [], []
    all_preds_methyl, all_targets_methyl = [], []
    all_preds_base, all_targets_base = [], []
    
    nat_to_me_abs = {}
    offset = len(NATURAL_AA_ALPHABET)
    for nme_rel, char in enumerate(METHYL_AA_ALPHABET):
        nat_idx = NMETHYL_TO_NATURAL_MAPPING[nme_rel]
        me_idx = nme_rel + offset
        nat_to_me_abs[nat_idx] = me_idx

    with torch.no_grad():
        for batch in loader:
            f = featurize_batch(batch, device)
            if f is None: continue
            X, S, mask, chain_M, residue_idx, chain_encoding_all = f
            
            # 验证也是盲测
            S_blind = torch.full_like(S, EXTENDED_AA_TO_INDEX['X'])
            
            lb, lm = model(X, S_blind, mask, chain_M, residue_idx, chain_encoding_all)
            
            pred_base_idx = torch.argmax(lb, -1) 
            probs_methyl = F.softmax(lm, dim=-1)[:, :, 1]
            pred_is_methyl = (probs_methyl > 0.5).long()
            
            final_pred = pred_base_idx.clone()
            for n_idx, m_idx in nat_to_me_abs.items():
                mask_update = (pred_is_methyl == 1) & (pred_base_idx == n_idx)
                final_pred[mask_update] = m_idx
            
            targets = S.cpu().numpy().flatten()
            preds_final = final_pred.cpu().numpy().flatten()
            preds_methyl = pred_is_methyl.cpu().numpy().flatten()
            preds_base_raw = pred_base_idx.cpu().numpy().flatten()
            
            mask_flat = mask.cpu().numpy().flatten().astype(bool)
            valid = mask_flat & (targets != EXTENDED_AA_TO_INDEX.get('X', -1))
            
            all_preds_combined.extend(preds_final[valid])
            all_targets_combined.extend(targets[valid])
            
            methyl_gt = (targets[valid] >= offset).astype(int)
            all_preds_methyl.extend(preds_methyl[valid])
            all_targets_methyl.extend(methyl_gt)
            
            base_gt = targets[valid].copy()
            for m_rel, n_idx in NMETHYL_TO_NATURAL_MAPPING.items():
                base_gt[base_gt == (m_rel + offset)] = n_idx
            all_preds_base.extend(preds_base_raw[valid])
            all_targets_base.extend(base_gt)

    if len(all_targets_combined) == 0: return 0, 0, 0, 0

    total_acc = np.mean(np.array(all_preds_combined) == np.array(all_targets_combined))
    methyl_acc = np.mean(np.array(all_preds_methyl) == np.array(all_targets_methyl))
    base_acc = np.mean(np.array(all_preds_base) == np.array(all_targets_base))
    
    tp = np.sum((np.array(all_preds_methyl) == 1) & (np.array(all_targets_methyl) == 1))
    fn = np.sum((np.array(all_preds_methyl) == 0) & (np.array(all_targets_methyl) == 1))
    recall = tp / (tp + fn + 1e-6)
    
    return base_acc, methyl_acc, total_acc, recall

# =============================================================================
# 6. 主循环 (V37: Blind Monk Mode)
# =============================================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrained_weights", type=str, required=True)
    parser.add_argument("--nmethyl_data", type=str, required=True)
    parser.add_argument("--test_data", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="./run_v37_blind_monk")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--methyl_loss_weight", type=float, default=2.0) 
    parser.add_argument("--sampler_weight", type=float, default=10.0)
    args = parser.parse_args()
    
    set_strict_seed(42)
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = DecoupledProteinMPNN(augment_eps=0.1).to(device) 
    advanced_load_weights(model, args.pretrained_weights, device)
    
    # 全解冻
    for param in model.parameters(): param.requires_grad = True

    train_ds = JSONLDataset(args.nmethyl_data)
    sampler = get_weighted_sampler(train_ds, oversample_weight=args.sampler_weight)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, sampler=sampler, collate_fn=collate_fn)
    test_ds = JSONLDataset(args.test_data)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)

    # 学习率给足
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    scheduler = OneCycleLR(optimizer, max_lr=1e-3, epochs=args.epochs, steps_per_epoch=len(train_loader))

    print("Starting Training (V37 - Blind Monk Mode)...")
    best_total_acc = 0.0
    
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss, n_steps = 0, 0
        for batch in train_loader:
            f = featurize_batch(batch, device)
            if f is None: continue
            optimizer.zero_grad()
            X, S, mask, chain_M, residue_idx, chain_encoding_all = f
            
            # 🔥🔥🔥 V37 核心改动 🔥🔥🔥
            # 训练时，强行把输入 S 变成全 X (Blind)。
            # 这样模型就不能依赖 S 也就是答案，必须学会看结构 X。
            S_blind_input = torch.full_like(S, EXTENDED_AA_TO_INDEX['X'])
            
            # 输入是 Blind，但计算 Loss 时用的 targets 依然是真实的 S
            lb, lm = model(X, S_blind_input, mask, chain_M, residue_idx, chain_encoding_all)
            
            loss, valid = calculate_loss(lb, lm, S, mask, methyl_weight=args.methyl_loss_weight)
            
            if valid > 0 and torch.isfinite(loss):
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                total_loss += loss.item()
                n_steps += 1
        
        avg_loss = total_loss / n_steps if n_steps > 0 else 0
        
        if epoch % 5 == 0: 
            print(f"Epoch {epoch}: Loss = {avg_loss:.4f}")
            base_acc, methyl_acc, total_acc, recall = validate(model, test_loader, device)
            
            print(f"   [Val] Base: {base_acc*100:.2f}% | Methyl: {methyl_acc*100:.2f}% | Total: {total_acc*100:.2f}% | Recall: {recall*100:.2f}%")
            
            if total_acc > best_total_acc:
                best_total_acc = total_acc
                torch.save({'model_state_dict': model.state_dict()}, os.path.join(args.output_dir, "best_model_v37.pt"))
                print(f"   🌟 BEST MODEL SAVED! (Total: {best_total_acc*100:.2f}%)")

if __name__ == "__main__":
    main()
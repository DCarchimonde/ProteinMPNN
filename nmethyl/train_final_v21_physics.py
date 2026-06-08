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

# --- 系统路径 ---
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# =============================================================================
# 0. 随机种子 & 物理常数
# =============================================================================
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

# 物理铁律：氢键距离阈值
HBOND_MAX_DIST = 3.5 

try:
    from model_utils import ProteinMPNN, gather_nodes, cat_neighbors_nodes
    from nmethyl.utils.nmethyl_config import (
        EXTENDED_AA_ALPHABET, NATURAL_AA_ALPHABET,
        NMETHYL_TO_NATURAL_MAPPING, EXTENDED_AA_TO_INDEX, METHYL_AA_ALPHABET
    )
except ImportError as e:
    print(f"Error: Missing project files. {e}")
    sys.exit(1)

# =============================================================================
# 1. 物理辅助函数
# =============================================================================
def compute_hbond_mask(X, mask):
    B, L, _, _ = X.shape
    N_atoms = X[:, :, 0, :] 
    O_atoms = X[:, :, 3, :]
    dists = torch.cdist(N_atoms, O_atoms)
    diag_mask = torch.eye(L, device=X.device).unsqueeze(0).expand(B, -1, -1)
    dists = dists.masked_fill(diag_mask > 0.5, 999.9)
    min_dist_per_N, _ = torch.min(dists, dim=-1)
    has_hbond = min_dist_per_N < HBOND_MAX_DIST
    return has_hbond & (mask > 0.5)

# =============================================================================
# 2. 模型定义 (V23: Decoupled Binary Detector)
# =============================================================================
class DecoupledBinaryProteinMPNN(ProteinMPNN):
    def __init__(self, hidden_dim=128, augment_eps=0.1, **kwargs):
        super().__init__(
            num_letters=21, 
            hidden_dim=hidden_dim, 
            vocab=21, 
            k_neighbors=48, 
            augment_eps=augment_eps, 
            **kwargs
        )
        self.W_s = nn.Embedding(len(EXTENDED_AA_ALPHABET), hidden_dim)
        
        # HEAD 1: Base Identity (你是谁？) -> 20分类
        self.W_out_base = nn.Linear(hidden_dim, len(NATURAL_AA_ALPHABET))
        
        # HEAD 2: Global Methyl Detector (有没有甲基？) -> 1分类 (Binary)
        # 这个头不关心你是A还是S，只关心 N 原子的几何特征
        self.W_out_binary = nn.Linear(hidden_dim, 1)

    def forward(self, X, S, mask, chain_M, residue_idx, chain_encoding_all):
        # --- Standard Encoder / Decoder ---
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
        decoding_order = torch.argsort(chain_M + 0.0001)
        if self.training:
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
        
        # --- V23 Outputs ---
        logits_base = self.W_out_base(h_V)     # [B, L, 20]
        logits_binary = self.W_out_binary(h_V) # [B, L, 1]
        
        return logits_base, logits_binary

# =============================================================================
# 3. 智能初始化
# =============================================================================
def smart_load_weights(model, pretrained_path, device):
    print(f"\n>>> [V23 Init] Loading {pretrained_path}...")
    ckpt = torch.load(pretrained_path, map_location=device)
    state_dict = ckpt.get('model_state_dict', ckpt)
    model_state = model.state_dict()
    load_dict = {}
    for k, v in state_dict.items():
        clean_k = k.replace("module.", "")
        if clean_k == "W_out_base.weight" and "W_out.weight" in state_dict:
             load_dict[clean_k] = state_dict["W_out.weight"][:20, :]
             continue
        if clean_k == "W_out_base.bias" and "W_out.bias" in state_dict:
             load_dict[clean_k] = state_dict["W_out.bias"][:20]
             continue
        if clean_k in model_state and v.shape == model_state[clean_k].shape:
            load_dict[clean_k] = v
    model.load_state_dict(load_dict, strict=False)
    
    # Init Embedding
    ws_file = state_dict.get('W_s.weight', state_dict.get('module.W_s.weight'))
    if ws_file is not None:
        with torch.no_grad():
            for i, char in enumerate(NATURAL_AA_ALPHABET):
                if char in EXTENDED_AA_TO_INDEX:
                    model.W_s.weight.data[EXTENDED_AA_TO_INDEX[char]].copy_(ws_file[i])
            for nme_idx, nat_idx in NMETHYL_TO_NATURAL_MAPPING.items():
                if nme_idx < len(METHYL_AA_ALPHABET):
                    nme_char = METHYL_AA_ALPHABET[nme_idx]
                    nat_char = NATURAL_AA_ALPHABET[nat_idx]
                    nat_emb_idx = EXTENDED_AA_TO_INDEX[nat_char]
                    nme_emb_idx = EXTENDED_AA_TO_INDEX[nme_char]
                    parent_emb = model.W_s.weight.data[nat_emb_idx]
                    model.W_s.weight.data[nme_emb_idx].copy_(parent_emb + torch.randn_like(parent_emb)*0.01)
    print("✅ V23 Decoupled Binary System Ready.")

# =============================================================================
# 4. 训练 Loss (Binary Decoupling)
# =============================================================================
def calculate_loss_v23(logits_base, logits_binary, targets, mask):
    mask_flat = mask.contiguous().view(-1).bool()
    if mask_flat.sum() == 0: return torch.tensor(0.0, device=logits_base.device), 0
    targets_flat = targets.contiguous().view(-1)[mask_flat]
    valid_pos = targets_flat != EXTENDED_AA_TO_INDEX.get('X', -1)
    if valid_pos.sum() == 0: return torch.tensor(0.0, device=logits_base.device), 0
    targets_flat = targets_flat[valid_pos]
    
    logits_base = logits_base.contiguous().view(-1, 20)[mask_flat][valid_pos]
    logits_binary = logits_binary.contiguous().view(-1)[mask_flat][valid_pos]
    
    # 1. Base Labels (Reduce Methyl to Natural)
    base_targets = targets_flat.clone()
    offset = len(NATURAL_AA_ALPHABET)
    for methyl_rel, natural_idx in NMETHYL_TO_NATURAL_MAPPING.items():
        base_targets[base_targets == (methyl_rel + offset)] = natural_idx
    
    # 2. Binary Labels (Is it Methylated? 0 or 1)
    # 凡是索引 >= 20 的，都是甲基化 (Label 1)
    # 凡是索引 < 20 的，都是天然 (Label 0)
    binary_targets = (targets_flat >= len(NATURAL_AA_ALPHABET)).float()
    
    # --- Loss Calculation ---
    
    # Loss 1: Identity (Who are you?)
    base_targets_clamped = base_targets.clone()
    base_targets_clamped[base_targets_clamped >= 20] = -100
    loss_base = F.cross_entropy(logits_base, base_targets_clamped, ignore_index=-100)
    
    # Loss 2: Binary Methylation (Are you weird?)
    # 这里加权 6.0，让模型高度重视甲基化样本
    loss_binary = F.binary_cross_entropy_with_logits(
        logits_binary, 
        binary_targets, 
        pos_weight=torch.tensor([6.0], device=logits_base.device)
    )

    return loss_base + 3.0 * loss_binary, base_targets.numel()

# =============================================================================
# 5. 数据处理
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
# 6. 验证函数 (V23: 独立判决 + 物理过滤)
# =============================================================================
def validate_and_report(model, loader, device, epoch_num):
    model.eval()
    print(f"\n🔍 [Epoch {epoch_num}] Binary Council Diagnosing...")
    
    all_targets, all_preds = [], []
    nat_idx_to_methyl_idx = {n: m + len(NATURAL_AA_ALPHABET) for m, n in NMETHYL_TO_NATURAL_MAPPING.items()}
    methyl_idx_to_nat_idx = {m + len(NATURAL_AA_ALPHABET): n for m, n in NMETHYL_TO_NATURAL_MAPPING.items()}

    with torch.no_grad():
        for batch in loader:
            f = featurize_batch(batch, device)
            if f is None: continue
            X, S, mask, chain_M, residue_idx, chain_encoding_all = f
            
            # 1. 模型预测
            l_base, l_binary = model(X, S, mask, chain_M, residue_idx, chain_encoding_all)
            
            # Base 决定身份 (你是A/C/D...)
            pred_base_idx = torch.argmax(l_base, -1) # [B, L]
            
            # Binary 决定性质 (你是天然还是甲基化?)
            # 只要 > 0.5 (logit > 0)，就认为是甲基化
            is_methyl_raw = (torch.sigmoid(l_binary.squeeze(-1)) > 0.5) # [B, L]
            
            # 2. 物理过滤 (如果已经有氢键，绝对不可能是甲基化)
            has_h_bond = compute_hbond_mask(X, mask)
            is_methyl_final = is_methyl_raw & (~has_h_bond)
            
            # 3. 组合最终结果
            final_pred = pred_base_idx.clone()
            
            # 遍历每一个位置
            # 如果 (Binary 说 Yes) AND (没有氢键) AND (Base 预测的氨基酸有对应的甲基化版本)
            # 那么就把它改成甲基化 ID
            
            # 我们需要把 pred_base_idx 映射到 methyl_idx
            # 比如 Base 预测是 A (idx 0)，如果有甲基化信号，就改成 Methyl-A (idx 20)
            
            # 构建一个映射 Tensor
            B, L = final_pred.shape
            methyl_map = torch.zeros_like(final_pred) # 默认 0
            
            for nat_idx, met_idx in nat_idx_to_methyl_idx.items():
                # 找到 Base 预测为 nat_idx 的位置
                mask_nat = (pred_base_idx == nat_idx)
                # 在这些位置填入对应的 met_idx
                methyl_map[mask_nat] = met_idx
                
            # 只有在有效映射的位置 (有些氨基酸可能没甲基化版) 且 判定为甲基化 时才替换
            # 注意：methyl_map 为 0 的地方意味着没有映射（或者映射到 A，需小心）
            # 只有当 pred_base_idx 在 nat_idx_to_methyl_idx 的 keys 里时才有效
            
            can_be_methylated = torch.zeros_like(pred_base_idx, dtype=torch.bool)
            for nat_idx in nat_idx_to_methyl_idx.keys():
                can_be_methylated |= (pred_base_idx == nat_idx)
            
            update_mask = is_methyl_final & can_be_methylated
            
            # 执行替换
            final_pred[update_mask] = methyl_map[update_mask]
            
            # Collect
            tgts = S.cpu().numpy().flatten()
            preds = final_pred.cpu().numpy().flatten()
            mask_flat = mask.cpu().numpy().flatten().astype(bool)
            valid = mask_flat & (tgts != EXTENDED_AA_TO_INDEX.get('X', -1))
            all_targets.extend(tgts[valid])
            all_preds.extend(preds[valid])

    all_targets = np.array(all_targets)
    all_preds = np.array(all_preds)

    methyl_mask = all_targets >= len(NATURAL_AA_ALPHABET)
    natural_mask = all_targets < len(NATURAL_AA_ALPHABET)
    
    methyl_correct = np.sum(all_preds[methyl_mask] == all_targets[methyl_mask])
    methyl_total = np.sum(methyl_mask)
    methyl_recall = methyl_correct / methyl_total if methyl_total > 0 else 0.0
    
    nat_preds_as_methyl = np.sum(all_preds[natural_mask] >= len(NATURAL_AA_ALPHABET))
    nat_total = np.sum(natural_mask)
    false_positive_rate = nat_preds_as_methyl / nat_total if nat_total > 0 else 0.0

    print(f"\n📊 --- REPORT CARD (Epoch {epoch_num}) ---")
    print(f"✅ Methyl Recall:    {methyl_recall*100:.2f}%  ({methyl_correct}/{methyl_total})")
    print(f"⚠️ False Alarm Rate: {false_positive_rate*100:.2f}%")
    
    print("\n🧐 Detailed Binary-Decoupled Report:")
    print(f"{'Methyl AA':<15} {'Count':<8} {'Recall':<10} {'Miss->Nat':<10} {'Miss->Other'}")
    print("-" * 60)
    for m_char in METHYL_AA_ALPHABET:
        m_idx = EXTENDED_AA_TO_INDEX[m_char]
        if m_idx not in methyl_idx_to_nat_idx: continue
        n_idx = methyl_idx_to_nat_idx[m_idx]
        indices = np.where(all_targets == m_idx)[0]
        count = len(indices)
        if count == 0: continue
        correct = np.sum(all_preds[indices] == m_idx)
        mis_as_nat = np.sum(all_preds[indices] == n_idx)
        mis_other = count - correct - mis_as_nat
        rec = correct / count * 100
        print(f"{m_char:<15} {count:<8} {rec:5.1f}%     {mis_as_nat:<10} {mis_other}")
    print("-" * 60)
    
    base_acc = np.mean(all_preds[natural_mask] == all_targets[natural_mask]) if nat_total > 0 else 0
    composite_score = (0.3 * base_acc) + (0.7 * methyl_recall)
    print(f"🏆 Binary Score: {composite_score:.4f} (BaseAcc: {base_acc:.4f}, MeRec: {methyl_recall:.4f})")
    return composite_score

# =============================================================================
# 7. 主循环
# =============================================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrained_weights", type=str, required=True)
    parser.add_argument("--nmethyl_data", type=str, required=True)
    parser.add_argument("--test_data", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="./run_v23_binary")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=8)
    args = parser.parse_args()
    
    set_strict_seed(42)
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = DecoupledBinaryProteinMPNN(augment_eps=0.1).to(device)
    smart_load_weights(model, args.pretrained_weights, device)

    train_ds = JSONLDataset(args.nmethyl_data)
    sampler = get_weighted_sampler(train_ds, oversample_weight=10.0)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, sampler=sampler, collate_fn=collate_fn)
    test_ds = JSONLDataset(args.test_data)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=5e-4, epochs=args.epochs, steps_per_epoch=len(train_loader), pct_start=0.2)

    print("🚀 Starting V23: Decoupled Binary Detector...")
    best_score = 0.0
    
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0
        steps = 0
        for batch in train_loader:
            f = featurize_batch(batch, device)
            if f is None: continue
            optimizer.zero_grad()
            X, S, mask, chain_M, residue_idx, chain_encoding_all = f
            l_base, l_binary = model(X, S, mask, chain_M, residue_idx, chain_encoding_all)
            
            loss, valid = calculate_loss_v23(l_base, l_binary, S, mask)
            
            if valid > 0:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                total_loss += loss.item()
                steps += 1
        
        avg_loss = total_loss / steps if steps > 0 else 0
        print(f"Epoch {epoch}: Loss = {avg_loss:.4f}", end="\r")

        if epoch % 5 == 0:
            current_score = validate_and_report(model, test_loader, device, epoch)
            if current_score > best_score:
                best_score = current_score
                save_path = os.path.join(args.output_dir, "best_model_v23.pt")
                torch.save({'model_state_dict': model.state_dict()}, save_path)
                print(f"🌟 New Best Model Saved! (Score: {best_score:.4f})")

if __name__ == "__main__":
    main()
import argparse
import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import json
import numpy as np
from torch.utils.data import DataLoader, Dataset
import torch.utils.checkpoint

# 解决多线程警告
os.environ["OMP_NUM_THREADS"] = "1"

# --- 系统路径 ---
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

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
# 1. 模型定义 (完美保留你原先的 V28 Robust 架构)
# =============================================================================
class RobustHierarchicalProteinMPNN(ProteinMPNN):
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
        
        # V28: MLP Head + LayerNorm
        self.W_out_base = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, len(NATURAL_AA_ALPHABET))
        )
        
        self.experts = nn.ModuleList([
            nn.Linear(hidden_dim, 1) for _ in range(len(NATURAL_AA_ALPHABET))
        ])

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
        expert_outputs = [expert(h_V) for expert in self.experts]
        logits_experts = torch.cat(expert_outputs, dim=-1)
        
        return logits_base, logits_experts

# =============================================================================
# 2. 数据处理工具
# =============================================================================
class JSONLDataset(Dataset):
    def __init__(self, jsonl_file):
        self.data = []
        with open(jsonl_file, 'r') as f:
            for line in f: self.data.append(json.loads(line))
    def __len__(self): return len(self.data)
    def __getitem__(self, idx): return self.data[idx]

def collate_fn(batch): return batch

def featurize_batch(batch, device):
    B = len(batch)
    batch = [b for b in batch if 'seq_chain_A' in b and len(b['seq_chain_A']) > 0]
    if not batch: return None
    lengths = [len(b['seq']) for b in batch] if 'seq' in batch[0] else [len(b['seq_chain_A']) for b in batch]
    L_max = max(lengths)
    X = np.zeros([B, L_max, 4, 3])
    S = np.zeros([B, L_max], dtype=np.int32)
    mask = np.zeros([B, L_max], dtype=np.float32)
    chain_M = np.zeros([B, L_max], dtype=np.float32)
    residue_idx = -100 * np.ones([B, L_max], dtype=np.int32)
    chain_encoding_all = np.zeros([B, L_max], dtype=np.int32)
    
    for i, b in enumerate(batch):
        all_chains = b.get('masked_list', []) + b.get('visible_list', [])
        if not all_chains: all_chains = ['A'] # Fallback
        
        l_p = 0
        for c_i, c_id in enumerate(all_chains):
            seq = b.get(f'seq_chain_{c_id}', '')
            
            # --- 核心物理修复保留 ---
            N = np.array(b.get(f'N_chain_{c_id}', []))
            CA = np.array(b.get(f'CA_chain_{c_id}', []))
            C = np.array(b.get(f'C_chain_{c_id}', []))
            O = np.array(b.get(f'O_chain_{c_id}', []))
            l = min(len(seq), len(CA))
            if l == 0: continue
            
            # 如果坐标反了，换回来
            if len(N) > 0 and len(CA) > 0 and len(O) > 0:
                dist_n_ca = np.linalg.norm(N[:1] - CA[:1])
                dist_n_o = np.linalg.norm(N[:1] - O[:1])
                if dist_n_o < dist_n_ca and dist_n_o < 1.6:
                    CA, O = O, CA
                    
            X[i, l_p:l_p+l, 0, :] = N[:l]
            X[i, l_p:l_p+l, 1, :] = CA[:l]
            X[i, l_p:l_p+l, 2, :] = C[:l]
            X[i, l_p:l_p+l, 3, :] = O[:l]
            # ----------------------
            
            indices = [EXTENDED_AA_TO_INDEX.get(aa, EXTENDED_AA_TO_INDEX.get('X', 40)) for aa in seq[:l]]
            S[i, l_p:l_p+l] = indices
            
            if c_id in b.get('masked_list', []): 
                chain_M[i, l_p:l_p+l] = 1.0
                
            residue_idx[i, l_p:l_p+l] = np.arange(l) + c_i * 100
            chain_encoding_all[i, l_p:l_p+l] = c_i
            l_p += l
            
    isnan = np.isnan(X)
    mask = np.isfinite(np.sum(X, (2,3))).astype(np.float32)
    X[isnan] = 0.
    return [torch.from_numpy(t).to(dtype=torch.long if t.dtype==np.int32 else torch.float32, device=device) for t in [X, S, mask, chain_M, residue_idx, chain_encoding_all]]

# =============================================================================
# 3. 阈值扫描逻辑
# =============================================================================
def sweep_thresholds(model, loader, device, thresholds):
    model.eval()
    print("🧪 Scanning Thresholds on Test Set...")
    
    all_targets_comb = []
    all_preds_base_raw = []
    all_probs_experts = [] # 记录专家头给出的概率
    
    methyl_idx_to_nat_idx = {m + len(NATURAL_AA_ALPHABET): n for m, n in NMETHYL_TO_NATURAL_MAPPING.items()}
    x_idx = EXTENDED_AA_TO_INDEX.get('X', 40)

    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            f = featurize_batch(batch, device)
            if f is None: continue
            X, S, mask, chain_M, residue_idx, chain_encoding_all = f
            
            l_base, l_experts = model(X, S, mask, chain_M, residue_idx, chain_encoding_all)
            
            # 1. Base Prediction
            pred_base_idx = torch.argmax(l_base, -1)
            
            # 2. 获取每个预测位置对应的专家给出的"是否甲基化"的概率 (Sigmoid)
            expert_logit = torch.gather(l_experts, -1, pred_base_idx.unsqueeze(-1)).squeeze(-1)
            prob_methyl = torch.sigmoid(expert_logit)
            
            # Collect Data
            tgts = S.cpu().numpy().flatten()
            pb_raw = pred_base_idx.cpu().numpy().flatten()
            p_methyl = prob_methyl.cpu().numpy().flatten()
            mask_flat = mask.cpu().numpy().flatten().astype(bool)
            
            # 只统计有效且非X的位置
            valid = mask_flat & (tgts != x_idx)
            
            all_targets_comb.extend(tgts[valid])
            all_preds_base_raw.extend(pb_raw[valid])
            all_probs_experts.extend(p_methyl[valid])

    all_targets_comb = np.array(all_targets_comb)
    all_preds_base_raw = np.array(all_preds_base_raw)
    all_probs_experts = np.array(all_probs_experts)
    
    # 计算真实的甲基化比例
    methyl_mask = all_targets_comb >= len(NATURAL_AA_ALPHABET)
    true_methyl_ratio = np.mean(methyl_mask) * 100
    
    print("\n" + "="*75)
    print(f"📊 [Ground Truth] Real Methylation Ratio in Test Set: {true_methyl_ratio:.2f}%")
    print("-" * 75)
    print(f"{'Threshold':<10} | {'Predicted Methyl Ratio':<25} | {'Total End-to-End Acc':<15}")
    print("-" * 75)
    
    best_acc = 0
    best_thresh = 0.5
    
    # 扫描阈值
    for thresh in thresholds:
        # 判断是否加甲基
        pred_is_methyl = (all_probs_experts > thresh).astype(int)
        
        # 组装最终结果
        final_pred = all_preds_base_raw.copy()
        for m_abs_idx, n_idx in methyl_idx_to_nat_idx.items():
            mask_update = (pred_is_methyl == 1) & (all_preds_base_raw == n_idx)
            final_pred[mask_update] = m_abs_idx
            
        pred_methyl_ratio = np.mean(pred_is_methyl) * 100
        total_acc = np.mean(final_pred == all_targets_comb) * 100
        
        marker = "⭐" if abs(pred_methyl_ratio - true_methyl_ratio) < 10 else ""
        print(f"Thr = {thresh:<4.2f} | Pred Ratio = {pred_methyl_ratio:>5.2f}% {marker:<10} | Total Acc = {total_acc:.2f}%")
        
        if total_acc > best_acc:
            best_acc = total_acc
            best_thresh = thresh
            
    print("-" * 75)
    print(f"💡 Suggestion: Pick the threshold where 'Pred Ratio' is close to ~{true_methyl_ratio:.1f}%.")

# =============================================================================
# 4. Main Execution
# =============================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--test_data", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=16)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = RobustHierarchicalProteinMPNN(augment_eps=0.0).to(device) 
    
    # 直接用你之前的安全加载方式
    checkpoint = torch.load(args.model_path, map_location=device)
    state_dict = checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint
    model.load_state_dict(state_dict)
    
    test_ds = JSONLDataset(args.test_data)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)

    # 扫描阈值区间
    thresholds_to_test = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.98, 0.99]
    sweep_thresholds(model, test_loader, device, thresholds_to_test)
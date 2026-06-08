import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint  # <--- 关键修复：显式导入 checkpoint
import json
import argparse
import sys
import os
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import classification_report

# =============================================================================
# 0. 配置与常量
# =============================================================================
NATURAL_AA_ALPHABET = "ACDEFGHIKLMNPQRSTVWY"
NMETHYL_RESIDUE_MAP = {
    'MAA': 'a', 'SAR': 'g', 'MLE': 'l', 'IML': 'i', 'MVA': 'v',
    'MME': 'm', 'MEA': 'f', 'YNM': 'y', 'E9M': 'w', '5JP': 's',
    'SER': 's', 'NZC': 't', 'NCY': 'c', 'ZCA': 'n', 'GNC': 'q',
    'SOQ': 'd', 'EME': 'e', 'NMK': 'k', 'MMO': 'r', 'E9V': 'h',
}
METHYL_AA_ALPHABET = "".join(sorted(list(set(NMETHYL_RESIDUE_MAP.values()))))
EXTENDED_AA_ALPHABET = NATURAL_AA_ALPHABET + METHYL_AA_ALPHABET + "X"
EXTENDED_AA_TO_INDEX = {aa: i for i, aa in enumerate(EXTENDED_AA_ALPHABET)}
NMETHYL_TO_NATURAL_MAPPING = {i: EXTENDED_AA_TO_INDEX[char.upper()] for i, char in enumerate(METHYL_AA_ALPHABET)}

# 尝试导入依赖
try:
    from model_utils import ProteinMPNN, gather_nodes, cat_neighbors_nodes
except ImportError as e:
    print(f"❌ 错误: 找不到 model_utils.py。\n{e}")
    sys.exit(1)

# =============================================================================
# 1. 模型定义
# =============================================================================
class DecoupledProteinMPNN(ProteinMPNN):
    def __init__(self, hidden_dim=128, augment_eps=0.1, **kwargs):
        super().__init__(num_letters=21, hidden_dim=hidden_dim, vocab=21, k_neighbors=48, augment_eps=augment_eps, **kwargs)
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
            # 这里现在应该可以正常工作了
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
        logits_methyl = self.W_out_methyl(h_V)
        return logits_base, logits_methyl

# =============================================================================
# 2. 数据处理
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
# 3. 核心诊断逻辑
# =============================================================================
def analyze_performance(model_path, test_data_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Loading model from: {model_path}")
    
    model = DecoupledProteinMPNN(augment_eps=0.0).to(device)
    try:
        ckpt = torch.load(model_path, map_location=device)
        model.load_state_dict(ckpt['model_state_dict'])
        print("✅ Model loaded successfully.")
    except Exception as e:
        print(f"❌ 加载模型失败: {e}")
        return

    test_ds = JSONLDataset(test_data_path)
    test_loader = DataLoader(test_ds, batch_size=8, shuffle=False, collate_fn=collate_fn)

    print("running inference...")
    model.eval()
    
    all_targets = []
    all_preds = []
    
    methyl_idx_to_nat_idx = {}
    offset = len(NATURAL_AA_ALPHABET)
    for m_rel, n_idx in NMETHYL_TO_NATURAL_MAPPING.items():
        methyl_idx_to_nat_idx[m_rel + offset] = n_idx

    idx_to_name = {v: k for k, v in EXTENDED_AA_TO_INDEX.items()}

    with torch.no_grad():
        for batch in test_loader:
            f = featurize_batch(batch, device)
            if f is None: continue
            X, S, mask, chain_M, residue_idx, chain_encoding_all = f
            
            logits_base, logits_methyl = model(X, S, mask, chain_M, residue_idx, chain_encoding_all)
            
            pred_base_idx = torch.argmax(logits_base, -1)
            probs_methyl = F.softmax(logits_methyl, dim=-1)[:, :, 1]
            pred_is_methyl = (probs_methyl > 0.4).long()
            
            final_pred = pred_base_idx.clone()
            for m_rel, n_idx in NMETHYL_TO_NATURAL_MAPPING.items():
                m_abs_idx = m_rel + offset
                mask_update = (pred_is_methyl == 1) & (pred_base_idx == n_idx)
                final_pred[mask_update] = m_abs_idx
            
            targets = S.cpu().numpy().flatten()
            preds = final_pred.cpu().numpy().flatten()
            mask_flat = mask.cpu().numpy().flatten().astype(bool)
            
            valid = mask_flat & (targets != EXTENDED_AA_TO_INDEX['X'])
            all_targets.extend(targets[valid])
            all_preds.extend(preds[valid])

    all_targets = np.array(all_targets)
    all_preds = np.array(all_preds)

    print("\n" + "="*80)
    print("🕵️‍♀️  TRUTH REVEALED: METHYLATION BREAKDOWN")
    print("="*80)
    
    methyl_stats = []
    total_methyl_count = 0
    total_methyl_correct = 0
    
    for m_char in METHYL_AA_ALPHABET:
        m_idx = EXTENDED_AA_TO_INDEX[m_char]
        n_idx = methyl_idx_to_nat_idx[m_idx]
        
        indices = np.where(all_targets == m_idx)[0]
        count = len(indices)
        
        if count == 0: continue
        total_methyl_count += count
        
        current_preds = all_preds[indices]
        correct = np.sum(current_preds == m_idx)
        total_methyl_correct += correct
        
        mis_as_natural = np.sum(current_preds == n_idx)
        mis_as_other = count - correct - mis_as_natural
        
        recall = correct / count * 100 if count > 0 else 0
        
        methyl_stats.append({
            "Methyl AA": f"{m_char} (Methyl-{idx_to_name[n_idx]})",
            "Count": count,
            "Correct": correct,
            "Recall (%)": f"{recall:.1f}%",
            "Miss -> Natural": mis_as_natural,
            "Miss -> Other": mis_as_other
        })

    df = pd.DataFrame(methyl_stats)
    if not df.empty:
        print(df.to_string(index=False))
        print("-" * 80)
        total_recall = total_methyl_correct / total_methyl_count * 100
        print(f"🔥 TOTAL METHYL RECALL: {total_recall:.2f}% ({total_methyl_correct}/{total_methyl_count})")
    else:
        print("测试集中没有发现甲基化氨基酸数据。")

    print("\n" + "="*80)
    print("📊 FULL CLASSIFICATION REPORT")
    print("="*80)
    
    unique_labels = sorted(list(set(all_targets) | set(all_preds)))
    target_names = [idx_to_name[i] for i in unique_labels]
    
    print(classification_report(all_targets, all_preds, labels=unique_labels, target_names=target_names, zero_division=0))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="./run_v14_breakthrough/best_model_v14.pt")
    parser.add_argument("--test_data", type=str, required=True)
    args = parser.parse_args()
    
    analyze_performance(args.model_path, args.test_data)
import argparse
import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint # <--- 之前缺了这一行！
import json
import numpy as np
import random
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from sklearn.metrics import classification_report

# --- 系统路径 ---
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

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
# 1. 模型定义 (必须与训练时完全一致)
# =============================================================================

class DecoupledProteinMPNN(ProteinMPNN):
    def __init__(self, node_features=128, edge_features=128, hidden_dim=128, 
                 num_encoder_layers=3, num_decoder_layers=3, k_neighbors=48, dropout=0.1, augment_eps=0.0, **kwargs):
        super().__init__(num_letters=21, node_features=node_features, edge_features=edge_features, 
                         hidden_dim=hidden_dim, num_encoder_layers=num_encoder_layers, 
                         num_decoder_layers=num_decoder_layers, vocab=21, k_neighbors=k_neighbors, 
                         dropout=dropout, augment_eps=augment_eps, **kwargs)

        self.W_s = nn.Embedding(len(EXTENDED_AA_ALPHABET), hidden_dim) 
        self.W_out_base = nn.Linear(hidden_dim, len(NATURAL_AA_ALPHABET)) 
        self.W_out_methyl = nn.Linear(hidden_dim, 2)

    def forward(self, X, S, mask, chain_M, residue_idx, chain_encoding_all):
        device = X.device
        E, E_idx = self.features(X, mask, residue_idx, chain_encoding_all)
        h_V = torch.zeros((E.shape[0], E.shape[1], E.shape[-1]), device=E.device)
        h_E = self.W_e(E)

        mask_attend = gather_nodes(mask.unsqueeze(-1), E_idx).squeeze(-1)
        mask_attend = mask.unsqueeze(-1) * mask_attend
        for layer in self.encoder_layers:
            # 修复点：确保 torch.utils.checkpoint 被正确调用
            h_V, h_E = torch.utils.checkpoint.checkpoint(layer, h_V, h_E, E_idx, mask, mask_attend)

        h_S = self.W_s(S)
        h_ES = cat_neighbors_nodes(h_S, h_E, E_idx)
        h_EX_encoder = cat_neighbors_nodes(torch.zeros_like(h_S), h_E, E_idx)
        h_EXV_encoder = cat_neighbors_nodes(h_V, h_EX_encoder, E_idx)
        
        chain_M = chain_M * mask
        decoding_order = torch.argsort((chain_M + 0.0001) * (torch.abs(torch.randn(chain_M.shape, device=device))))
        mask_size = E_idx.shape[1]
        permutation_matrix_reverse = torch.nn.functional.one_hot(decoding_order, num_classes=mask_size).float()
        order_mask_backward = torch.einsum('ij, biq, bjp->bqp', (1 - torch.triu(torch.ones(mask_size, mask_size, device=device))), permutation_matrix_reverse, permutation_matrix_reverse)
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
# 2. 数据处理辅助
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
    # 简化的 featurize，仅用于推理
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

# =============================================================================
# 3. 核心评估函数 (带阈值)
# =============================================================================

def evaluate_with_threshold(model, loader, device, methyl_threshold=0.1):
    """
    methyl_threshold: 只要甲基化头的概率 > threshold，就认为是甲基化
    """
    model.eval()
    print(f"\n>>> 正在测试... 甲基化判定阈值: {methyl_threshold}")
    
    all_preds = []
    all_targets = []
    
    # 准备映射
    offset = len(NATURAL_AA_ALPHABET)
    natural_to_methyl_abs = {}
    for methyl_rel, natural_idx in NMETHYL_TO_NATURAL_MAPPING.items():
        natural_to_methyl_abs[natural_idx] = methyl_rel + offset

    with torch.no_grad():
        for batch in loader:
            f = featurize_batch(batch, device)
            if f is None: continue
            X, S, mask, chain_M, residue_idx, chain_encoding_all = f
            
            # 获取输出
            logits_base, logits_methyl = model(X, S, mask, chain_M, residue_idx, chain_encoding_all)
            
            # 1. 基础类型 (argmax)
            pred_base_idx = torch.argmax(logits_base, dim=-1)
            
            # 2. 甲基化判定 (Softmax + Threshold)
            probs_methyl = F.softmax(logits_methyl, dim=-1) # [B, L, 2]
            prob_is_methyl = probs_methyl[:, :, 1] # 取出 "是甲基化" 的概率
            
            # 只要概率 > threshold，就判为 1
            pred_is_methyl = (prob_is_methyl > methyl_threshold).long()
            
            # 3. 组合结果
            final_preds = pred_base_idx.clone()
            
            B, L = pred_base_idx.shape
            for b in range(B):
                for l in range(L):
                    base = pred_base_idx[b, l].item()
                    is_me = pred_is_methyl[b, l].item()
                    
                    if is_me == 1:
                        if base in natural_to_methyl_abs:
                            final_preds[b, l] = natural_to_methyl_abs[base]
            
            # 收集结果
            targets_flat = S.cpu().numpy().flatten()
            preds_flat = final_preds.cpu().numpy().flatten()
            mask_flat = mask.cpu().numpy().flatten().astype(bool)
            
            x_idx = EXTENDED_AA_TO_INDEX.get('X', -1)
            valid = mask_flat & (targets_flat != x_idx)
            
            all_preds.extend(preds_flat[valid])
            all_targets.extend(targets_flat[valid])

    # 计算指标
    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)
    
    if len(all_targets) == 0:
        print("无有效样本。")
        return

    acc = np.mean(all_preds == all_targets)
    
    # 计算天然氨基酸恢复率
    natural_mask = all_targets < len(NATURAL_AA_ALPHABET)
    natural_acc = np.mean(all_targets[natural_mask] == all_preds[natural_mask]) if natural_mask.sum() > 0 else 0

    print(f"Overall Accuracy (Threshold {methyl_threshold}): {acc:.4f}")
    print(f"Natural AA Recovery: {natural_acc:.4f}")
    
    # 打印详细报告
    labels = sorted(np.unique(np.concatenate([all_targets, all_preds])).astype(int))
    names = [EXTENDED_AA_ALPHABET[i] for i in labels if i < len(EXTENDED_AA_ALPHABET)]
    print(classification_report(all_targets, all_preds, labels=labels, target_names=names, zero_division=0))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True, help="Path to best_model.pt")
    parser.add_argument("--test_data", type=str, required=True)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 加载模型
    print(f"Loading weights from {args.model_path}...")
    model = DecoupledProteinMPNN(augment_eps=0.0).to(device)
    state = torch.load(args.model_path, map_location=device)
    if 'model_state_dict' in state: state = state['model_state_dict']
    model.load_state_dict(state)
    print("Weights loaded successfully.")
    
    test_ds = JSONLDataset(args.test_data)
    test_loader = DataLoader(test_ds, batch_size=8, shuffle=False, collate_fn=collate_fn)
    
    # 尝试不同的阈值
    # 注意：如果模型非常保守，可能需要极低的阈值才能召回
    for t in [0.5, 0.3, 0.1, 0.05, 0.01]:
        evaluate_with_threshold(model, test_loader, device, methyl_threshold=t)
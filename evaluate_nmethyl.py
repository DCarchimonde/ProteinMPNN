import argparse
import os
import sys
import torch
import torch.nn as nn
import torch.utils.checkpoint  # <--- [修复] 补全关键导入
import json
import numpy as np
import random
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import classification_report

# --- 系统路径适配 ---
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

# --- 尝试导入项目依赖 ---
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
# 1. 模型定义 (DecoupledProteinMPNN)
# =============================================================================
class DecoupledProteinMPNN(ProteinMPNN):
    def __init__(self, hidden_dim=128, augment_eps=0.0, **kwargs):
        super().__init__(
            num_letters=21, 
            hidden_dim=hidden_dim, 
            vocab=21, 
            k_neighbors=48, 
            augment_eps=augment_eps, 
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
        order_mask_backward = torch.einsum('ij, biq, bjp->bqp', 
                                           (1 - torch.triu(torch.ones(mask_size, mask_size, device=X.device))), 
                                           permutation_matrix_reverse, 
                                           permutation_matrix_reverse)
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
# 2. 数据加载与特征化
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
    # [修复] 移除了之前那个可能导致空列表的 if 'seq' in b 检查
    alphabet = EXTENDED_AA_ALPHABET
    B = len(batch)
    # 简单的非空检查
    batch = [b for b in batch if b is not None]
    if not batch: return None

    # 计算最大长度
    lengths = []
    for b in batch:
        current_len = 0
        all_chains = b.get('masked_list', []) + b.get('visible_list', [])
        for c_id in all_chains:
            current_len += len(b.get(f'seq_chain_{c_id}', ''))
        lengths.append(current_len)
    
    if not lengths: return None
    L_max = max(lengths)
    if L_max == 0: return None

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
            
            X[i, l_p:l_p+l, 0, :] = N[:l]
            X[i, l_p:l_p+l, 1, :] = CA[:l]
            X[i, l_p:l_p+l, 2, :] = C[:l]
            X[i, l_p:l_p+l, 3, :] = O[:l]
            
            indices = []
            for aa in seq[:l]:
                idx = EXTENDED_AA_TO_INDEX.get(aa, EXTENDED_AA_TO_INDEX['X'])
                indices.append(idx)
            S[i, l_p:l_p+l] = indices
            
            if c_id in b.get('masked_list', []): chain_M[i, l_p:l_p+l] = 1.0
            residue_idx[i, l_p:l_p+l] = np.arange(l) + c_i * 100
            chain_encoding_all[i, l_p:l_p+l] = c_i
            l_p += l
            
    isnan = np.isnan(X)
    mask = np.isfinite(np.sum(X, (2,3))).astype(np.float32)
    X[isnan] = 0.
    
    return [
        torch.from_numpy(t).to(dtype=torch.long if t.dtype==np.int32 else torch.float32, device=device) 
        for t in [X, S, mask, chain_M, residue_idx, chain_encoding_all]
    ]

# =============================================================================
# 3. 详细评估统计
# =============================================================================
def detailed_evaluation(model, loader, device):
    model.eval()
    print("\n=== 开始详细评估 N-甲基化恢复率 ===")
    
    all_preds, all_targets = [], []
    nat_to_me_abs = {v: k + len(NATURAL_AA_ALPHABET) for k, v in NMETHYL_TO_NATURAL_MAPPING.items()}
    
    with torch.no_grad():
        for batch in loader:
            f = featurize_batch(batch, device)
            if f is None: continue
            X, S, mask, chain_M, residue_idx, chain_encoding_all = f
            
            # 前向传播
            lb, lm = model(X, S, mask, chain_M, residue_idx, chain_encoding_all)
            
            # 解码
            base = torch.argmax(lb, -1)
            is_me = torch.argmax(lm, -1)
            final = base.clone()
            
            # 融合逻辑
            for n_idx, m_idx in nat_to_me_abs.items():
                mask_update = (is_me == 1) & (base == n_idx)
                final[mask_update] = m_idx
            
            # 数据收集
            tgts = S.cpu().numpy().flatten()
            preds = final.cpu().numpy().flatten()
            mask_flat = mask.cpu().numpy().flatten().astype(bool)
            
            # 过滤无效位
            valid = mask_flat & (tgts != EXTENDED_AA_TO_INDEX.get('X', -1))
            all_preds.extend(preds[valid])
            all_targets.extend(tgts[valid])

    all_preds, all_targets = np.array(all_preds), np.array(all_targets)
    
    if len(all_targets) == 0:
        print("警告: 测试数据集为空或未找到有效样本。请检查 jsonl 文件内容。")
        return

    # --- 计算统计指标 ---
    # 1. 总体准确率
    overall_acc = np.mean(all_preds == all_targets)
    
    # 2. 天然氨基酸恢复率 (<20)
    nat_mask = all_targets < len(NATURAL_AA_ALPHABET)
    nat_acc = np.mean(all_preds[nat_mask] == all_targets[nat_mask]) if nat_mask.sum() > 0 else 0.0
    
    # 3. N-甲基化恢复率 (>=20)
    nme_mask = all_targets >= len(NATURAL_AA_ALPHABET)
    nme_acc = np.mean(all_preds[nme_mask] == all_targets[nme_mask]) if nme_mask.sum() > 0 else 0.0

    # --- 打印报告 ---
    labels = sorted(np.unique(np.concatenate([all_targets, all_preds])).astype(int))
    names = [EXTENDED_AA_ALPHABET[i] for i in labels if i < len(EXTENDED_AA_ALPHABET)]
    
    print("\n[详细分类报告]")
    report_dict = classification_report(all_targets, all_preds, labels=labels, target_names=names, zero_division=0, output_dict=True)
    print(classification_report(all_targets, all_preds, labels=labels, target_names=names, zero_division=0))

    print("\n" + "="*60)
    print(f"{'N-甲基化类型':<12} | {'恢复率 (Recall)':<18} | {'样本数 (Support)':<15}")
    print("-" * 55)

    methyl_stats = []
    methyl_indices = [k + len(NATURAL_AA_ALPHABET) for k in NMETHYL_TO_NATURAL_MAPPING.keys()]
    
    for idx in methyl_indices:
        aa_char = EXTENDED_AA_ALPHABET[idx]
        if aa_char in report_dict:
            stats = report_dict[aa_char]
            recall = stats['recall']
            support = stats['support']
            methyl_stats.append((aa_char, recall, support))
        else:
            methyl_stats.append((aa_char, 0.0, 0))

    # 排序：按恢复率降序
    methyl_stats.sort(key=lambda x: x[1], reverse=True)

    for aa_char, recall, support in methyl_stats:
        # 只显示样本数大于0的，或者你可以注释掉 if 来显示所有
        if support > 0:
            print(f"{aa_char:<12} | {recall*100:6.2f}%            | {support:<15}")
    
    print("="*60)
    
    # --- 最终汇总 ---
    print("\n" + "#"*45)
    print("           最 终 评 估 汇 总 (SUMMARY)           ")
    print("#"*45)
    print(f"总体准确率 (Overall Accuracy):     {overall_acc*100:.2f}%")
    print(f"天然氨基酸恢复率 (Natural AA):     {nat_acc*100:.2f}%")
    print(f"N-甲基化总恢复率 (N-Methyl All):   {nme_acc*100:.2f}%")
    print("#"*45 + "\n")

# =============================================================================
# 4. 主函数 (Entry Point)
# =============================================================================
def main():
    parser = argparse.ArgumentParser()
    # 默认路径设为你刚才确认过的路径
    parser.add_argument("--test_data", type=str, default="nmethyl_data/test_sets/test.jsonl", help="Path to test dataset")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint (.pt)")
    args = parser.parse_args()
    
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. 初始化模型
    model = DecoupledProteinMPNN(augment_eps=0.0).to(device)
    
    # 2. 加载权重
    print(f"Loading weights from: {args.checkpoint}")
    try:
        ckpt = torch.load(args.checkpoint, map_location=device)
        state_dict = ckpt.get('model_state_dict', ckpt)
        model.load_state_dict(state_dict, strict=False)
        print("Model weights loaded successfully.")
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        sys.exit(1)
    
    # 3. 加载数据
    if not os.path.exists(args.test_data):
        print(f"Error: Test file not found at {args.test_data}")
        sys.exit(1)
        
    print(f"Loading test data from: {args.test_data}")
    test_ds = JSONLDataset(args.test_data)
    test_loader = DataLoader(test_ds, batch_size=8, shuffle=False, collate_fn=collate_fn)
    
    # 4. 执行评估
    detailed_evaluation(model, test_loader, device)

# [修复] 确保程序有入口
if __name__ == "__main__":
    main()
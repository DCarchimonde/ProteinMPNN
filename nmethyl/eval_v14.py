import torch
import torch.nn as nn
import torch.nn.functional as F
import json
import argparse
import sys
import os
import numpy as np
import pandas as pd # 用于打印漂亮的表格
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import classification_report, confusion_matrix

# =============================================================================
# 0. 配置与常量 (必须与 V14 一致)
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
# 映射关系: 0(Ala) -> 0, 20(MAA) -> 0
NMETHYL_TO_NATURAL_MAPPING = {i: EXTENDED_AA_TO_INDEX[char.upper()] for i, char in enumerate(METHYL_AA_ALPHABET)}

# 尝试导入依赖，如果不行就报错
try:
    from model_utils import ProteinMPNN, gather_nodes, cat_neighbors_nodes
except ImportError as e:
    print(f"❌ 错误: 找不到 model_utils.py。请把此脚本放在与 model_utils.py 同级目录下。\n{e}")
    sys.exit(1)

# =============================================================================
# 1. 模型定义 (必须完全复制 V14 以加载权重)
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
# 2. 数据处理与加载器
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
    # (这里直接使用你 V14 中的 featurize_batch 逻辑，为节省篇幅省略，假设它在 model_utils 或直接复制过来)
    # 为了保证代码独立运行，这里必须完整复制一遍 featurize_batch
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
    
    # 1. Load Model
    model = DecoupledProteinMPNN(augment_eps=0.0).to(device)
    try:
        ckpt = torch.load(model_path, map_location=device)
        model.load_state_dict(ckpt['model_state_dict'])
        print("✅ Model loaded successfully.")
    except FileNotFoundError:
        print(f"❌ 找不到文件: {model_path}")
        print("请检查路径，或者在命令行参数中指定 --model_path")
        return

    # 2. Setup Data
    test_ds = JSONLDataset(test_data_path)
    test_loader = DataLoader(test_ds, batch_size=8, shuffle=False, collate_fn=collate_fn)

    # 3. Inference
    print("running inference...")
    model.eval()
    
    all_targets = []
    all_preds = []
    
    # 辅助字典：从甲基化索引查天然索引
    methyl_idx_to_nat_idx = {}
    offset = len(NATURAL_AA_ALPHABET)
    for m_rel, n_idx in NMETHYL_TO_NATURAL_MAPPING.items():
        methyl_idx_to_nat_idx[m_rel + offset] = n_idx

    # 辅助字典：索引转名称
    idx_to_name = {v: k for k, v in EXTENDED_AA_TO_INDEX.items()}

    with torch.no_grad():
        for batch in test_loader:
            f = featurize_batch(batch, device)
            if f is None: continue
            X, S, mask, chain_M, residue_idx, chain_encoding_all = f
            
            # Forward
            logits_base, logits_methyl = model(X, S, mask, chain_M, residue_idx, chain_encoding_all)
            
            # Predict
            pred_base_idx = torch.argmax(logits_base, -1)
            probs_methyl = F.softmax(logits_methyl, dim=-1)[:, :, 1]
            pred_is_methyl = (probs_methyl > 0.4).long() # 阈值 0.4
            
            # Construct Final Prediction
            final_pred = pred_base_idx.clone()
            for m_rel, n_idx in NMETHYL_TO_NATURAL_MAPPING.items():
                m_abs_idx = m_rel + offset
                # 逻辑：如果预测是甲基化，且基底预测正确(是对应的天然氨基酸)，则认为是该甲基化氨基酸
                mask_update = (pred_is_methyl == 1) & (pred_base_idx == n_idx)
                final_pred[mask_update] = m_abs_idx
            
            # Collect
            targets = S.cpu().numpy().flatten()
            preds = final_pred.cpu().numpy().flatten()
            mask_flat = mask.cpu().numpy().flatten().astype(bool)
            
            valid = mask_flat & (targets != EXTENDED_AA_TO_INDEX['X'])
            all_targets.extend(targets[valid])
            all_preds.extend(preds[valid])

    # 4. Analysis
    all_targets = np.array(all_targets)
    all_preds = np.array(all_preds)

    print("\n" + "="*60)
    print("🕵️‍♀️  TRUTH REVEALED: METHYLATION BREAKDOWN")
    print("="*60)
    
    methyl_stats = []
    
    total_methyl_count = 0
    total_methyl_correct = 0
    
    # 遍历所有可能的甲基化氨基酸
    for m_char in METHYL_AA_ALPHABET:
        m_idx = EXTENDED_AA_TO_INDEX[m_char]
        n_idx = methyl_idx_to_nat_idx[m_idx] # 它的天然形态
        
        # 找出真实标签是该甲基化氨基酸的所有位置
        indices = np.where(all_targets == m_idx)[0]
        count = len(indices)
        
        if count == 0: continue
        
        total_methyl_count += count
        
        # 具体的预测情况
        current_preds = all_preds[indices]
        
        correct = np.sum(current_preds == m_idx)
        total_methyl_correct += correct
        
        mis_as_natural = np.sum(current_preds == n_idx) # 没检测出甲基化，但猜对了基底
        mis_as_other = count - correct - mis_as_natural # 彻底猜错了
        
        recall = correct / count * 100
        
        methyl_stats.append({
            "Methyl AA": f"{m_char} (Methyl-{idx_to_name[n_idx]})",
            "Count": count,
            "Correct": correct,
            "Recall (%)": f"{recall:.1f}%",
            "Miss -> Natural": mis_as_natural,
            "Miss -> Other": mis_as_other
        })

    # 创建 DataFrame 展示
    df = pd.DataFrame(methyl_stats)
    if not df.empty:
        print(df.to_string(index=False))
        print("-" * 60)
        total_recall = total_methyl_correct / total_methyl_count * 100
        print(f"🔥 TOTAL METHYL RECALL: {total_recall:.2f}% ({total_methyl_correct}/{total_methyl_count})")
        
        if total_recall < 10.0:
            print("\n🚨 警报: 模型几乎完全忽略了甲基化！这是严重的 Mode Collapse。")
            print("原因可能是: 1. Loss权重不够 2. 数据特征里根本看不出来。")
        elif total_recall > 80.0:
            print("\n🎉 恭喜！模型是真的学会了，而不是瞎猜的！")
    else:
        print("测试集中没有发现甲基化氨基酸数据。")

    print("\n" + "="*60)
    print("📊 FULL CLASSIFICATION REPORT")
    print("="*60)
    
    # 获取出现的 label 用于 report
    unique_labels = sorted(list(set(all_targets) | set(all_preds)))
    target_names = [idx_to_name[i] for i in unique_labels]
    
    print(classification_report(all_targets, all_preds, labels=unique_labels, target_names=target_names, zero_division=0))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # 默认路径指向你 V14 代码中的默认保存位置
    parser.add_argument("--model_path", type=str, default="./run_v14_breakthrough/best_model_v14.pt", help="Path to .pt file")
    parser.add_argument("--test_data", type=str, required=True, help="Path to test.jsonl")
    args = parser.parse_args()
    
    analyze_performance(args.model_path, args.test_data)
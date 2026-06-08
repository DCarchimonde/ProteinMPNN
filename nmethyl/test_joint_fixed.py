import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint  # <--- [关键修复] 必须显式导入这个模块
import numpy as np
from sklearn.metrics import classification_report
import sys
import os
import argparse
import json 

# --- 系统路径 ---
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    # 导入配置
    from nmethyl.utils.nmethyl_config import (
        EXTENDED_AA_ALPHABET, NATURAL_AA_ALPHABET,
        NMETHYL_TO_NATURAL_MAPPING, EXTENDED_AA_TO_INDEX
    )
    # 导入核心组件 (必须是你最新修改过的 model_utils)
    from model_utils import ProteinMPNN
    # 导入 Dataset
    from torch.utils.data import DataLoader, Dataset
except ImportError as e:
    print(f"Error importing modules: {e}")
    sys.exit(1)

# =============================================================================
# 1. 显式定义模型类 (确保参数对齐)
# =============================================================================
class DecoupledProteinMPNN(ProteinMPNN):
    def __init__(self, hidden_dim=128, augment_eps=0.0, **kwargs):
        # 这里的关键是显式传入 node_features 和 edge_features 的默认值
        # 你的 model_utils.py 里 ProteinMPNN 的默认值可能是 128
        super().__init__(
            num_letters=21, 
            node_features=128, 
            edge_features=128, 
            hidden_dim=hidden_dim, 
            vocab=21, 
            k_neighbors=48, 
            augment_eps=augment_eps, 
            **kwargs
        )
        # 覆盖 W_s 和 输出层 (与训练时一致)
        self.W_s = nn.Embedding(len(EXTENDED_AA_ALPHABET), hidden_dim)
        self.W_out_base = nn.Linear(hidden_dim, len(NATURAL_AA_ALPHABET))
        self.W_out_methyl = nn.Linear(hidden_dim, 2)

    def forward(self, X, S, mask, chain_M, residue_idx, chain_encoding_all):
        # 这一步会调用 model_utils.py 里修改过的 features (含虚拟甲基)
        E, E_idx = self.features(X, mask, residue_idx, chain_encoding_all)
        
        h_V = torch.zeros((E.shape[0], E.shape[1], E.shape[-1]), device=E.device)
        h_E = self.W_e(E)

        # 简单的 gather_nodes 实现，防止 import 失败
        def gather_nodes(nodes, neighbor_idx):
            neighbors_flat = neighbor_idx.view((neighbor_idx.shape[0], -1))
            neighbors_flat = neighbors_flat.unsqueeze(-1).expand(-1, -1, nodes.size(2))
            neighbor_features = torch.gather(nodes, 1, neighbors_flat)
            neighbor_features = neighbor_features.view(list(neighbor_idx.shape)[:3] + [-1])
            return neighbor_features

        def cat_neighbors_nodes(h_nodes, h_neighbors, E_idx):
            h_nodes = gather_nodes(h_nodes, E_idx)
            h_nn = torch.cat([h_neighbors, h_nodes], -1)
            return h_nn

        mask_attend = gather_nodes(mask.unsqueeze(-1),  E_idx).squeeze(-1)
        mask_attend = mask.unsqueeze(-1) * mask_attend
        
        # [关键修复] 这里使用显式导入的 checkpoint
        for layer in self.encoder_layers:
            h_V, h_E = torch.utils.checkpoint.checkpoint(layer, h_V, h_E, E_idx, mask, mask_attend)

        h_S = self.W_s(S)
        h_ES = cat_neighbors_nodes(h_S, h_E, E_idx)
        h_EX_encoder = cat_neighbors_nodes(torch.zeros_like(h_S), h_E, E_idx)
        h_EXV_encoder = cat_neighbors_nodes(h_V, h_EX_encoder, E_idx)

        chain_M = chain_M * mask
        # 测试时不需要随机噪声，但为了代码兼容性保留
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
            # [关键修复] 这里也是
            h_V = torch.utils.checkpoint.checkpoint(layer, h_V, h_ESV, mask)
        
        logits_base = self.W_out_base(h_V)
        logits_methyl = self.W_out_methyl(h_V)
        return logits_base, logits_methyl

# =============================================================================
# 2. 补全缺失的辅助函数 (Dataset & Featurize)
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
    """
    将 JSONL 数据转换为模型输入张量
    """
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
            
    isnan = np.isnan(X); mask = np.isfinite(np.sum(X, (2,3))).astype(np.float32); X[isnan] = 0.
    return [torch.from_numpy(t).to(dtype=torch.long if t.dtype==np.int32 else torch.float32, device=device) for t in [X, S, mask, chain_M, residue_idx, chain_encoding_all]]

# =============================================================================
# 3. 评估逻辑 (Joint Probability)
# =============================================================================
def evaluate_joint(model, loader, device):
    model.eval()
    print("\n>>> 正在进行联合概率推理 (Joint Probability Inference)...")
    
    all_preds = []
    all_targets = []
    
    num_natural = len(NATURAL_AA_ALPHABET) # 20
    num_extended = len(EXTENDED_AA_ALPHABET) # ~40
    
    nat_to_methyl_map = {}
    for methyl_rel_idx, nat_idx in NMETHYL_TO_NATURAL_MAPPING.items():
        nat_to_methyl_map[nat_idx] = methyl_rel_idx + num_natural

    with torch.no_grad():
        for batch in loader:
            # 这里的 featurize_batch 现在直接在本地定义了
            f = featurize_batch(batch, device)
            if f is None: continue
            X, S, mask, chain_M, residue_idx, chain_encoding_all = f
            
            logits_base, logits_methyl = model(X, S, mask, chain_M, residue_idx, chain_encoding_all)
            
            log_probs_base = F.log_softmax(logits_base, dim=-1)   # [B, L, 20]
            log_probs_methyl = F.log_softmax(logits_methyl, dim=-1) # [B, L, 2]
            
            B, L, _ = logits_base.shape
            final_scores = torch.full((B, L, num_extended), -1e9, device=device)
            
            # A. 天然得分
            for i in range(num_natural):
                final_scores[:, :, i] = log_probs_base[:, :, i] + log_probs_methyl[:, :, 0]
                
            # B. 甲基化得分
            for nat_idx, ext_idx in nat_to_methyl_map.items():
                if ext_idx < num_extended:
                    final_scores[:, :, ext_idx] = log_probs_base[:, :, nat_idx] + log_probs_methyl[:, :, 1]
            
            final_preds = torch.argmax(final_scores, dim=-1)
            
            targets_flat = S.cpu().numpy().flatten()
            preds_flat = final_preds.cpu().numpy().flatten()
            mask_flat = mask.cpu().numpy().flatten().astype(bool)
            
            x_idx = EXTENDED_AA_TO_INDEX.get('X', -1)
            valid = mask_flat & (targets_flat != x_idx)
            
            all_preds.extend(preds_flat[valid])
            all_targets.extend(targets_flat[valid])

    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)
    
    acc = np.mean(all_preds == all_targets)
    nat_mask = all_targets < num_natural
    nat_acc = np.mean(all_preds[nat_mask] == all_targets[nat_mask]) if nat_mask.sum() > 0 else 0
    methyl_mask = all_targets >= num_natural
    methyl_acc = np.mean(all_preds[methyl_mask] == all_targets[methyl_mask]) if methyl_mask.sum() > 0 else 0

    print("\n" + "="*50)
    print(f"JOINT INFERENCE RESULTS")
    print("="*50)
    print(f"Overall Accuracy:       {acc:.4f} ({acc*100:.2f}%)")
    print(f"Natural AA Recovery:    {nat_acc:.4f}")
    print(f"N-Methyl AA Recovery:   {methyl_acc:.4f}")
    print("-" * 50)
    
    labels = sorted(np.unique(np.concatenate([all_targets, all_preds])).astype(int))
    names = [EXTENDED_AA_ALPHABET[i] for i in labels if i < len(EXTENDED_AA_ALPHABET)]
    print(classification_report(all_targets, all_preds, labels=labels, target_names=names, zero_division=0))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--test_data", type=str, required=True)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 初始化模型 (显式传入参数以防报错)
    model = DecoupledProteinMPNN(hidden_dim=128, augment_eps=0.0).to(device)
    
    print(f"Loading: {args.model_path}")
    state = torch.load(args.model_path, map_location=device)
    if 'model_state_dict' in state: state = state['model_state_dict']
    
    # 容错加载 (以防 key 不匹配)
    model.load_state_dict(state, strict=False)
    
    test_ds = JSONLDataset(args.test_data)
    test_loader = DataLoader(test_ds, batch_size=8, shuffle=False, collate_fn=collate_fn)
    
    evaluate_joint(model, test_loader, device)
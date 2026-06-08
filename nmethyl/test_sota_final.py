import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import classification_report
import sys
import os
import argparse
import json 

# --- 系统路径 ---
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from nmethyl.utils.nmethyl_config import (
        EXTENDED_AA_ALPHABET, NATURAL_AA_ALPHABET,
        NMETHYL_TO_NATURAL_MAPPING, EXTENDED_AA_TO_INDEX
    )
    from model_utils import ProteinMPNN
    from torch.utils.data import DataLoader, Dataset
except ImportError as e:
    print(f"Error importing modules: {e}")
    sys.exit(1)

# =============================================================================
# 1. 模型定义 (完整复刻，补回 Decoder Mask)
# =============================================================================
class DecoupledProteinMPNN(ProteinMPNN):
    def __init__(self, hidden_dim=128, augment_eps=0.0, **kwargs):
        super().__init__(
            num_letters=21, node_features=128, edge_features=128, 
            hidden_dim=hidden_dim, vocab=21, k_neighbors=48, 
            augment_eps=augment_eps, **kwargs
        )
        self.W_s = nn.Embedding(len(EXTENDED_AA_ALPHABET), hidden_dim)
        self.W_out_base = nn.Linear(hidden_dim, len(NATURAL_AA_ALPHABET))
        self.W_out_methyl = nn.Linear(hidden_dim, 2)

    def forward(self, X, S, mask, chain_M, residue_idx, chain_encoding_all):
        # 1. 特征提取
        E, E_idx = self.features(X, mask, residue_idx, chain_encoding_all)
        h_V = torch.zeros((E.shape[0], E.shape[1], E.shape[-1]), device=E.device)
        h_E = self.W_e(E)
        
        # 2. 辅助函数
        def gather_nodes(nodes, neighbor_idx):
            neighbors_flat = neighbor_idx.view((neighbor_idx.shape[0], -1))
            neighbors_flat = neighbors_flat.unsqueeze(-1).expand(-1, -1, nodes.size(2))
            return torch.gather(nodes, 1, neighbors_flat).view(list(neighbor_idx.shape)[:3] + [-1])
        def cat_neighbors_nodes(h_nodes, h_neighbors, E_idx):
            h_nodes = gather_nodes(h_nodes, E_idx)
            return torch.cat([h_neighbors, h_nodes], -1)

        # 3. Encoder 掩码
        mask_attend = gather_nodes(mask.unsqueeze(-1),  E_idx).squeeze(-1)
        mask_attend = mask.unsqueeze(-1) * mask_attend
        
        # 4. Encoder 运行
        for layer in self.encoder_layers:
            h_V, h_E = layer(h_V, h_E, E_idx, mask, mask_attend)

        # 5. Decoder 准备
        h_S = self.W_s(S)
        h_ES = cat_neighbors_nodes(h_S, h_E, E_idx)
        h_EX_encoder = cat_neighbors_nodes(torch.zeros_like(h_S), h_E, E_idx)
        h_EXV_encoder = cat_neighbors_nodes(h_V, h_EX_encoder, E_idx)
        
        # 6. [CRITICAL FIX] 补回 Decoder Mask 逻辑 (之前缺失的部分!)
        # 即使是测试，也需要一个解码顺序（这里用随机顺序，模拟真实生成场景）
        chain_M = chain_M * mask
        decoding_order = torch.argsort((chain_M + 0.0001) * (torch.abs(torch.randn(chain_M.shape, device=X.device))))
        mask_size = E_idx.shape[1]
        permutation_matrix_reverse = torch.nn.functional.one_hot(decoding_order, num_classes=mask_size).float()
        order_mask_backward = torch.einsum('ij, biq, bjp->bqp', (1 - torch.triu(torch.ones(mask_size, mask_size, device=X.device))), permutation_matrix_reverse, permutation_matrix_reverse)
        
        # 这才是 Decoder 真正需要的 mask_attend (带方向性)
        mask_attend = torch.gather(order_mask_backward, 2, E_idx).unsqueeze(-1)
        
        mask_1D = mask.view([mask.size(0), mask.size(1), 1, 1])
        mask_bw = mask_1D * mask_attend
        mask_fw = mask_1D * (1. - mask_attend)
        h_EXV_encoder_fw = mask_fw * h_EXV_encoder

        # 7. Decoder 运行
        for layer in self.decoder_layers:
            h_ESV = cat_neighbors_nodes(h_V, h_ES, E_idx)
            h_ESV = mask_bw * h_ESV + h_EXV_encoder_fw
            h_V = layer(h_V, h_ESV, mask)
        
        logits_base = self.W_out_base(h_V)
        logits_methyl = self.W_out_methyl(h_V)
        return logits_base, logits_methyl

# =============================================================================
# 2. 数据读取 (SOTA 修复版)
# =============================================================================
def featurize_batch_sota_test(batch, device):
    B = len(batch)
    batch = [b for b in batch if 'seq' in b and len(b['seq']) > 0]
    if not batch: return None
    lengths = [len(b['seq']) for b in batch]
    L_max = max(lengths)
    
    X = np.full([B, L_max, 4, 3], np.nan)
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
            coords_N = np.array(b.get(f'N_chain_{c_id}', []))
            coords_CA = np.array(b.get(f'CA_chain_{c_id}', []))
            coords_C = np.array(b.get(f'C_chain_{c_id}', []))
            coords_O = np.array(b.get(f'O_chain_{c_id}', []))
            
            l = len(seq)
            def safe_assign(source, atom_idx):
                if len(source) > 0:
                    v_len = min(l, len(source))
                    X[i, l_p:l_p+v_len, atom_idx, :] = source[:v_len]
            safe_assign(coords_N, 0); safe_assign(coords_CA, 1)
            safe_assign(coords_C, 2); safe_assign(coords_O, 3)
            
            indices = []
            for aa in seq[:l]: indices.append(EXTENDED_AA_TO_INDEX.get(aa, EXTENDED_AA_TO_INDEX['X']))
            S[i, l_p:l_p+l] = indices
            if c_id in b.get('masked_list', []): chain_M[i, l_p:l_p+l] = 1.0
            residue_idx[i, l_p:l_p+l] = np.arange(l) + c_i * 100
            chain_encoding_all[i, l_p:l_p+l] = c_i
            l_p += l
    
    mask = np.isfinite(X[:, :, 1, 0]).astype(np.float32)
    X[np.isnan(X)] = 0.0 
    return [torch.from_numpy(t).to(dtype=torch.long if t.dtype==np.int32 else torch.float32, device=device) for t in [X, S, mask, chain_M, residue_idx, chain_encoding_all]]

# =============================================================================
# 3. 评估逻辑
# =============================================================================
class JSONLDataset(Dataset):
    def __init__(self, jsonl_file):
        self.data = []
        with open(jsonl_file, 'r') as f:
            for line in f: self.data.append(json.loads(line))
    def __len__(self): return len(self.data)
    def __getitem__(self, idx): return self.data[idx]

def evaluate_joint(model, loader, device):
    model.eval()
    print("\n>>> [SOTA] 正在进行联合概率推理...")
    all_preds, all_targets = [], []
    num_natural = len(NATURAL_AA_ALPHABET)
    num_extended = len(EXTENDED_AA_ALPHABET)
    nat_to_methyl_map = {}
    for m_idx, n_idx in NMETHYL_TO_NATURAL_MAPPING.items():
        nat_to_methyl_map[n_idx] = m_idx + num_natural

    with torch.no_grad():
        for batch in loader:
            f = featurize_batch_sota_test(batch, device)
            if f is None: continue
            X, S, mask, chain_M, residue_idx, chain_encoding_all = f
            
            logits_base, logits_methyl = model(X, S, mask, chain_M, residue_idx, chain_encoding_all)
            log_p_base = F.log_softmax(logits_base, -1)
            log_p_methyl = F.log_softmax(logits_methyl, -1)
            
            scores = torch.full((logits_base.shape[0], logits_base.shape[1], num_extended), -1e9, device=device)
            for i in range(num_natural):
                scores[:, :, i] = log_p_base[:, :, i] + log_p_methyl[:, :, 0]
            for n_idx, e_idx in nat_to_methyl_map.items():
                if e_idx < num_extended:
                    scores[:, :, e_idx] = log_p_base[:, :, n_idx] + log_p_methyl[:, :, 1]
            
            preds = torch.argmax(scores, -1)
            valid = mask.bool() & (S != EXTENDED_AA_TO_INDEX.get('X', -1))
            all_preds.extend(preds[valid].cpu().numpy())
            all_targets.extend(S[valid].cpu().numpy())

    all_preds = np.array(all_preds); all_targets = np.array(all_targets)
    if len(all_targets) == 0: 
        print("没有有效的测试数据。")
        return

    acc = np.mean(all_preds == all_targets)
    nat_mask = all_targets < num_natural
    nat_acc = np.mean(all_preds[nat_mask] == all_targets[nat_mask]) if nat_mask.sum() > 0 else 0
    met_mask = all_targets >= num_natural
    met_acc = np.mean(all_preds[met_mask] == all_targets[met_mask]) if met_mask.sum() > 0 else 0

    print("\n" + "="*50)
    print(f"SOTA FINAL RESULTS")
    print("="*50)
    print(f"Overall Accuracy:       {acc:.4f} ({acc*100:.2f}%)")
    print(f"Natural AA Recovery:    {nat_acc:.4f}")
    print(f"N-Methyl AA Recovery:   {met_acc:.4f}")
    print("-" * 50)
    labels = sorted(np.unique(np.concatenate([all_targets, all_preds])).astype(int))
    names = [EXTENDED_AA_ALPHABET[i] for i in labels if i < len(EXTENDED_AA_ALPHABET)]
    print(classification_report(all_targets, all_preds, labels=labels, target_names=names, zero_division=0))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="./run_nature_sota/best_model_sota.pt")
    parser.add_argument("--test_data", type=str, required=True)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DecoupledProteinMPNN(hidden_dim=128).to(device)
    
    print(f"Loading: {args.model_path}")
    state = torch.load(args.model_path, map_location=device)
    if 'model_state_dict' in state: state = state['model_state_dict']
    model.load_state_dict(state, strict=False)
    
    ds = JSONLDataset(args.test_data)
    loader = DataLoader(ds, batch_size=8, shuffle=False, collate_fn=lambda x: x)
    evaluate_joint(model, loader, device)
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint
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
# 1. 核心特征类 (修复维度问题的根源)
# =============================================================================

class BaseFeatures(nn.Module):
    def __init__(self, edge_features, node_features, num_positional_embeddings=16,
        num_rbf=16, top_k=30, augment_eps=0.):
        super(BaseFeatures, self).__init__()
        self.top_k = top_k
        self.num_rbf = num_rbf
        self.num_positional_embeddings = num_positional_embeddings
        self.augment_eps = augment_eps
        
        # 基础 RBF 通道数 = 25
        self.edge_in = num_positional_embeddings + num_rbf * 25
        
        self.embeddings = nn.Linear(2*32+1+1, num_positional_embeddings)
        self.norm_edges = nn.LayerNorm(edge_features)

    def _dist(self, X, mask, eps=1E-6):
        # mask: [B, L]
        # mask_2D: [B, L, L]
        mask_2D = torch.unsqueeze(mask,1) * torch.unsqueeze(mask,2)
        dX = torch.unsqueeze(X,1) - torch.unsqueeze(X,2)
        D = mask_2D * torch.sqrt(torch.sum(dX**2, 3) + eps)
        D_max, _ = torch.max(D, -1, keepdim=True)
        D_adjust = D + (1. - mask_2D) * D_max
        D_neighbors, E_idx = torch.topk(D_adjust, np.minimum(self.top_k, X.shape[1]), dim=-1, largest=False)
        return D_neighbors, E_idx

    def _rbf(self, D):
        # D: [B, L, K]
        device = D.device
        D_min, D_max, D_count = 2., 22., self.num_rbf
        D_mu = torch.linspace(D_min, D_max, D_count, device=device).view([1,1,1,-1])
        D_sigma = (D_max - D_min) / D_count
        D_expand = torch.unsqueeze(D, -1) # [B, L, K, 1]
        RBF = torch.exp(-((D_expand - D_mu) / D_sigma)**2) # [B, L, K, 16]
        return RBF

    def _get_rbf(self, A, B, E_idx):
        D_A_B = torch.sqrt(torch.sum((A[:,:,None,:] - B[:,None,:,:])**2,-1) + 1e-6)
        D_A_B_neighbors = torch.gather(D_A_B, 2, E_idx) # [B, L, K]
        return self._rbf(D_A_B_neighbors)

    def _get_pos_embedding(self, residue_idx, E_idx, mask):
        # residue_idx: [B, L]
        offset = residue_idx[:,:,None] - residue_idx[:,None,:] # [B, L, L]
        offset = torch.gather(offset, 2, E_idx) # [B, L, K]
        
        max_rel = 32
        
        # [CRITICAL FIX] 修正 mask 的广播维度
        # mask: [B, L] -> [B, L, 1]
        mask_expanded = mask.unsqueeze(-1) 
        
        # d: [B, L, K]
        d = torch.clip(offset + max_rel, 0, 2*max_rel) * mask_expanded + (1-mask_expanded)*(2*max_rel+1)
        
        # d_onehot: [B, L, K, 66]
        d_onehot = torch.nn.functional.one_hot(d.long(), 2*max_rel+1+1)
        
        # E_positional: [B, L, K, 16] (Rank 4)
        E_positional = self.embeddings(d_onehot.float())
        return E_positional

# --- Vanilla (普通版) ---
class Vanilla_ProteinFeatures(BaseFeatures):
    def __init__(self, edge_features, node_features, **kwargs):
        super(Vanilla_ProteinFeatures, self).__init__(edge_features, node_features, **kwargs)
        self.edge_embedding = nn.Linear(self.edge_in, edge_features, bias=False)

    def forward(self, X, mask, residue_idx, chain_labels):
        Ca, N, C, O = X[:,:,1,:], X[:,:,0,:], X[:,:,2,:], X[:,:,3,:]
        b = Ca - N; c = C - Ca; a = torch.cross(b, c, dim=-1)
        Cb = -0.5827*a + 0.5680*b - 0.5407*c + Ca
        
        D_neighbors, E_idx = self._dist(Ca, mask)

        RBF_all = []
        RBF_all.append(self._rbf(D_neighbors)) 
        RBF_all.extend([self._get_rbf(N, N, E_idx), self._get_rbf(C, C, E_idx), self._get_rbf(O, O, E_idx), self._get_rbf(Cb, Cb, E_idx)])
        RBF_all.extend([self._get_rbf(Ca, N, E_idx), self._get_rbf(Ca, C, E_idx), self._get_rbf(Ca, O, E_idx), self._get_rbf(Ca, Cb, E_idx)])
        RBF_all.extend([self._get_rbf(N, C, E_idx), self._get_rbf(N, O, E_idx), self._get_rbf(N, Cb, E_idx)])
        RBF_all.extend([self._get_rbf(Cb, C, E_idx), self._get_rbf(Cb, O, E_idx), self._get_rbf(O, C, E_idx)])
        RBF_all.extend([self._get_rbf(N, Ca, E_idx), self._get_rbf(C, Ca, E_idx), self._get_rbf(O, Ca, E_idx), self._get_rbf(Cb, Ca, E_idx)])
        RBF_all.extend([self._get_rbf(C, N, E_idx), self._get_rbf(O, N, E_idx), self._get_rbf(Cb, N, E_idx)])
        RBF_all.extend([self._get_rbf(C, Cb, E_idx), self._get_rbf(O, Cb, E_idx), self._get_rbf(C, O, E_idx)])
        
        RBF_all = torch.cat(tuple(RBF_all), dim=-1) # [B, L, K, 25*16]
        E_positional = self._get_pos_embedding(residue_idx, E_idx, mask)
        
        E = torch.cat((E_positional, RBF_all), -1)
        E = self.edge_embedding(E)
        E = self.norm_edges(E)
        return E, E_idx

# --- SOTA (顶刊版 - 含虚拟甲基) ---
class SOTA_ProteinFeatures(BaseFeatures):
    def __init__(self, edge_features, node_features, **kwargs):
        super(SOTA_ProteinFeatures, self).__init__(edge_features, node_features, **kwargs)
        # SOTA 增加 5 个通道
        self.edge_in = self.edge_in + self.num_rbf * 5
        self.edge_embedding = nn.Linear(self.edge_in, edge_features, bias=False)

    def forward(self, X, mask, residue_idx, chain_labels):
        Ca, N, C, O = X[:,:,1,:], X[:,:,0,:], X[:,:,2,:], X[:,:,3,:]
        b = Ca - N; c = C - Ca; a = torch.cross(b, c, dim=-1)
        Cb = -0.5827*a + 0.5680*b - 0.5407*c + Ca
        
        # 虚拟甲基计算
        v1 = (Ca - N) / (torch.norm(Ca - N, dim=-1, keepdim=True) + 1e-6)
        v2 = (C - N) / (torch.norm(C - N, dim=-1, keepdim=True) + 1e-6)
        bisector = -(v1 + v2)
        bisector = bisector / (torch.norm(bisector, dim=-1, keepdim=True) + 1e-6)
        Virtual_CM = N + 1.47 * bisector

        D_neighbors, E_idx = self._dist(Ca, mask)

        RBF_all = []
        RBF_all.append(self._rbf(D_neighbors)) 
        RBF_all.extend([self._get_rbf(N, N, E_idx), self._get_rbf(C, C, E_idx), self._get_rbf(O, O, E_idx), self._get_rbf(Cb, Cb, E_idx)])
        RBF_all.extend([self._get_rbf(Ca, N, E_idx), self._get_rbf(Ca, C, E_idx), self._get_rbf(Ca, O, E_idx), self._get_rbf(Ca, Cb, E_idx)])
        RBF_all.extend([self._get_rbf(N, C, E_idx), self._get_rbf(N, O, E_idx), self._get_rbf(N, Cb, E_idx)])
        RBF_all.extend([self._get_rbf(Cb, C, E_idx), self._get_rbf(Cb, O, E_idx), self._get_rbf(O, C, E_idx)])
        RBF_all.extend([self._get_rbf(N, Ca, E_idx), self._get_rbf(C, Ca, E_idx), self._get_rbf(O, Ca, E_idx), self._get_rbf(Cb, Ca, E_idx)])
        RBF_all.extend([self._get_rbf(C, N, E_idx), self._get_rbf(O, N, E_idx), self._get_rbf(Cb, N, E_idx)])
        RBF_all.extend([self._get_rbf(C, Cb, E_idx), self._get_rbf(O, Cb, E_idx), self._get_rbf(C, O, E_idx)])
        
        # 新增 5 个特征
        RBF_all.extend([
            self._get_rbf(Virtual_CM, N, E_idx),
            self._get_rbf(Virtual_CM, Ca, E_idx),
            self._get_rbf(Virtual_CM, C, E_idx),
            self._get_rbf(Virtual_CM, O, E_idx),
            self._get_rbf(N, O, E_idx)
        ])

        RBF_all = torch.cat(tuple(RBF_all), dim=-1) # [B, L, K, 30*16]
        E_positional = self._get_pos_embedding(residue_idx, E_idx, mask)
        
        E = torch.cat((E_positional, RBF_all), -1)
        E = self.edge_embedding(E)
        E = self.norm_edges(E)
        return E, E_idx

# =============================================================================
# 2. 自动适配模型
# =============================================================================
class AutoAdapterProteinMPNN(ProteinMPNN):
    def __init__(self, feature_type='vanilla', hidden_dim=128, augment_eps=0.0, **kwargs):
        # [Fix] 显式传入必需参数
        super().__init__(
            num_letters=21, node_features=128, edge_features=128, 
            hidden_dim=hidden_dim, vocab=21, k_neighbors=48, 
            augment_eps=augment_eps, **kwargs
        )
        
        # 动态选择特征层
        if feature_type == 'sota':
            print("  [Auto] Detecting SOTA Architecture (Virtual Methyl)...")
            self.features = SOTA_ProteinFeatures(128, 128, top_k=48, augment_eps=augment_eps)
        else:
            print("  [Auto] Detecting Vanilla Architecture (Standard)...")
            self.features = Vanilla_ProteinFeatures(128, 128, top_k=48, augment_eps=augment_eps)
            
        self.W_s = nn.Embedding(len(EXTENDED_AA_ALPHABET), hidden_dim)
        self.W_out_base = nn.Linear(hidden_dim, len(NATURAL_AA_ALPHABET))
        self.W_out_methyl = nn.Linear(hidden_dim, 2)

    def forward(self, X, S, mask, chain_M, residue_idx, chain_encoding_all):
        E, E_idx = self.features(X, mask, residue_idx, chain_encoding_all)
        h_V = torch.zeros((E.shape[0], E.shape[1], E.shape[-1]), device=E.device)
        h_E = self.W_e(E)

        # 内置 gather 避免依赖
        def gather_nodes(nodes, neighbor_idx):
            neighbors_flat = neighbor_idx.view((neighbor_idx.shape[0], -1))
            neighbors_flat = neighbors_flat.unsqueeze(-1).expand(-1, -1, nodes.size(2))
            return torch.gather(nodes, 1, neighbors_flat).view(list(neighbor_idx.shape)[:3] + [-1])

        def cat_neighbors_nodes(h_nodes, h_neighbors, E_idx):
            h_nodes = gather_nodes(h_nodes, E_idx)
            return torch.cat([h_neighbors, h_nodes], -1)

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
# 3. 严格数据处理
# =============================================================================
class JSONLDataset(Dataset):
    def __init__(self, jsonl_file):
        self.data = []
        with open(jsonl_file, 'r') as f:
            for line in f: self.data.append(json.loads(line))
    def __len__(self): return len(self.data)
    def __getitem__(self, idx): return self.data[idx]

def collate_fn(batch): return batch

def featurize_batch_strict(batch, device):
    """Strict mask filtering"""
    B = len(batch)
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

            safe_assign(coords_N, 0)
            safe_assign(coords_CA, 1)
            safe_assign(coords_C, 2)
            safe_assign(coords_O, 3)
            
            indices = []
            for aa in seq[:l]:
                indices.append(EXTENDED_AA_TO_INDEX.get(aa, EXTENDED_AA_TO_INDEX['X']))
            S[i, l_p:l_p+l] = indices
            
            if c_id in b.get('masked_list', []): chain_M[i, l_p:l_p+l] = 1.0
            residue_idx[i, l_p:l_p+l] = np.arange(l) + c_i * 100
            chain_encoding_all[i, l_p:l_p+l] = c_i
            l_p += l
            
    has_N = np.isfinite(X[:, :, 0, 0])
    has_CA = np.isfinite(X[:, :, 1, 0])
    has_C = np.isfinite(X[:, :, 2, 0])
    mask = (has_N & has_CA & has_C).astype(np.float32)
    X[np.isnan(X)] = 0.0 
    
    return [torch.from_numpy(t).to(dtype=torch.long if t.dtype==np.int32 else torch.float32, device=device) for t in [X, S, mask, chain_M, residue_idx, chain_encoding_all]]

def evaluate_joint(model, loader, device):
    model.eval()
    print("\n>>> 正在进行联合概率推理...")
    all_preds, all_targets = [], []
    num_natural = len(NATURAL_AA_ALPHABET)
    num_extended = len(EXTENDED_AA_ALPHABET)
    nat_to_methyl_map = {}
    for m_idx, n_idx in NMETHYL_TO_NATURAL_MAPPING.items():
        nat_to_methyl_map[n_idx] = m_idx + num_natural

    with torch.no_grad():
        for batch in loader:
            f = featurize_batch_strict(batch, device)
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

    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)
    if len(all_targets) == 0: return

    acc = np.mean(all_preds == all_targets)
    nat_mask = all_targets < num_natural
    nat_acc = np.mean(all_preds[nat_mask] == all_targets[nat_mask]) if nat_mask.sum() > 0 else 0
    met_mask = all_targets >= num_natural
    met_acc = np.mean(all_preds[met_mask] == all_targets[met_mask]) if met_mask.sum() > 0 else 0

    print("\n" + "="*50)
    print(f"UNIVERSAL TEST RESULTS (Fixed)")
    print("="*50)
    print(f"Overall Accuracy:       {acc:.4f}")
    print(f"Natural AA Recovery:    {nat_acc:.4f}")
    print(f"N-Methyl AA Recovery:   {met_acc:.4f}")
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
    print(f"Inspecting checkpoint: {args.model_path}")
    state = torch.load(args.model_path, map_location=device)
    state_dict = state.get('model_state_dict', state)
    feature_type = 'vanilla'
    if 'features.edge_embedding.weight' in state_dict:
        # SOTA: 496, Vanilla: 416
        if state_dict['features.edge_embedding.weight'].shape[1] > 450:
            feature_type = 'sota'
    
    model = AutoAdapterProteinMPNN(feature_type=feature_type, hidden_dim=128).to(device)
    model.load_state_dict(state_dict, strict=False)
    
    ds = JSONLDataset(args.test_data)
    loader = DataLoader(ds, batch_size=8, shuffle=False, collate_fn=collate_fn)
    evaluate_joint(model, loader, device)
import os
import sys
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import json
import numpy as np
from torch.utils.data import Dataset, DataLoader 
import pandas as pd

os.environ["OMP_NUM_THREADS"] = "1"

# =============================================================================
# 1. 基础配置
# =============================================================================
NATURAL_AA_ALPHABET = "ACDEFGHIKLMNPQRSTVWY"
EXTENDED_AA_ALPHABET = "ACDEFGHIKLMNPQRSTVWYacdefghiklmnqrstvwvyX"
EXTENDED_AA_TO_INDEX = {aa: i for i, aa in enumerate(EXTENDED_AA_ALPHABET)}

NMETHYL_TO_NATURAL_MAPPING = {} 
for i, aa in enumerate(NATURAL_AA_ALPHABET):
    lower_aa = aa.lower()
    if lower_aa in EXTENDED_AA_TO_INDEX:
        NMETHYL_TO_NATURAL_MAPPING[EXTENDED_AA_TO_INDEX[lower_aa]] = i

# =============================================================================
# 2. 模型结构与加载 (完美保留)
# =============================================================================
def gather_nodes(nodes, neighbor_idx):
    neighbors_flat = neighbor_idx.view((neighbor_idx.shape[0], -1))
    return torch.gather(nodes, 1, neighbors_flat.unsqueeze(-1).expand(-1, -1, nodes.size(2))).view(list(neighbor_idx.shape)[:3] + [-1])

def cat_neighbors_nodes(h_nodes, h_neighbors, E_idx):
    return torch.cat([h_neighbors, gather_nodes(h_nodes, E_idx)], -1)

class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.W_in = nn.Linear(d_model, d_ff); self.W_out = nn.Linear(d_ff, d_model); self.act = nn.GELU()
    def forward(self, x): return self.W_out(self.act(self.W_in(x)))

class EncLayer(nn.Module):
    def __init__(self, num_hidden, num_in, dropout=0.1, scale=30):
        super().__init__()
        self.scale = scale
        self.dropout1, self.dropout2 = nn.Dropout(dropout), nn.Dropout(dropout)
        self.norm1, self.norm2 = nn.LayerNorm(num_hidden), nn.LayerNorm(num_hidden)
        self.W1, self.W2, self.W3 = nn.Linear(num_hidden+num_in, num_hidden), nn.Linear(num_hidden, num_hidden), nn.Linear(num_hidden, num_hidden)
        self.act, self.dense = nn.GELU(), PositionwiseFeedForward(num_hidden, num_hidden*4, dropout)
    def forward(self, h_V, h_E, E_idx, mask_V=None, mask_attend=None):
        h_EV = torch.cat([h_V.unsqueeze(-2).expand(-1,-1,h_E.size(-2),-1), gather_nodes(h_V, E_idx), h_E], -1) 
        h_message = self.W3(self.act(self.W2(self.act(self.W1(h_EV)))))
        if mask_attend is not None: h_message = mask_attend.unsqueeze(-1) * h_message
        h_V = self.norm1(h_V + self.dropout1(torch.sum(h_message, -2)/self.scale))
        return self.norm2(h_V + self.dropout2(self.dense(h_V))), h_E

class DecLayer(nn.Module):
    def __init__(self, num_hidden, num_in, dropout=0.1, scale=30):
        super().__init__()
        self.scale = scale
        self.dropout1, self.dropout2 = nn.Dropout(dropout), nn.Dropout(dropout)
        self.norm1, self.norm2 = nn.LayerNorm(num_hidden), nn.LayerNorm(num_hidden)
        self.W1, self.W2, self.W3 = nn.Linear(num_hidden+num_in, num_hidden), nn.Linear(num_hidden, num_hidden), nn.Linear(num_hidden, num_hidden)
        self.act, self.dense = nn.GELU(), PositionwiseFeedForward(num_hidden, num_hidden*4, dropout)
    def forward(self, h_V, h_E, mask_V=None, mask_attend=None):
        h_cat = torch.cat([h_V.unsqueeze(-2).expand(-1,-1,h_E.size(-2),-1), h_E], -1) 
        h_message = self.W3(self.act(self.W2(self.act(self.W1(h_cat))))) 
        if mask_attend is not None: h_message = mask_attend.unsqueeze(-1) * h_message
        h_V = self.norm1(h_V + self.dropout1(torch.sum(h_message, -2)/self.scale))
        return self.norm2(h_V + self.dropout2(self.dense(h_V)))

class RobustHierarchicalProteinMPNN(nn.Module):
    def __init__(self, hidden_dim=128, k_neighbors=48, augment_eps=0.0, dropout=0.1): 
        super().__init__()
        self.hidden_dim, self.k_neighbors, self.augment_eps = hidden_dim, k_neighbors, augment_eps
        self.features = nn.ModuleDict({'embeddings': nn.ModuleDict({'linear': nn.Linear(66, 16)}), 'edge_embedding': nn.Linear(416, 128, bias=False), 'norm_edges': nn.LayerNorm(128)})
        self.W_e, self.W_s = nn.Linear(128, hidden_dim, bias=True), nn.Embedding(len(EXTENDED_AA_ALPHABET), hidden_dim)
        self.encoder_layers = nn.ModuleList([EncLayer(hidden_dim, hidden_dim*2, dropout=dropout) for _ in range(3)])
        self.decoder_layers = nn.ModuleList([DecLayer(hidden_dim, hidden_dim*3, dropout=dropout) for _ in range(3)])
        self.W_out_base = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, len(NATURAL_AA_ALPHABET)))
        self.experts = nn.ModuleList([nn.Linear(hidden_dim, 1) for _ in range(len(NATURAL_AA_ALPHABET))])

    def forward(self, X, S, mask, chain_M, residue_idx, chain_encoding_all):
        b, c = X[:,:,1,:] - X[:,:,0,:], X[:,:,2,:] - X[:,:,1,:]
        a = torch.cross(b, c, dim=-1)
        Cb = -0.58273431*a + 0.56802827*b - 0.54067466*c + X[:,:,1,:]
        dist = torch.norm(X[:,:,1,:].unsqueeze(1) - X[:,:,1,:].unsqueeze(2), dim=-1) + (1.0 - (mask.unsqueeze(1)*mask.unsqueeze(2)))*1e8
        E_idx = torch.topk(dist, min(self.k_neighbors, dist.shape[-1]), dim=-1, largest=False)[1]
        offset = torch.gather(residue_idx.unsqueeze(1) - residue_idx.unsqueeze(2), 2, E_idx)
        pos_emb = self.features['embeddings']['linear'](F.one_hot(torch.clip(offset + 32, 0, 64), 66).float())
        RBF_all = [self._rbf(torch.gather(dist, 2, E_idx))]
        for i, a1 in enumerate([X[:,:,0,:], X[:,:,2,:], X[:,:,3,:], Cb, X[:,:,1,:]]):
            for j, a2 in enumerate([X[:,:,0,:], X[:,:,2,:], X[:,:,3,:], Cb, X[:,:,1,:]]):
                if i!=4 or j!=4: RBF_all.append(self._rbf(torch.gather(torch.norm(a1.unsqueeze(1)-a2.unsqueeze(2), dim=-1), 2, E_idx)))
        E = self.features['norm_edges'](self.features['edge_embedding'](torch.cat((pos_emb, torch.cat(RBF_all, dim=-1)), -1)))
        h_V, h_E = torch.zeros((E.shape[0], E.shape[1], E.shape[-1]), device=E.device), self.W_e(E)
        mask_attend = mask.unsqueeze(-1) * gather_nodes(mask.unsqueeze(-1), E_idx).squeeze(-1)
        for layer in self.encoder_layers: h_V, h_E = layer(h_V, h_E, E_idx, mask, mask_attend)
        h_ES = cat_neighbors_nodes(self.W_s(S), h_E, E_idx)
        h_EXV_encoder = cat_neighbors_nodes(h_V, cat_neighbors_nodes(torch.zeros_like(self.W_s(S)), h_E, E_idx), E_idx)
        permutation_matrix_reverse = torch.nn.functional.one_hot(torch.argsort(chain_M*mask + 0.0001), num_classes=E_idx.shape[1]).float()
        order_mask_backward = torch.einsum('ij, biq, bjp->bqp', (1 - torch.triu(torch.ones(E_idx.shape[1], E_idx.shape[1], device=X.device))), permutation_matrix_reverse, permutation_matrix_reverse)
        mask_attend = torch.gather(order_mask_backward, 2, E_idx).unsqueeze(-1)
        mask_bw, mask_fw = mask.view([mask.size(0), mask.size(1), 1, 1]) * mask_attend, mask.view([mask.size(0), mask.size(1), 1, 1]) * (1. - mask_attend)
        for layer in self.decoder_layers: h_V = layer(h_V, mask_bw * cat_neighbors_nodes(h_V, h_ES, E_idx) + mask_fw * h_EXV_encoder, mask)
        
        logits_base = self.W_out_base(h_V)
        logits_experts = torch.cat([expert(h_V) for expert in self.experts], dim=-1)
        return logits_base, logits_experts

    def _rbf(self, D): return torch.exp(-((D.unsqueeze(-1) - torch.linspace(2., 22., 16, device=D.device)) / ((22.-2.)/16)) ** 2)

def bulletproof_load_weights(model, checkpoint_path, device):
    ckpt = torch.load(checkpoint_path, map_location=device)
    state_dict = ckpt.get('model_state_dict', ckpt)
    if 'W_out.weight' in state_dict:
        state_dict['W_out_base.3.weight'] = state_dict.pop('W_out.weight')
        state_dict['W_out_base.3.bias'] = state_dict.pop('W_out.bias')
    elif 'module.W_out.weight' in state_dict:
        state_dict['W_out_base.3.weight'] = state_dict.pop('module.W_out.weight')
        state_dict['W_out_base.3.bias'] = state_dict.pop('module.W_out.bias')
        
    model_state = model.state_dict()
    new_state_dict = {}
    for k, v in state_dict.items():
        clean_k = k.replace('module.', '')
        if clean_k in model_state:
            if v.shape != model_state[clean_k].shape:
                new_v = model_state[clean_k].clone()
                slices = tuple(slice(0, min(dim_v, dim_m)) for dim_v, dim_m in zip(v.shape, new_v.shape))
                new_v[slices] = v[slices]
                new_state_dict[clean_k] = new_v
            else:
                new_state_dict[clean_k] = v
    model.load_state_dict(new_state_dict, strict=False)

def featurize_inference_entry(b, device):
    X = np.zeros([1, len(b['seq_chain_A']), 4, 3])
    S = np.zeros([1, len(b['seq_chain_A'])], dtype=np.int32)
    N, CA, C, O = np.array(b['N_chain_A']), np.array(b['CA_chain_A']), np.array(b['C_chain_A']), np.array(b['O_chain_A'])
    l = min(len(b['seq_chain_A']), len(CA))
    if l == 0: return None
    
    if len(N) > 0 and len(CA) > 0 and len(O) > 0:
        if np.linalg.norm(N[:1] - O[:1]) < np.linalg.norm(N[:1] - CA[:1]) and np.linalg.norm(N[:1] - O[:1]) < 1.6: 
            CA, O = O, CA
            
    X[0, :l, 0, :], X[0, :l, 1, :], X[0, :l, 2, :], X[0, :l, 3, :] = N[:l], CA[:l], C[:l], O[:l]
    S[0, :l] = [EXTENDED_AA_TO_INDEX.get(aa, 40) for aa in b['seq_chain_A'][:l]]
    
    mask = torch.ones([1, X.shape[1]], dtype=torch.float32, device=device)
    chain_M = torch.ones([1, X.shape[1]], dtype=torch.float32, device=device)
    residue_idx = torch.from_numpy(np.arange(X.shape[1])[None, :]).to(dtype=torch.long, device=device)
    chain_encoding_all = torch.zeros([1, X.shape[1]], dtype=torch.long, device=device)
    
    X = torch.from_numpy(X).to(dtype=torch.float32, device=device)
    S = torch.from_numpy(S).to(dtype=torch.long, device=device)
    return [X, S, mask, chain_M, residue_idx, chain_encoding_all]

# =============================================================================
# 3. 生成清晰度满分的对比表格
# =============================================================================
def generate_clear_report(model, input_jsonl, device, save_file="Methylation_07_vs_08_Details.csv"):
    model.eval()
    print(f"🕵️‍♀️ 正在提取每一个氨基酸的打分细节...")
    
    detailed_data = []
    
    with open(input_jsonl, 'r') as f:
        for line in f:
            b = json.loads(line)
            pdb_id = b['name']
            
            features = featurize_inference_entry(b, device)
            if features is None: continue
            X, S, mask, chain_M, residue_idx, chain_encoding_all = features
            
            with torch.no_grad():
                l_base, l_experts = model(X, S, mask, chain_M, residue_idx, chain_encoding_all)
                
                true_labels = S[0].cpu().numpy()
                true_base_idx = S[0].clone()
                for m_idx_raw, n_idx_rel in NMETHYL_TO_NATURAL_MAPPING.items():
                    true_base_idx[true_base_idx == m_idx_raw] = n_idx_rel
                true_base_idx[true_base_idx >= 20] = 0 
                
                expert_logit = torch.gather(l_experts, -1, true_base_idx[None, :].unsqueeze(-1)).squeeze(-1)
                prob_methyl = torch.sigmoid(expert_logit)[0].cpu().numpy()
                
                for res_i in range(S.shape[1]):
                    if true_labels[res_i] == 40: continue # 跳过未知氨基酸 X
                    
                    real_aa_idx = true_base_idx[res_i].item()
                    real_aa = NATURAL_AA_ALPHABET[real_aa_idx]
                    is_nmethyl_true = "Yes (1)" if true_labels[res_i] >= 20 else "No (0)"
                    
                    current_prob = prob_methyl[res_i]
                    
                    # 记录这个残基的信息
                    detailed_data.append({
                        "PDB_ID (蛋白质)": pdb_id,
                        "Res_Idx (第几个氨基酸)": res_i + 1,
                        "Native_AA (原本是什么)": real_aa,
                        "GroundTruth (真实该加甲基吗)": is_nmethyl_true,
                        "AI_Probability (模型打分)": round(current_prob, 4),
                        "If_Thr=0.7 (及格线0.7时)": "✅ 加甲基" if current_prob > 0.70 else "❌ 不加",
                        "If_Thr=0.8 (及格线0.8时)": "✅ 加甲基" if current_prob > 0.80 else "❌ 不加"
                    })
                    
    # 转为表格并保存
    df = pd.DataFrame(detailed_data)
    
    # 核心：为了让你们看得清楚，按 PDB_ID 和 氨基酸顺序 排序！
    df = df.sort_values(by=["PDB_ID (蛋白质)", "Res_Idx (第几个氨基酸)"])
    
    df.to_csv(save_file, index=False)
    print(f"\n🎉 搞定！详细报告已生成：'{save_file}'")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True, help="你的缝合怪模型 ./frankenstein_v28.pt")
    parser.add_argument("--test_data", type=str, required=True, help="测试集 jsonl 文件")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = RobustHierarchicalProteinMPNN(augment_eps=0.0).to(device)
    bulletproof_load_weights(model, args.model_path, device)
    
    generate_clear_report(model, args.test_data, device)
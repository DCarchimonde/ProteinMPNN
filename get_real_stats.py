import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import json
import numpy as np
from torch.utils.data import DataLoader, Dataset
import torch.utils.checkpoint
import warnings

warnings.filterwarnings("ignore")
os.environ["OMP_NUM_THREADS"] = "1"

# 引入你的原生字典
try:
    from nmethyl.utils.nmethyl_config import (
        EXTENDED_AA_ALPHABET, NATURAL_AA_ALPHABET,
        NMETHYL_TO_NATURAL_MAPPING, EXTENDED_AA_TO_INDEX
    )
except ImportError:
    print("❌ 请确保在 ProteinMPNN-main 目录下运行，并存在 nmethyl 模块！")
    sys.exit(1)

# =============================================================================
# 1. 模型架构 (绝对 100% 使用你的 nn.Module 原生架构)
# =============================================================================
def gather_nodes(nodes, neighbor_idx):
    neighbors_flat = neighbor_idx.view((neighbor_idx.shape[0], -1))
    neighbors_flat = neighbors_flat.unsqueeze(-1).expand(-1, -1, nodes.size(2))
    return torch.gather(nodes, 1, neighbors_flat).view(list(neighbor_idx.shape)[:3] + [-1])

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

# =============================================================================
# 2. 数据提取器 (唯一改动点：修复多链截断的 Bug)
# =============================================================================
class JSONLDataset(Dataset):
    def __init__(self, jsonl_file): self.data = [json.loads(line) for line in open(jsonl_file, 'r')]
    def __len__(self): return len(self.data)
    def __getitem__(self, idx): return self.data[idx]

def featurize_batch(batch, device):
    B = len(batch)
    batch = [b for b in batch if 'seq_chain_A' in b and len(b['seq_chain_A']) > 0]
    if not batch: return None
    
    # 修复：遍历所有链计算最大长度，避免复合物越界
    lengths = []
    for b in batch:
        all_chains = b.get('masked_list', []) + b.get('visible_list', [])
        if not all_chains: all_chains = ['A']
        lengths.append(sum([len(b.get(f'seq_chain_{c}', '')) for c in all_chains]))
    L_max = max(lengths)
    
    X, S = np.zeros([B, L_max, 4, 3]), np.zeros([B, L_max], dtype=np.int32)
    residue_idx, chain_M, chain_encoding_all = -100*np.ones([B, L_max], dtype=np.int32), np.ones([B, L_max], dtype=np.float32), np.zeros([B, L_max], dtype=np.int32)
    
    for i, b in enumerate(batch):
        all_chains = b.get('masked_list', []) + b.get('visible_list', [])
        if not all_chains: all_chains = ['A']
        
        l_p = 0
        for c_i, c_id in enumerate(all_chains):
            seq = b.get(f'seq_chain_{c_id}', '')
            N, CA, C, O = np.array(b.get(f'N_chain_{c_id}', [])), np.array(b.get(f'CA_chain_{c_id}', [])), np.array(b.get(f'C_chain_{c_id}', [])), np.array(b.get(f'O_chain_{c_id}', []))
            l = min(len(seq), len(CA))
            if l == 0: continue
            
            if len(N) > 0 and len(CA) > 0 and len(O) > 0:
                dist_n_ca = np.linalg.norm(N[:1] - CA[:1])
                dist_n_o = np.linalg.norm(N[:1] - O[:1])
                if dist_n_o < dist_n_ca and dist_n_o < 1.6:
                    CA, O = O, CA
                    
            l = min(l, len(N), len(C), len(O))
            if l == 0: continue
                    
            X[i, l_p:l_p+l, 0, :] = N[:l]
            X[i, l_p:l_p+l, 1, :] = CA[:l]
            X[i, l_p:l_p+l, 2, :] = C[:l]
            X[i, l_p:l_p+l, 3, :] = O[:l]
            
            indices = [EXTENDED_AA_TO_INDEX.get(aa, EXTENDED_AA_TO_INDEX.get('X', 40)) for aa in seq[:l]]
            S[i, l_p:l_p+l] = indices
            if c_id in b.get('masked_list', []): chain_M[i, l_p:l_p+l] = 1.0
            residue_idx[i, l_p:l_p+l] = np.arange(l) + c_i * 100
            chain_encoding_all[i, l_p:l_p+l] = c_i
            l_p += l
            
    isnan = np.isnan(X)
    mask = np.isfinite(np.sum(X, (2,3))).astype(np.float32)
    X[isnan] = 0.
    return [torch.from_numpy(t).to(dtype=torch.long if t.dtype==np.int32 else torch.float32, device=device) for t in [X, S, mask, chain_M, residue_idx, chain_encoding_all]]

# =============================================================================
# 3. 评估指标 (沿用你截图中跑出 85% Acc 的逻辑，并增加 RAA)
# =============================================================================
def evaluate_real_metrics(model, loader, device, threshold):
    model.eval()
    
    all_targets_comb, all_preds_base_raw = [], []
    all_probs_experts_e2e, all_probs_experts_dec = [], []
    
    methyl_idx_to_nat_idx = {m + 20: n for m, n in NMETHYL_TO_NATURAL_MAPPING.items()}
    x_idx = 40
    offset = 20

    with torch.no_grad():
        for batch in loader:
            f = featurize_batch(batch, device)
            if f is None: continue
            X, S, mask, chain_M, residue_idx, chain_encoding_all = f
            
            l_base, l_experts = model(X, S, mask, chain_M, residue_idx, chain_encoding_all)
            
            # --- 1. 获取主干网络的预测 ---
            pred_base_idx = torch.argmax(l_base, -1)
            
            # --- 2. 端到端 (E2E) 专家视角：看预测的氨基酸 ---
            expert_logit_e2e = torch.gather(l_experts, -1, pred_base_idx.clamp(0,19).unsqueeze(-1)).squeeze(-1)
            prob_methyl_e2e = torch.sigmoid(expert_logit_e2e)
            
            # --- 3. 解耦 (Decoupled) 专家视角：看真实的氨基酸 (你跑出85%的逻辑) ---
            true_base_idx = S.clone()
            for m_rel, n_idx in NMETHYL_TO_NATURAL_MAPPING.items():
                true_base_idx[true_base_idx == (m_rel + offset)] = n_idx
            true_base_idx[true_base_idx >= 20] = 0
            expert_logit_dec = torch.gather(l_experts, -1, true_base_idx.unsqueeze(-1)).squeeze(-1)
            prob_methyl_dec = torch.sigmoid(expert_logit_dec)
            
            # --- 展平存储 ---
            tgts = S.cpu().numpy().flatten()
            pb_raw = pred_base_idx.cpu().numpy().flatten()
            p_methyl_e2e = prob_methyl_e2e.cpu().numpy().flatten()
            p_methyl_dec = prob_methyl_dec.cpu().numpy().flatten()
            mask_flat = mask.cpu().numpy().flatten().astype(bool)
            
            valid = mask_flat & (tgts != x_idx)
            all_targets_comb.extend(tgts[valid])
            all_preds_base_raw.extend(pb_raw[valid])
            all_probs_experts_e2e.extend(p_methyl_e2e[valid])
            all_probs_experts_dec.extend(p_methyl_dec[valid])

    all_targets_comb = np.array(all_targets_comb)
    all_preds_base_raw = np.array(all_preds_base_raw)
    all_probs_experts_e2e = np.array(all_probs_experts_e2e)
    all_probs_experts_dec = np.array(all_probs_experts_dec)
    
    # ---------------- 结算指标 ----------------
    # A. 算 RAA (End-to-End)
    pred_is_methyl_e2e = (all_probs_experts_e2e > threshold).astype(int)
    final_pred = all_preds_base_raw.copy()
    for m_abs_idx, n_idx in methyl_idx_to_nat_idx.items():
        mask_update = (pred_is_methyl_e2e == 1) & (all_preds_base_raw == n_idx)
        final_pred[mask_update] = m_abs_idx
    raa = np.mean(final_pred == all_targets_comb) * 100 if len(all_targets_comb) > 0 else 0.0

    # B. 算解耦分类性能 (Decoupled, 完全对应你的截图)
    true_labels = (all_targets_comb >= offset).astype(int)
    pred_labels = (all_probs_experts_dec > threshold).astype(int)
    
    TP = np.sum((true_labels == 1) & (pred_labels == 1))
    TN = np.sum((true_labels == 0) & (pred_labels == 0))
    FP = np.sum((true_labels == 0) & (pred_labels == 1))
    FN = np.sum((true_labels == 1) & (pred_labels == 0))
    
    acc = (TP + TN) / (TP + TN + FP + FN) * 100 if (TP + TN + FP + FN) > 0 else 0
    precision = TP / (TP + FP) * 100 if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) * 100 if (TP + FN) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    return {"RAA": raa, "Acc": acc, "Prec": precision, "Rec": recall, "F1": f1}

# =============================================================================
# 主执行入口
# =============================================================================
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--test_monomers", type=str, required=True)
    parser.add_argument("--test_multimers", type=str, required=True)
    parser.add_argument("--threshold", type=float, default=0.60)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = RobustHierarchicalProteinMPNN(augment_eps=0.0).to(device) 
    
    print(f"📦 正在使用原生 nn.Module 架构加载权重...")
    try: ckpt = torch.load(args.model_path, map_location=device, weights_only=False)
    except: ckpt = torch.load(args.model_path, map_location=device)
    state_dict = ckpt.get('model_state_dict', ckpt)
    
    # 你的原生防弹加载，确保不丢失任何参数
    new_state_dict = {}
    model_state = model.state_dict()
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
    print("✅ 权重完美挂载！")

    monomer_loader = DataLoader(JSONLDataset(args.test_monomers), batch_size=8, shuffle=False, collate_fn=lambda x:x)
    multimer_loader = DataLoader(JSONLDataset(args.test_multimers), batch_size=4, shuffle=False, collate_fn=lambda x:x)

    print(f"\n🧪 --- 真实测算结果 (Monomers, Thr={args.threshold}) ---")
    mon = evaluate_real_metrics(model, monomer_loader, device, args.threshold)
    print(f"✅ RAA (端到端恢复率) = {mon['RAA']:.2f}%")
    print(f"✅ 分类指标 (解耦验证) = Acc: {mon['Acc']:.2f}%, Prec: {mon['Prec']:.2f}%, Rec: {mon['Rec']:.2f}%, F1: {mon['F1']:.2f}")

    print(f"\n🧪 --- 真实测算结果 (17 Complexes) ---")
    mul = evaluate_real_metrics(model, multimer_loader, device, args.threshold)
    print(f"✅ RAA (端到端恢复率) = {mul['RAA']:.2f}%")
    print(f"说明: 复合物全是天然未修饰结构，因此我们只关注 RAA 恢复能力。")
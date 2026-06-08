import argparse
import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
import json
import copy
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
try:
    from torch.amp import GradScaler, autocast
except ImportError:
    from torch.cuda.amp import GradScaler, autocast

# =============================================================================
# 1. 配置与物理常量
# =============================================================================
EXTENDED_AA_ALPHABET = "ACDEFGHIKLMNPQRSTVWYXacdefghiklmnpqrstvwy"
NATURAL_AA_ALPHABET = "ACDEFGHIKLMNPQRSTVWY"
NMETHYL_TO_NATURAL_MAPPING = {i+21: i for i in range(20)}
EXTENDED_AA_TO_INDEX = {aa: i for i, aa in enumerate(EXTENDED_AA_ALPHABET)}

# [PHYSICS RULE] 氢键检测器
def compute_hbond_mask(X, mask):
    if X.dtype != torch.float32: X = X.float()
    B, L = X.shape[:2]
    N = X[:, :, 0, :]; O = X[:, :, 3, :]
    
    # 距离矩阵 [B, L, L]
    dist = torch.norm(N.unsqueeze(2) - O.unsqueeze(1), p=2, dim=-1)
    
    # 排除局部连接 (简单对角线掩码)
    diag_mask = torch.eye(L, device=X.device).unsqueeze(0)
    
    # 物理阈值：N...O < 3.5A 视为氢键
    is_hb = (dist < 3.5) & (dist > 0.5) & (diag_mask < 0.5)
    
    # 如果 N 参与了任何氢键，则不能甲基化
    has_hbond = is_hb.any(dim=2).float() 
    return has_hbond * mask

# =============================================================================
# 2. 模型核心 (极简 Gather 修复)
# =============================================================================
def gather_nodes(nodes, neighbor_idx):
    # nodes: [B, L, C], neighbor_idx: [B, L, K]
    neighbors_flat = neighbor_idx.view((neighbor_idx.shape[0], -1))
    neighbors_flat = neighbors_flat.unsqueeze(-1).expand(-1, -1, nodes.size(2))
    return torch.gather(nodes, 1, neighbors_flat).view(list(neighbor_idx.shape)[:3] + [-1])

def cat_neighbors_nodes(h_nodes, h_neighbors, E_idx):
    h_nodes = gather_nodes(h_nodes, E_idx)
    return torch.cat([h_neighbors, h_nodes], -1)

class PositionalEncodings(nn.Module):
    def __init__(self, num_embeddings, max_relative_feature=32):
        super().__init__()
        self.linear = nn.Linear(2*max_relative_feature+1+1, num_embeddings)
    def forward(self, offset, mask):
        d = torch.clip(offset + 32, 0, 64)*mask + (1-mask)*65
        d_onehot = torch.nn.functional.one_hot(d, 66)
        return self.linear(d_onehot.float())

class ProteinFeatures(nn.Module):
    def __init__(self, edge_features, node_features, num_positional_embeddings=16, num_rbf=16, top_k=30, augment_eps=0., dropout=0.1):
        super().__init__()
        self.edge_features = edge_features; self.node_features = node_features; self.top_k = top_k; self.augment_eps = augment_eps 
        self.num_rbf = num_rbf; self.num_positional_embeddings = num_positional_embeddings
        self.embeddings = PositionalEncodings(num_positional_embeddings)
        edge_in = num_positional_embeddings + num_rbf * 25
        self.edge_embedding = nn.Linear(edge_in, edge_features, bias=False)
        self.norm_edges = nn.LayerNorm(edge_features)

    def _dist(self, X, mask, eps=1E-6):
        mask_2D = torch.unsqueeze(mask,1) * torch.unsqueeze(mask,2)
        dX = torch.unsqueeze(X,1) - torch.unsqueeze(X,2)
        D = mask_2D * torch.sqrt(torch.sum(dX**2, 3) + eps)
        D_max, _ = torch.max(D, -1, keepdim=True)
        D_adjust = D + (1. - mask_2D) * D_max
        D_neighbors, E_idx = torch.topk(D_adjust, np.minimum(self.top_k, X.shape[1]), dim=-1, largest=False)
        return D_neighbors, E_idx

    def _rbf(self, D):
        device = D.device
        D_min, D_max, D_count = 2., 22., self.num_rbf
        D_mu = torch.linspace(D_min, D_max, D_count, device=device).view([1,1,1,-1])
        D_sigma = (D_max - D_min) / D_count
        return torch.exp(-((D.unsqueeze(-1) - D_mu) / D_sigma)**2)

    def _get_rbf(self, A, B, E_idx):
        D_A_B = torch.sqrt(torch.sum((A[:,:,None,:] - B[:,None,:,:])**2,-1) + 1e-6)
        # [Fix] Direct gather (no unsqeeze/expand needed for 3D)
        D_A_B_neighbors = torch.gather(D_A_B, 2, E_idx)
        return self._rbf(D_A_B_neighbors)

    def forward(self, X, mask, residue_idx, chain_labels):
        if self.training and self.augment_eps > 0: X = X + self.augment_eps * torch.randn_like(X)
        b = X[:,:,1,:] - X[:,:,0,:]; c = X[:,:,2,:] - X[:,:,1,:]
        a = torch.cross(b, c, dim=-1)
        Cb = -0.5827*a + 0.5680*b - 0.5407*c + X[:,:,1,:]
        Ca, N, C, O = X[:,:,1,:], X[:,:,0,:], X[:,:,2,:], X[:,:,3,:]
        
        # 1. Dist
        D_neighbors, E_idx = self._dist(Ca, mask)

        # 2. RBF
        RBF_all = []
        RBF_all.append(self._rbf(D_neighbors)) 
        for atom1 in [N, C, O, Cb, Ca]:
            for atom2 in [N, C, O, Cb, Ca]:
                if atom1 is Ca and atom2 is Ca: continue
                # Dist: [B, L, L]
                dist = torch.sqrt(torch.sum((atom1.unsqueeze(1) - atom2.unsqueeze(2))**2, -1) + 1e-6)
                # Gather Neighbors: [B, L, K]
                # [Fix] Direct gather
                D_neighbor = torch.gather(dist, 2, E_idx)
                RBF_all.append(self._rbf(D_neighbor))
        
        RBF_all = torch.cat(tuple(RBF_all), dim=-1)

        # 3. Positional
        offset = residue_idx[:,:,None] - residue_idx[:,None,:] # [B, L, L]
        # [Fix] Direct gather
        offset = torch.gather(offset, 2, E_idx) 
        
        d_chains = ((chain_labels[:, :, None] - chain_labels[:,None,:])==0).long()
        # [Fix] Direct gather
        E_chains = torch.gather(d_chains, 2, E_idx)
        
        E_positional = self.embeddings(offset.long(), E_chains)
        E = torch.cat((E_positional, RBF_all), -1)
        return self.norm_edges(self.edge_embedding(E)), E_idx

class DecoupledProteinMPNN(nn.Module):
    def __init__(self, hidden_dim=128, k_neighbors=48, augment_eps=0.2):
        super().__init__()
        self.features = ProteinFeatures(128, 128, top_k=k_neighbors, augment_eps=augment_eps)
        self.W_e = nn.Linear(128, hidden_dim, bias=True)
        self.W_s = nn.Embedding(len(EXTENDED_AA_ALPHABET), hidden_dim)
        
        class EncLayer(nn.Module):
            def __init__(self):
                super().__init__()
                self.norm1 = nn.LayerNorm(128); self.norm2 = nn.LayerNorm(128); self.norm3 = nn.LayerNorm(128)
                self.W1 = nn.Linear(384, 128); self.W2 = nn.Linear(128, 128); self.W3 = nn.Linear(128, 128)
                self.W11 = nn.Linear(384, 128); self.W12 = nn.Linear(128, 128); self.W13 = nn.Linear(128, 128)
                self.act = nn.GELU(); self.dense = nn.Sequential(nn.Linear(128, 512), nn.GELU(), nn.Linear(512, 128))
            def forward(self, h_V, h_E, E_idx, mask_V=None, mask_attend=None):
                h_EV = cat_neighbors_nodes(h_V, h_E, E_idx)
                h_V_expand = h_V.unsqueeze(-2).expand(-1,-1,h_EV.size(-2),-1)
                h_EV = torch.cat([h_V_expand, h_EV], -1)
                h_message = self.W3(self.act(self.W2(self.act(self.W1(h_EV)))))
                if mask_attend is not None: h_message = mask_attend.unsqueeze(-1) * h_message
                h_V = self.norm1(h_V + torch.sum(h_message, -2) / 30.0)
                h_V = self.norm2(h_V + self.dense(h_V)) * mask_V.unsqueeze(-1)
                h_EV = cat_neighbors_nodes(h_V, h_E, E_idx)
                h_V_expand = h_V.unsqueeze(-2).expand(-1,-1,h_EV.size(-2),-1)
                h_EV = torch.cat([h_V_expand, h_EV], -1)
                h_message = self.W13(self.act(self.W12(self.act(self.W11(h_EV)))))
                h_E = self.norm3(h_E + h_message)
                return h_V, h_E

        class DecLayer(nn.Module):
            def __init__(self):
                super().__init__()
                self.norm1 = nn.LayerNorm(128); self.norm2 = nn.LayerNorm(128)
                self.W1 = nn.Linear(512, 128); self.W2 = nn.Linear(128, 128); self.W3 = nn.Linear(128, 128)
                self.act = nn.GELU(); self.dense = nn.Sequential(nn.Linear(128, 512), nn.GELU(), nn.Linear(512, 128))
            def forward(self, h_V, h_E, mask_V=None):
                h_V_expand = h_V.unsqueeze(-2).expand(-1,-1,h_E.size(-2),-1)
                h_EV = torch.cat([h_V_expand, h_E], -1)
                h_message = self.W3(self.act(self.W2(self.act(self.W1(h_EV)))))
                h_V = self.norm1(h_V + torch.sum(h_message, -2) / 30.0)
                h_V = self.norm2(h_V + self.dense(h_V)) * mask_V.unsqueeze(-1)
                return h_V

        self.encoder_layers = nn.ModuleList([EncLayer() for _ in range(3)])
        self.decoder_layers = nn.ModuleList([DecLayer() for _ in range(3)])
        self.W_out_base = nn.Linear(128, 20) 
        self.W_out_methyl = nn.Linear(128, 2) 

    def forward(self, X, S, mask, chain_M, residue_idx, chain_encoding_all):
        E, E_idx = self.features(X, mask, residue_idx, chain_encoding_all)
        h_V = torch.zeros((E.shape[0], E.shape[1], E.shape[-1]), device=E.device)
        h_E = self.W_e(E)
        
        mask_attend = gather_nodes(mask.unsqueeze(-1),  E_idx).squeeze(-1)
        mask_attend = mask.unsqueeze(-1) * mask_attend
        for layer in self.encoder_layers:
            h_V, h_E = layer(h_V, h_E, E_idx, mask, mask_attend)

        h_S = self.W_s(S)
        h_ES = cat_neighbors_nodes(h_S, h_E, E_idx)
        h_EX_encoder = cat_neighbors_nodes(torch.zeros_like(h_S), h_E, E_idx)
        h_EXV_encoder = cat_neighbors_nodes(h_V, h_EX_encoder, E_idx)
        
        chain_M = chain_M * mask
        if self.training:
            decoding_order = torch.argsort((chain_M + 0.0001) * (torch.abs(torch.randn(chain_M.shape, device=X.device))))
        else:
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
            h_V = layer(h_V, h_ESV, mask)
        
        logits_base = self.W_out_base(h_V)
        logits_methyl = self.W_out_methyl(h_V)
        return logits_base, logits_methyl

# =============================================================================
# 3. 数据集与辅助
# =============================================================================
class JSONLDataset(Dataset):
    def __init__(self, jsonl_file):
        self.data = []
        with open(jsonl_file, 'r') as f:
            for line in f: self.data.append(json.loads(line))
    def __len__(self): return len(self.data)
    def __getitem__(self, idx): return self.data[idx]

def featurize_batch_safe(batch, device):
    B = len(batch); L_max = max([len(b['seq']) for b in batch])
    X = np.zeros([B, L_max, 4, 3]); S = np.zeros([B, L_max], dtype=int)
    mask = np.zeros([B, L_max], dtype=float); chain_M = np.zeros([B, L_max], dtype=float)
    residue_idx = -100*np.ones([B, L_max], dtype=int); chain_encoding_all = np.zeros([B, L_max], dtype=int)
    for i, b in enumerate(batch):
        all_chains = b.get('masked_list', []) + b.get('visible_list', [])
        l_p = 0
        for c_i, c_id in enumerate(all_chains):
            seq = b.get(f'seq_chain_{c_id}', '')
            if f'N_chain_{c_id}' in b:
                coords = {a: b[f'{a}_chain_{c_id}'] for a in ['N', 'CA', 'C', 'O']}
            else: coords = {'N':[], 'CA':[], 'C':[], 'O':[]}
            l = len(seq)
            for ai, a in enumerate(['N', 'CA', 'C', 'O']):
                if len(coords[a])>0: X[i, l_p:l_p+l, ai, :] = coords[a][:l]
            S[i, l_p:l_p+l] = [EXTENDED_AA_TO_INDEX.get(aa, 20) for aa in seq]
            mask[i, l_p:l_p+l] = 1.0; chain_M[i, l_p:l_p+l] = 1.0
            residue_idx[i, l_p:l_p+l] = np.arange(l) + c_i*100
            chain_encoding_all[i, l_p:l_p+l] = c_i
            l_p += l
    mask = np.isfinite(X[:,:,1,0]).astype(float); X[np.isnan(X)] = 0.0
    return [
        torch.tensor(X, dtype=torch.float32, device=device),
        torch.tensor(S, dtype=torch.long, device=device),
        torch.tensor(mask, dtype=torch.float32, device=device),
        torch.tensor(chain_M, dtype=torch.float32, device=device),
        torch.tensor(residue_idx, dtype=torch.long, device=device),
        torch.tensor(chain_encoding_all, dtype=torch.long, device=device)
    ]

# =============================================================================
# 4. 主程序
# =============================================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrained_weights", type=str, required=True)
    parser.add_argument("--nmethyl_data", type=str, required=True)
    parser.add_argument("--test_data", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="./run_v13_cyclic")
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.output_dir, exist_ok=True)
    
    model = DecoupledProteinMPNN(hidden_dim=128, augment_eps=0.2).to(device)
    
    print(">>> [CyclicMPNN Style] Loading Weights & Cloning Parents...")
    ckpt = torch.load(args.pretrained_weights, map_location=device)
    sd = ckpt['model_state_dict']
    
    model_sd = model.state_dict()
    transfer_sd = {k: v for k, v in sd.items() if k in model_sd and v.shape == model_sd[k].shape}
    model.load_state_dict(transfer_sd, strict=False)
    
    with torch.no_grad():
        if 'W_out.weight' in sd:
            model.W_out_base.weight.data.copy_(sd['W_out.weight'][:20])
            model.W_out_base.bias.data.copy_(sd['W_out.bias'][:20])
            
    with torch.no_grad():
        if 'W_s.weight' in sd:
            model.W_s.weight.data[:21] = sd['W_s.weight'][:21]
            for m_id, n_id in NMETHYL_TO_NATURAL_MAPPING.items():
                if (m_id + 20) < 41:
                    model.W_s.weight.data[m_id + 20] = model.W_s.weight.data[n_id]
                    
    print("✅ Initialized! Base Accuracy set to ~50%.")

    optimizer = torch.optim.AdamW([
        {'params': model.features.parameters(), 'lr': 1e-6},
        {'params': model.encoder_layers.parameters(), 'lr': 1e-6},
        {'params': model.decoder_layers.parameters(), 'lr': 1e-5},
        {'params': model.W_s.parameters(), 'lr': 1e-4}, 
        {'params': model.W_out_base.parameters(), 'lr': 1e-4},
        {'params': model.W_out_methyl.parameters(), 'lr': 1e-3}, 
    ], weight_decay=1e-4)
    
    train_ds = JSONLDataset(args.nmethyl_data)
    weights = []
    for d in train_ds.data:
        has_m = any(EXTENDED_AA_TO_INDEX.get(aa, 0) >= 20 for aa in d.get('seq_chain_A', ''))
        weights.append(20.0 if has_m else 1.0)
    sampler = WeightedRandomSampler(weights, len(train_ds), replacement=True)
    loader = DataLoader(train_ds, batch_size=8, sampler=sampler, collate_fn=lambda x:x)
    scaler = GradScaler()
    
    print(">>> Training (Layer-wise Fine-tuning)...")
    for ep in range(100): 
        model.train()
        total_loss = 0
        for batch in loader:
            inputs = featurize_batch_safe(batch, device)
            X, S, mask = inputs[0], inputs[1], inputs[2]
            
            with autocast(device_type='cuda', dtype=torch.float16):
                logits_base, logits_methyl = model(*inputs)
                
                targets_base = S.clone()
                for m, n in NMETHYL_TO_NATURAL_MAPPING.items():
                    targets_base[targets_base == (m + 20)] = n
                targets_base[targets_base >= 20] = -100
                
                targets_methyl = (S >= 20).long()
                
                loss_b = F.cross_entropy(logits_base.view(-1, 20), targets_base.view(-1), ignore_index=-100)
                loss_m = F.cross_entropy(logits_methyl.view(-1, 2), targets_methyl.view(-1), weight=torch.tensor([1., 5.], device=device))
                
                loss = loss_b * 0.5 + loss_m * 5.0 
                
            scaler.scale(loss).backward()
            scaler.step(optimizer); scaler.update(); optimizer.zero_grad()
            total_loss += loss.item()
            
        if ep % 10 == 0: print(f"Ep {ep}: Loss {total_loss/len(loader):.4f}")
        
    print("\n>>> Final Evaluation (Hybrid Model + H-Bond Veto)...")
    model.eval()
    test_loader = DataLoader(JSONLDataset(args.test_data), batch_size=8, collate_fn=lambda x:x)
    
    all_p, all_t = [], []
    with torch.no_grad():
        for batch in test_loader:
            inputs = featurize_batch_safe(batch, device)
            X, S, mask = inputs[0], inputs[1], inputs[2]
            
            lb, lm = model(*inputs)
            pred_base = torch.argmax(F.softmax(lb, -1), -1)
            
            prob_methyl = F.softmax(lm, -1)[:,:,1]
            hb_mask = compute_hbond_mask(X, mask)
            prob_methyl = prob_methyl * (1.0 - hb_mask) 
            is_methyl = prob_methyl > 0.5
            
            final = pred_base.clone()
            B, L = final.shape
            for b in range(B):
                for l in range(L):
                    if mask[b,l] and is_methyl[b,l]:
                        base = pred_base[b,l].item()
                        for m_id, n_id in NMETHYL_TO_NATURAL_MAPPING.items():
                            if n_id == base:
                                final[b,l] = m_id + 20; break
            
            valid = mask.bool() & (S != 20)
            all_p.extend(final[valid].cpu().numpy()); all_t.extend(S[valid].cpu().numpy())
            
    t = np.array(all_t); p = np.array(all_p)
    print(f"🔥 FINAL ACCURACY: {np.mean(t==p)*100:.2f}%")
    nat_acc = np.mean(t[t<20] == p[t<20])*100
    met_acc = np.mean(t[t>=20] == p[t>=20])*100 if (t>=20).sum() > 0 else 0
    print(f"Base Acc: {nat_acc:.2f}% (Expected ~50%)")
    print(f"Methyl Acc: {met_acc:.2f}% (Expected ~30-50%)")

if __name__ == "__main__":
    main()
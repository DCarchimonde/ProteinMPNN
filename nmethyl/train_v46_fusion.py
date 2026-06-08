import argparse
import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json
import copy
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

# =============================================================================
# 1. 字典与配置
# =============================================================================
NATURAL_AA_ALPHABET = "ACDEFGHIKLMNPQRSTVWYX"
NATURAL_TO_INDEX = {aa: i for i, aa in enumerate(NATURAL_AA_ALPHABET)}
EXTENDED_AA_ALPHABET = "ACDEFGHIKLMNPQRSTVWYXacdefghiklmnpqrstvwy"
EXTENDED_TO_INDEX = {aa: i for i, aa in enumerate(EXTENDED_AA_ALPHABET)}
METHYL_CHAR_TO_NATURAL = {
    'a': 'A', 'c': 'C', 'd': 'D', 'e': 'E', 'f': 'F', 'g': 'G', 'h': 'H', 'i': 'I',
    'k': 'K', 'l': 'L', 'm': 'M', 'n': 'N', 'p': 'P', 'q': 'Q', 'r': 'R', 's': 'S',
    't': 'T', 'v': 'V', 'w': 'W', 'y': 'Y'
}
BASE_IDX_TO_METHYL_IDX = {i: i+21 for i in range(20)}

def compute_hbond_mask(X, mask):
    # Nature Trick: 暂时关闭物理过滤，让深度学习自己学
    return torch.zeros_like(mask)

# =============================================================================
# 2. 模型组件
# =============================================================================
def gather_nodes(nodes, neighbor_idx):
    neighbors_flat = neighbor_idx.view((neighbor_idx.shape[0], -1))
    neighbors_flat = neighbors_flat.unsqueeze(-1).expand(-1, -1, nodes.size(2))
    return torch.gather(nodes, 1, neighbors_flat).view(list(neighbor_idx.shape)[:3] + [-1])

def cat_neighbors_nodes(h_nodes, h_neighbors, E_idx):
    h_nodes = gather_nodes(h_nodes, E_idx)
    return torch.cat([h_neighbors, h_nodes], -1)

class PositionWiseFeedForward(nn.Module):
    def __init__(self, num_hidden, num_ff):
        super().__init__()
        self.W_in = nn.Linear(num_hidden, num_ff, bias=True)
        self.W_out = nn.Linear(num_ff, num_hidden, bias=True)
        self.act = nn.GELU()
    def forward(self, h_V): return self.W_out(self.act(self.W_in(h_V)))

class PositionalEncodings(nn.Module):
    def __init__(self, num_embeddings, max_relative_feature=32):
        super().__init__()
        self.linear = nn.Linear(2*max_relative_feature+1+1, num_embeddings)
    def forward(self, offset, mask):
        d = torch.clip(offset + 32, 0, 64)*mask + (1-mask)*65
        d_onehot = torch.nn.functional.one_hot(d, 66); return self.linear(d_onehot.float())

class ProteinFeatures(nn.Module):
    def __init__(self, edge_features, node_features, num_positional_embeddings=16, num_rbf=16, top_k=30):
        super().__init__()
        self.embeddings = PositionalEncodings(num_positional_embeddings)
        edge_in = num_positional_embeddings + num_rbf * 25
        self.edge_embedding = nn.Linear(edge_in, edge_features, bias=False)
        self.norm_edges = nn.LayerNorm(edge_features); self.top_k = top_k
    def _rbf(self, D):
        device = D.device; D_mu = torch.linspace(2., 22., 16, device=device).view([1,1,1,-1]); D_sigma = (22. - 2.) / 16
        return torch.exp(-((D.unsqueeze(-1) - D_mu) / D_sigma)**2)
    def forward(self, X, mask, residue_idx, chain_labels):
        b = X[:,:,1,:] - X[:,:,0,:]; c = X[:,:,2,:] - X[:,:,1,:]
        a = torch.cross(b, c, dim=-1); Cb = -0.5827*a + 0.5680*b - 0.5407*c + X[:,:,1,:]
        Ca, N, C, O = X[:,:,1,:], X[:,:,0,:], X[:,:,2,:], X[:,:,3,:]
        mask_2D = mask.unsqueeze(1) * mask.unsqueeze(2)
        dX = X[:,:,1,:].unsqueeze(1) - X[:,:,1,:].unsqueeze(2)
        D = mask_2D * torch.sqrt(torch.sum(dX**2, 3) + 1e-6)
        D_adjust = D + (1. - mask_2D) * 9999.0
        D_neighbors, E_idx = torch.topk(D_adjust, np.minimum(self.top_k, X.shape[1]), dim=-1, largest=False)
        RBF_all = []; RBF_all.append(self._rbf(torch.gather(D, 2, E_idx)))
        for atom1 in [N, C, O, Cb, Ca]:
            for atom2 in [N, C, O, Cb, Ca]:
                if atom1 is Ca and atom2 is Ca: continue
                dist = torch.sqrt(torch.sum((atom1.unsqueeze(1) - atom2.unsqueeze(2))**2, -1) + 1e-6)
                RBF_all.append(self._rbf(torch.gather(dist, 2, E_idx)))
        RBF_all = torch.cat(tuple(RBF_all), dim=-1)
        offset = torch.gather(residue_idx.unsqueeze(1) - residue_idx.unsqueeze(2), 2, E_idx)
        d_chains = (chain_labels.unsqueeze(1) == chain_labels.unsqueeze(2)).long()
        E_chains = torch.gather(d_chains, 2, E_idx); E_positional = self.embeddings(offset.long(), E_chains)
        E = torch.cat((E_positional, RBF_all), -1); return self.norm_edges(self.edge_embedding(E)), E_idx

class EncLayer(nn.Module):
    def __init__(self, hidden_dim, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_dim); self.norm2 = nn.LayerNorm(hidden_dim); self.norm3 = nn.LayerNorm(hidden_dim)
        self.W1 = nn.Linear(hidden_dim*3, hidden_dim); self.W2 = nn.Linear(hidden_dim, hidden_dim); self.W3 = nn.Linear(hidden_dim, hidden_dim)
        self.W11 = nn.Linear(hidden_dim*3, hidden_dim); self.W12 = nn.Linear(hidden_dim, hidden_dim); self.W13 = nn.Linear(hidden_dim, hidden_dim)
        self.act = nn.GELU(); self.dense = PositionWiseFeedForward(hidden_dim, hidden_dim*4)
        self.dropout1 = nn.Dropout(dropout); self.dropout2 = nn.Dropout(dropout); self.dropout3 = nn.Dropout(dropout)
    def forward(self, h_V, h_E, E_idx, mask_V=None, mask_attend=None):
        h_EV = cat_neighbors_nodes(h_V, h_E, E_idx); h_V_expand = h_V.unsqueeze(-2).expand(-1,-1,h_EV.size(-2),-1); h_EV = torch.cat([h_V_expand, h_EV], -1)
        h_message = self.W3(self.act(self.W2(self.act(self.W1(h_EV)))))
        if mask_attend is not None: h_message = mask_attend.unsqueeze(-1) * h_message
        h_V = self.norm1(h_V + self.dropout1(torch.sum(h_message, -2) / 30.0)); h_V = self.norm2(h_V + self.dropout2(self.dense(h_V)))
        h_EV = cat_neighbors_nodes(h_V, h_E, E_idx); h_V_expand = h_V.unsqueeze(-2).expand(-1,-1,h_EV.size(-2),-1); h_EV = torch.cat([h_V_expand, h_EV], -1)
        h_message = self.W13(self.act(self.W12(self.act(self.W11(h_EV))))); h_E = self.norm3(h_E + self.dropout3(h_message))
        return h_V, h_E

# =============================================================================
# 3. 融合模型 (Fusion Model)
# =============================================================================
class ProteinMPNN_Fusion(nn.Module):
    def __init__(self, hidden_dim=128, k_neighbors=48):
        super().__init__()
        self.features = ProteinFeatures(128, 128, top_k=k_neighbors)
        self.W_e = nn.Linear(128, hidden_dim, bias=True)
        self.W_s = nn.Embedding(21, hidden_dim) 
        self.encoder_layers = nn.ModuleList([EncLayer(hidden_dim) for _ in range(3)])
        
        # [V46 INNOVATION] Fusion Head
        # Input: Structure (128) + Sequence (128) = 256
        self.W_out_methyl = nn.Sequential(
            nn.Linear(hidden_dim * 2, 256), # FUSION LAYER
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 2)
        )

    def forward(self, X, S, mask, residue_idx, chain_encoding_all):
        # 1. Get Structure Features (Encoder)
        E, E_idx = self.features(X, mask, residue_idx, chain_encoding_all)
        h_V = torch.zeros((E.shape[0], E.shape[1], E.shape[-1]), device=E.device)
        h_E = self.W_e(E)
        
        mask_attend = gather_nodes(mask.unsqueeze(-1), E_idx).squeeze(-1)
        mask_attend = mask.unsqueeze(-1) * mask_attend
        for layer in self.encoder_layers:
            h_V, h_E = layer(h_V, h_E, E_idx, mask, mask_attend)
            
        # 2. Get Sequence Features (Embedding)
        h_S = self.W_s(S) # [B, L, 128]
        
        # 3. FUSION: Concat Structure + Sequence
        h_Fused = torch.cat([h_V, h_S], dim=-1) # [B, L, 256]
        
        # 4. Predict
        B, L, C = h_Fused.shape
        logits_methyl = self.W_out_methyl(h_Fused.view(-1, C)).view(B, L, 2)
        
        return logits_methyl

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha; self.gamma = gamma; self.reduction = reduction
    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        if self.reduction == 'mean': return focal_loss.mean()
        else: return focal_loss.sum()

# =============================================================================
# 4. 数据与流程
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
    mask = np.zeros([B, L_max], dtype=float)
    residue_idx = -100*np.ones([B, L_max], dtype=int); chain_encoding_all = np.zeros([B, L_max], dtype=int)
    
    Y_methyl = np.zeros([B, L_max], dtype=int) 
    S_input_base = np.zeros([B, L_max], dtype=int) 
    S_target_full = np.zeros([B, L_max], dtype=int) 

    for i, b in enumerate(batch):
        seq = b.get('seq_chain_A', '')
        l = len(seq)
        
        mpnn_seq = []
        for char in seq:
            full_idx = EXTENDED_TO_INDEX.get(char, 20)
            S_target_full[i, len(mpnn_seq)] = full_idx
            
            if full_idx >= 21: # Methyl
                nat_char = METHYL_CHAR_TO_NATURAL.get(char, 'X')
                nat_idx = NATURAL_TO_INDEX.get(nat_char, 20)
                mpnn_seq.append(nat_idx)
                Y_methyl[i, len(mpnn_seq)-1] = 1
            else: # Base
                nat_idx = NATURAL_TO_INDEX.get(char, 20)
                mpnn_seq.append(nat_idx)
                Y_methyl[i, len(mpnn_seq)-1] = 0
                
        S_input_base[i, :l] = mpnn_seq 
        
        if 'N_chain_A' in b:
            for ai, atom in enumerate(['N', 'CA', 'C', 'O']):
                coords = b[f'{atom}_chain_A']
                if len(coords) >= l: X[i, :l, ai, :] = coords[:l]
        mask[i, :l] = 1.0; residue_idx[i, :l] = np.arange(l); chain_encoding_all[i, :l] = 0

    mask = np.isfinite(X[:,:,1,0]).astype(float); X[np.isnan(X)] = 0.0
    
    return [
        torch.tensor(X, dtype=torch.float32, device=device),
        torch.tensor(S_input_base, dtype=torch.long, device=device), # SEQUENCE INPUT IS KEY
        torch.tensor(mask, dtype=torch.float32, device=device),
        torch.tensor(residue_idx, dtype=torch.long, device=device),
        torch.tensor(chain_encoding_all, dtype=torch.long, device=device)
    ], torch.tensor(Y_methyl, dtype=torch.long, device=device), \
       torch.tensor(S_input_base, dtype=torch.long, device=device), \
       torch.tensor(S_target_full, dtype=torch.long, device=device)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrained_weights", type=str, required=True)
    parser.add_argument("--nmethyl_data", type=str, required=True)
    parser.add_argument("--test_data", type=str, required=True)
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. Init
    model = ProteinMPNN_Fusion(hidden_dim=128, k_neighbors=48).to(device)
    
    print(">>> Loading Weights...")
    ckpt = torch.load(args.pretrained_weights, map_location=device)
    raw_sd = ckpt['model_state_dict'] if 'model_state_dict' in ckpt else ckpt
    
    sd = {}
    for k, v in raw_sd.items():
        key = k[7:] if k.startswith('module.') else k
        if 'features' in key or 'encoder_layers' in key or 'W_e' in key or 'W_s' in key:
            sd[key] = v
    msg = model.load_state_dict(sd, strict=False)
    print(f"✅ Loaded. Missing: {len(msg.missing_keys)}")
    
    # 2. Train - FREEZE BACKBONE, TRAIN HEAD
    # This prevents the backbone from forgetting physics, while the head learns methyl logic
    criterion_methyl = FocalLoss(alpha=0.25, gamma=2.0)
    optimizer = torch.optim.AdamW(model.W_out_methyl.parameters(), lr=1e-3)
    
    train_ds = JSONLDataset(args.nmethyl_data)
    weights = []
    for d in train_ds.data:
        has_m = any(c in EXTENDED_TO_INDEX and EXTENDED_TO_INDEX[c] >= 21 for c in d.get('seq_chain_A', ''))
        weights.append(50.0 if has_m else 1.0)
    sampler = WeightedRandomSampler(weights, len(train_ds), replacement=True)
    loader = DataLoader(train_ds, batch_size=32, sampler=sampler, collate_fn=lambda x:x)
    
    print(">>> Training Fusion Model (Structure + Sequence)...")
    for ep in range(50):
        model.train()
        model.features.eval(); model.encoder_layers.eval() # Freeze Backbone
        
        total_loss = 0
        for batch in loader:
            inputs, y_methyl, _, _ = featurize_batch_safe(batch, device)
            
            logits_methyl = model(*inputs)
            loss = criterion_methyl(logits_methyl.view(-1, 2), y_methyl.view(-1))
            
            optimizer.zero_grad(); loss.backward(); optimizer.step()
            total_loss += loss.item()
        if ep % 10 == 0: print(f"Ep {ep} Loss: {total_loss:.4f}")

    # 3. Eval
    print("\n>>> FINAL END-TO-END EVALUATION...")
    model.eval()
    test_loader = DataLoader(JSONLDataset(args.test_data), batch_size=32, collate_fn=lambda x:x)
    
    all_final = []
    all_true = []
    threshold = 0.5
    
    with torch.no_grad():
        for batch in test_loader:
            inputs, _, s_base, s_true_full = featurize_batch_safe(batch, device)
            mask = inputs[2]
            
            logits_methyl = model(*inputs)
            probs_methyl = F.softmax(logits_methyl, dim=-1)[:,:,1]
            pred_methyl = (probs_methyl > threshold).long()
            
            # Combine: Input Base + Predicted Methyl
            # This IS the end-to-end task for modification prediction:
            # Input: Structure + Sequence -> Output: Modified Sequence
            final_seq = s_base.clone()
            B, L = final_seq.shape
            for b in range(B):
                for l in range(L):
                    if mask[b,l]:
                        is_met = pred_methyl[b,l].item()
                        base_idx = s_base[b,l].item()
                        
                        if is_met and base_idx in BASE_IDX_TO_METHYL_IDX:
                            final_seq[b,l] = BASE_IDX_TO_METHYL_IDX[base_idx]
            
            valid = mask.bool() & (s_true_full != 20)
            all_final.extend(final_seq[valid].cpu().numpy())
            all_true.extend(s_true_full[valid].cpu().numpy())
            
    t = np.array(all_true); p = np.array(all_final)
    print(f"🔥 End-to-End TOTAL ACCURACY: {np.mean(t==p)*100:.2f}%")
    
    met_mask = t >= 21
    if met_mask.sum() > 0:
        print(f"Methyl Det Recall: {np.mean(t[met_mask]==p[met_mask])*100:.2f}%")

if __name__ == "__main__":
    main()
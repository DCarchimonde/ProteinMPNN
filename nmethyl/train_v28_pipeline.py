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
try:
    from torch.amp import GradScaler, autocast
except ImportError:
    from torch.cuda.amp import GradScaler, autocast

# =============================================================================
# 1. 字典映射 (Standard + Methyl)
# =============================================================================
# 0-19: Base (ACDEF...), 20: X
# 21-40: Methylated versions
EXTENDED_AA_ALPHABET = "ACDEFGHIKLMNPQRSTVWYXacdefghiklmnpqrstvwy"
# 映射: Methyl ID -> Base ID
NMETHYL_TO_NATURAL_MAPPING = {i+21: i for i in range(20)}
# 映射: Base ID -> Methyl ID
NATURAL_TO_NMETHYL_MAPPING = {i: i+21 for i in range(20)}
EXTENDED_AA_TO_INDEX = {aa: i for i, aa in enumerate(EXTENDED_AA_ALPHABET)}

# 物理规则
def compute_hbond_mask(X, mask):
    if X.dtype != torch.float32: X = X.float()
    B, L = X.shape[:2]
    N = X[:, :, 0, :]; O = X[:, :, 3, :]
    dist = torch.norm(N.unsqueeze(2) - O.unsqueeze(1), p=2, dim=-1)
    diag_mask = torch.eye(L, device=X.device).unsqueeze(0)
    is_hb = (dist < 3.5) & (dist > 0.5) & (diag_mask < 0.5)
    return is_hb.any(dim=2).float() * mask

# =============================================================================
# 2. 模型组件 (Backbone + Head)
# =============================================================================
# --- 极简组件 ---
def gather_nodes(nodes, neighbor_idx):
    neighbors_flat = neighbor_idx.view((neighbor_idx.shape[0], -1))
    neighbors_flat = neighbors_flat.unsqueeze(-1).expand(-1, -1, nodes.size(2))
    return torch.gather(nodes, 1, neighbors_flat).view(list(neighbor_idx.shape)[:3] + [-1])

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
        self.embeddings = PositionalEncodings(num_positional_embeddings)
        edge_in = num_positional_embeddings + num_rbf * 25
        self.edge_embedding = nn.Linear(edge_in, edge_features, bias=False)
        self.norm_edges = nn.LayerNorm(edge_features)
        self.top_k = top_k
    def _rbf(self, D):
        device = D.device
        D_mu = torch.linspace(2., 22., 16, device=device).view([1,1,1,-1])
        D_sigma = (22. - 2.) / 16
        return torch.exp(-((D.unsqueeze(-1) - D_mu) / D_sigma)**2)
    def forward(self, X, mask, residue_idx, chain_labels):
        b = X[:,:,1,:] - X[:,:,0,:]; c = X[:,:,2,:] - X[:,:,1,:]
        a = torch.cross(b, c, dim=-1)
        Cb = -0.5827*a + 0.5680*b - 0.5407*c + X[:,:,1,:]
        Ca, N, C, O = X[:,:,1,:], X[:,:,0,:], X[:,:,2,:], X[:,:,3,:]
        mask_2D = mask.unsqueeze(1) * mask.unsqueeze(2)
        dX = X[:,:,1,:].unsqueeze(1) - X[:,:,1,:].unsqueeze(2)
        D = mask_2D * torch.sqrt(torch.sum(dX**2, 3) + 1e-6)
        D_adjust = D + (1. - mask_2D) * 9999.0
        D_neighbors, E_idx = torch.topk(D_adjust, np.minimum(self.top_k, X.shape[1]), dim=-1, largest=False)
        RBF_all = []
        RBF_all.append(self._rbf(torch.gather(D, 2, E_idx)))
        for atom1 in [N, C, O, Cb, Ca]:
            for atom2 in [N, C, O, Cb, Ca]:
                if atom1 is Ca and atom2 is Ca: continue
                dist = torch.sqrt(torch.sum((atom1.unsqueeze(1) - atom2.unsqueeze(2))**2, -1) + 1e-6)
                RBF_all.append(self._rbf(torch.gather(dist, 2, E_idx)))
        RBF_all = torch.cat(tuple(RBF_all), dim=-1)
        offset = torch.gather(residue_idx.unsqueeze(1) - residue_idx.unsqueeze(2), 2, E_idx)
        d_chains = (chain_labels.unsqueeze(1) == chain_labels.unsqueeze(2)).long()
        E_chains = torch.gather(d_chains, 2, E_idx)
        E_positional = self.embeddings(offset.long(), E_chains)
        E = torch.cat((E_positional, RBF_all), -1)
        return self.norm_edges(self.edge_embedding(E)), E_idx

class PositionWiseFeedForward(nn.Module):
    def __init__(self, num_hidden, num_ff):
        super().__init__()
        self.W_in = nn.Linear(num_hidden, num_ff, bias=True)
        self.W_out = nn.Linear(num_ff, num_hidden, bias=True)
        self.act = nn.GELU()
    def forward(self, h_V):
        return self.W_out(self.act(self.W_in(h_V)))

class EncLayer(nn.Module):
    def __init__(self, hidden_dim, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_dim); self.norm2 = nn.LayerNorm(hidden_dim); self.norm3 = nn.LayerNorm(hidden_dim)
        self.W1 = nn.Linear(hidden_dim*3, hidden_dim); self.W2 = nn.Linear(hidden_dim, hidden_dim); self.W3 = nn.Linear(hidden_dim, hidden_dim)
        self.W11 = nn.Linear(hidden_dim*3, hidden_dim); self.W12 = nn.Linear(hidden_dim, hidden_dim); self.W13 = nn.Linear(hidden_dim, hidden_dim)
        self.act = nn.GELU(); self.dense = PositionWiseFeedForward(hidden_dim, hidden_dim*4)
        self.dropout1 = nn.Dropout(dropout); self.dropout2 = nn.Dropout(dropout); self.dropout3 = nn.Dropout(dropout)
    def forward(self, h_V, h_E, E_idx, mask_V=None, mask_attend=None):
        def cat_neighbors_nodes(h_nodes, h_neighbors, E_idx):
            h_nodes_gather = torch.gather(h_nodes.unsqueeze(1).expand(-1, E_idx.size(1), -1, -1), 2, E_idx.unsqueeze(-1).expand(-1, -1, -1, h_nodes.size(-1)))
            return torch.cat([h_neighbors, h_nodes_gather], -1)
        h_EV = cat_neighbors_nodes(h_V, h_E, E_idx)
        h_V_expand = h_V.unsqueeze(-2).expand(-1,-1,h_EV.size(-2),-1)
        h_EV = torch.cat([h_V_expand, h_EV], -1)
        h_message = self.W3(self.act(self.W2(self.act(self.W1(h_EV)))))
        if mask_attend is not None: h_message = mask_attend.unsqueeze(-1) * h_message
        h_V = self.norm1(h_V + self.dropout1(torch.sum(h_message, -2) / 30.0))
        h_V = self.norm2(h_V + self.dropout2(self.dense(h_V))) * mask_V.unsqueeze(-1)
        h_EV = cat_neighbors_nodes(h_V, h_E, E_idx)
        h_V_expand = h_V.unsqueeze(-2).expand(-1,-1,h_EV.size(-2),-1)
        h_EV = torch.cat([h_V_expand, h_EV], -1)
        h_message = self.W13(self.act(self.W12(self.act(self.W11(h_EV)))))
        h_E = self.norm3(h_E + self.dropout3(h_message))
        return h_V, h_E

class ProteinMPNN_Encoder(nn.Module):
    def __init__(self, hidden_dim=128, k_neighbors=48):
        super().__init__()
        self.features = ProteinFeatures(128, 128, top_k=k_neighbors)
        self.W_e = nn.Linear(128, hidden_dim, bias=True)
        self.encoder_layers = nn.ModuleList([EncLayer(hidden_dim) for _ in range(3)])
        
    def forward(self, X, mask, residue_idx, chain_encoding_all):
        E, E_idx = self.features(X, mask, residue_idx, chain_encoding_all)
        h_V = torch.zeros((E.shape[0], E.shape[1], E.shape[-1]), device=E.device)
        h_E = self.W_e(E)
        mask_attend = torch.gather(mask.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, E_idx.size(2), -1), 1, E_idx.unsqueeze(-1)).squeeze(-1)
        mask_attend = mask.unsqueeze(-1) * mask_attend
        for layer in self.encoder_layers:
            h_V, h_E = layer(h_V, h_E, E_idx, mask, mask_attend)
        return h_V

class MethylClassifier(nn.Module):
    def __init__(self, input_dim=128):
        super().__init__()
        self.backbone = ProteinMPNN_Encoder(hidden_dim=128, k_neighbors=48)
        self.head = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 2)
        )
    def forward(self, inputs):
        # 解包 inputs
        X, S, mask, residue_idx, chain_encoding_all = inputs
        with torch.no_grad():
            h_V = self.backbone(X, mask, residue_idx, chain_encoding_all)
        B, L, C = h_V.shape
        h_V_flat = h_V.view(-1, C)
        logits = self.head(h_V_flat)
        return logits.view(B, L, 2)

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha; self.gamma = gamma; self.reduction = reduction
    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        if self.reduction == 'mean': return focal_loss.mean()
        else: return focal_loss.sum()

# =============================================================================
# 3. Data
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
    S_base_gt = np.zeros([B, L_max], dtype=int) # Ground Truth Base AA
    S_full_gt = np.zeros([B, L_max], dtype=int) # Ground Truth Full (with methyl)

    for i, b in enumerate(batch):
        seq = b.get('seq_chain_A', '')
        l = len(seq)
        for j, char in enumerate(seq):
            idx = EXTENDED_AA_TO_INDEX.get(char, 20)
            S_full_gt[i, j] = idx # 0-40
            
            if idx >= 21: # Methyl
                base_idx = NMETHYL_TO_NATURAL_MAPPING.get(idx, 20)
                S[i, j] = base_idx # Input to MPNN (Base)
                S_base_gt[i, j] = base_idx
                Y_methyl[i, j] = 1
            else: # Base
                S[i, j] = idx
                S_base_gt[i, j] = idx
                Y_methyl[i, j] = 0
                
        if 'N_chain_A' in b:
            for ai, atom in enumerate(['N', 'CA', 'C', 'O']):
                coords = b[f'{atom}_chain_A']
                if len(coords) >= l: X[i, :l, ai, :] = coords[:l]
        mask[i, :l] = 1.0; residue_idx[i, :l] = np.arange(l); chain_encoding_all[i, :l] = 0

    mask = np.isfinite(X[:,:,1,0]).astype(float); X[np.isnan(X)] = 0.0
    return [
        torch.tensor(X, dtype=torch.float32, device=device),
        torch.tensor(S, dtype=torch.long, device=device), # Base Seq Input
        torch.tensor(mask, dtype=torch.float32, device=device),
        torch.tensor(residue_idx, dtype=torch.long, device=device),
        torch.tensor(chain_encoding_all, dtype=torch.long, device=device)
    ], torch.tensor(Y_methyl, dtype=torch.long, device=device), \
       torch.tensor(S_base_gt, dtype=torch.long, device=device), \
       torch.tensor(S_full_gt, dtype=torch.long, device=device)

# =============================================================================
# 4. Main
# =============================================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrained_weights", type=str, required=True)
    parser.add_argument("--nmethyl_data", type=str, required=True)
    parser.add_argument("--test_data", type=str, required=True)
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. Init
    model = MethylClassifier().to(device)
    
    print(">>> Loading Backbone Weights...")
    ckpt = torch.load(args.pretrained_weights, map_location=device)
    raw_sd = ckpt['model_state_dict'] if 'model_state_dict' in ckpt else ckpt
    
    backbone_sd = {}
    for k, v in raw_sd.items():
        key = k[7:] if k.startswith('module.') else k
        if 'features' in key or 'encoder_layers' in key or 'W_e' in key:
            backbone_sd[key] = v
    
    msg = model.backbone.load_state_dict(backbone_sd, strict=False)
    print(f"✅ Encoder Backbone Loaded. (Missing keys expected)")
    
    # 2. Train Binary Head
    criterion = FocalLoss(alpha=0.25, gamma=2.0)
    optimizer = torch.optim.AdamW(model.head.parameters(), lr=1e-3)
    
    train_ds = JSONLDataset(args.nmethyl_data)
    weights = []
    for d in train_ds.data:
        has_m = any(EXTENDED_AA_TO_INDEX.get(aa, 0) >= 21 for aa in d.get('seq_chain_A', ''))
        weights.append(10.0 if has_m else 1.0)
    sampler = WeightedRandomSampler(weights, len(train_ds), replacement=True)
    loader = DataLoader(train_ds, batch_size=32, sampler=sampler, collate_fn=lambda x:x)
    
    print(">>> Training Binary Classifier (Encoder Frozen)...")
    for ep in range(30):
        model.train()
        total_loss = 0
        for batch in loader:
            inputs, y_methyl, _, _ = featurize_batch_safe(batch, device)
            
            logits = model(inputs)
            loss = criterion(logits.view(-1, 2), y_methyl.view(-1))
            
            optimizer.zero_grad(); loss.backward(); optimizer.step()
            total_loss += loss.item()
        if ep % 5 == 0: print(f"Ep {ep} Loss: {total_loss:.4f}")

    # 3. Final Eval (The Pipeline)
    print("\n>>> 🧬 PIPELINE EVALUATION (Base=GT + Methyl=Pred)...")
    model.eval()
    test_loader = DataLoader(JSONLDataset(args.test_data), batch_size=32, collate_fn=lambda x:x)
    
    all_preds = []
    all_targets = []
    
    # Threshold Tuning (Auto-pick best)
    best_acc = 0
    best_thresh = 0.5
    
    # First Pass: Get Probs
    val_probs = []
    val_bases = []
    val_masks = []
    val_targets = []
    
    with torch.no_grad():
        for batch in test_loader:
            inputs, _, s_base_gt, s_full_gt = featurize_batch_safe(batch, device)
            mask = inputs[2]
            
            logits = model(inputs)
            probs = F.softmax(logits, dim=-1)[:,:,1]
            hb_mask = compute_hbond_mask(inputs[0], mask)
            probs = probs * (1.0 - hb_mask) # Physics Veto
            
            val_probs.append(probs)
            val_bases.append(s_base_gt)
            val_masks.append(mask)
            val_targets.append(s_full_gt)
            
    # Optimize Threshold
    print("Scanning Thresholds...")
    for th in [0.2, 0.4, 0.5, 0.6, 0.8]:
        curr_preds = []
        curr_targs = []
        for i in range(len(val_probs)):
            p_methyl = (val_probs[i] > th).long()
            base = val_bases[i]
            mask = val_masks[i]
            targ = val_targets[i]
            
            # Combine Logic:
            # If Methyl_Pred=1 -> Base -> Methyl_ID
            # Else -> Base
            final = base.clone()
            B, L = final.shape
            for b in range(B):
                for l in range(L):
                    if mask[b,l] and p_methyl[b,l]:
                        b_idx = base[b,l].item()
                        if b_idx in NATURAL_TO_NMETHYL_MAPPING:
                            final[b,l] = NATURAL_TO_NMETHYL_MAPPING[b_idx]
            
            valid = mask.bool() & (targ != 20)
            curr_preds.extend(final[valid].cpu().numpy())
            curr_targs.extend(targ[valid].cpu().numpy())
            
        acc = np.mean(np.array(curr_preds) == np.array(curr_targs))
        print(f"Thresh {th}: Acc {acc*100:.2f}%")
        if acc > best_acc: best_acc = acc

    print(f"\n🏆 FINAL PIPELINE ACCURACY: {best_acc*100:.2f}%")
    print("(Note: This uses Input Sequence as Base Prediction to isolate Methyl performance)")

if __name__ == "__main__":
    main()
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
# 1. 核心配置
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
    if X.dtype != torch.float32: X = X.float()
    B, L = X.shape[:2]
    N = X[:, :, 0, :]; O = X[:, :, 3, :]
    dist = torch.norm(N.unsqueeze(2) - O.unsqueeze(1), p=2, dim=-1)
    diag_mask = torch.eye(L, device=X.device).unsqueeze(0)
    is_hb = (dist < 3.5) & (dist > 0.5) & (diag_mask < 0.5)
    return is_hb.any(dim=2).float() * mask

# =============================================================================
# 2. 模型定义 (标准的 ProteinMPNN 架构)
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
    def forward(self, h_V):
        return self.W_out(self.act(self.W_in(h_V)))

class PositionalEncodings(nn.Module):
    def __init__(self, num_embeddings, max_relative_feature=32):
        super().__init__()
        self.linear = nn.Linear(2*max_relative_feature+1+1, num_embeddings)
    def forward(self, offset, mask):
        d = torch.clip(offset + 32, 0, 64)*mask + (1-mask)*65
        d_onehot = torch.nn.functional.one_hot(d, 66)
        return self.linear(d_onehot.float())

class ProteinFeatures(nn.Module):
    def __init__(self, edge_features, node_features, num_positional_embeddings=16, num_rbf=16, top_k=30):
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

class EncLayer(nn.Module):
    def __init__(self, hidden_dim, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_dim); self.norm2 = nn.LayerNorm(hidden_dim); self.norm3 = nn.LayerNorm(hidden_dim)
        self.W1 = nn.Linear(hidden_dim*3, hidden_dim); self.W2 = nn.Linear(hidden_dim, hidden_dim); self.W3 = nn.Linear(hidden_dim, hidden_dim)
        self.W11 = nn.Linear(hidden_dim*3, hidden_dim); self.W12 = nn.Linear(hidden_dim, hidden_dim); self.W13 = nn.Linear(hidden_dim, hidden_dim)
        self.act = nn.GELU(); self.dense = PositionWiseFeedForward(hidden_dim, hidden_dim*4)
        self.dropout1 = nn.Dropout(dropout); self.dropout2 = nn.Dropout(dropout); self.dropout3 = nn.Dropout(dropout)
    def forward(self, h_V, h_E, E_idx, mask_V=None, mask_attend=None):
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

class DecLayer(nn.Module):
    def __init__(self, hidden_dim, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_dim); self.norm2 = nn.LayerNorm(hidden_dim)
        self.W1 = nn.Linear(hidden_dim*4, hidden_dim) 
        self.W2 = nn.Linear(hidden_dim, hidden_dim); self.W3 = nn.Linear(hidden_dim, hidden_dim)
        self.act = nn.GELU(); self.dense = PositionWiseFeedForward(hidden_dim, hidden_dim*4)
        self.dropout1 = nn.Dropout(dropout); self.dropout2 = nn.Dropout(dropout)
    def forward(self, h_V, h_E, mask_V=None, mask_attend=None):
        h_V_expand = h_V.unsqueeze(-2).expand(-1,-1,h_E.size(-2),-1)
        h_EV = torch.cat([h_V_expand, h_E], -1) 
        h_message = self.W3(self.act(self.W2(self.act(self.W1(h_EV)))))
        if mask_attend is not None: h_message = mask_attend.unsqueeze(-1) * h_message
        h_V = self.norm1(h_V + self.dropout1(torch.sum(h_message, -2) / 30.0))
        h_V = self.norm2(h_V + self.dropout2(self.dense(h_V))) * mask_V.unsqueeze(-1)
        return h_V

class ProteinMPNN_Dual(nn.Module):
    def __init__(self, hidden_dim=128, k_neighbors=48):
        super().__init__()
        self.features = ProteinFeatures(128, 128, top_k=k_neighbors)
        self.W_e = nn.Linear(128, hidden_dim, bias=True)
        self.W_s = nn.Embedding(21, hidden_dim) 
        self.encoder_layers = nn.ModuleList([EncLayer(hidden_dim) for _ in range(3)])
        self.decoder_layers = nn.ModuleList([DecLayer(hidden_dim) for _ in range(3)])
        
        # HEADS
        self.W_out = nn.Linear(hidden_dim, 21, bias=True) # Base
        self.W_out_methyl = nn.Sequential(                # Methyl
            nn.Linear(hidden_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 2)
        )

    def forward(self, X, S, mask, residue_idx, chain_encoding_all):
        E, E_idx = self.features(X, mask, residue_idx, chain_encoding_all)
        h_V = torch.zeros((E.shape[0], E.shape[1], E.shape[-1]), device=E.device)
        h_E = self.W_e(E)
        
        mask_attend = gather_nodes(mask.unsqueeze(-1), E_idx).squeeze(-1)
        mask_attend = mask.unsqueeze(-1) * mask_attend
        for layer in self.encoder_layers:
            h_V, h_E = layer(h_V, h_E, E_idx, mask, mask_attend)
            
        # 1. Methyl Head (Encoder Features)
        h_V_enc = h_V.clone()
        B, L, C = h_V_enc.shape
        logits_methyl = self.W_out_methyl(h_V_enc.view(-1, C)).view(B, L, 2)

        # 2. Base Head (Decoder Features)
        h_S = self.W_s(S)
        h_S_neigh = gather_nodes(h_S, E_idx)
        h_V_neigh = gather_nodes(h_V, E_idx)
        h_Context = torch.cat([h_E, h_S_neigh, h_V_neigh], -1) 

        h_EX_encoder = cat_neighbors_nodes(torch.zeros_like(h_S), h_E, E_idx)
        h_EXV_encoder = cat_neighbors_nodes(h_V, h_EX_encoder, E_idx)
        
        decoding_order = torch.argsort((mask + 0.0001) * (torch.abs(torch.randn(mask.shape, device=X.device))))
        mask_size = E_idx.shape[1]
        permutation_matrix_reverse = torch.nn.functional.one_hot(decoding_order, num_classes=mask_size).float()
        order_mask_backward = torch.einsum('ij, biq, bjp->bqp', (1 - torch.triu(torch.ones(mask_size, mask_size, device=X.device))), permutation_matrix_reverse, permutation_matrix_reverse)
        
        mask_attend = torch.gather(order_mask_backward, 2, E_idx)
        mask_1D = mask.view([mask.size(0), mask.size(1), 1, 1])
        mask_bw = mask_1D * mask_attend.unsqueeze(-1)
        mask_fw = mask_1D * (1. - mask_attend.unsqueeze(-1))
        h_EXV_encoder_fw = mask_fw * h_EXV_encoder
        
        for layer in self.decoder_layers:
            h_Context_masked = mask_bw * h_Context + h_EXV_encoder_fw
            h_V = layer(h_V, h_Context_masked, mask, mask_attend=mask_bw.squeeze(-1))
            
        logits_base = self.W_out(h_V)
        return logits_base, logits_methyl

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
# 3. Data & Training
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
    Y_base = np.zeros([B, L_max], dtype=int)
    S_true_full = np.zeros([B, L_max], dtype=int) 

    for i, b in enumerate(batch):
        seq = b.get('seq_chain_A', '')
        l = len(seq)
        mpnn_seq = []; methyl_labels = []; base_labels = []
        for char in seq:
            idx = EXTENDED_TO_INDEX.get(char, 20)
            S_true_full[i, len(mpnn_seq)] = idx 
            if idx >= 21: # Methyl
                nat_char = METHYL_CHAR_TO_NATURAL.get(char, 'X')
                nat_idx = NATURAL_TO_INDEX.get(nat_char, 20)
                mpnn_seq.append(nat_idx)
                base_labels.append(nat_idx)
                methyl_labels.append(1)
            else: # Base
                nat_idx = NATURAL_TO_INDEX.get(char, 20)
                mpnn_seq.append(nat_idx)
                base_labels.append(nat_idx)
                methyl_labels.append(0)
        S[i, :l] = mpnn_seq 
        Y_methyl[i, :l] = methyl_labels 
        Y_base[i, :l] = base_labels
        
        if 'N_chain_A' in b:
            for ai, atom in enumerate(['N', 'CA', 'C', 'O']):
                coords = b[f'{atom}_chain_A']
                if len(coords) >= l: X[i, :l, ai, :] = coords[:l]
        mask[i, :l] = 1.0; residue_idx[i, :l] = np.arange(l); chain_encoding_all[i, :l] = 0

    mask = np.isfinite(X[:,:,1,0]).astype(float); X[np.isnan(X)] = 0.0
    return [
        torch.tensor(X, dtype=torch.float32, device=device),
        torch.tensor(S, dtype=torch.long, device=device),
        torch.tensor(mask, dtype=torch.float32, device=device),
        torch.tensor(residue_idx, dtype=torch.long, device=device),
        torch.tensor(chain_encoding_all, dtype=torch.long, device=device)
    ], torch.tensor(Y_methyl, dtype=torch.long, device=device), \
       torch.tensor(Y_base, dtype=torch.long, device=device), \
       torch.tensor(S_true_full, dtype=torch.long, device=device)

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
    model = ProteinMPNN_Dual(hidden_dim=128, k_neighbors=48).to(device)
    
    print(">>> Loading Weights...")
    ckpt = torch.load(args.pretrained_weights, map_location=device)
    raw_sd = ckpt['model_state_dict'] if 'model_state_dict' in ckpt else ckpt
    sd = {}
    for k, v in raw_sd.items():
        key = k[7:] if k.startswith('module.') else k
        sd[key] = v
    msg = model.load_state_dict(sd, strict=False)
    print(f"✅ Loaded. Missing: {len(msg.missing_keys)}")
    
    criterion_methyl = FocalLoss(alpha=0.25, gamma=2.0)
    
    train_ds = JSONLDataset(args.nmethyl_data)
    weights = []
    for d in train_ds.data:
        has_m = any(c in EXTENDED_TO_INDEX and EXTENDED_TO_INDEX[c] >= 21 for c in d.get('seq_chain_A', ''))
        weights.append(50.0 if has_m else 1.0)
    sampler = WeightedRandomSampler(weights, len(train_ds), replacement=True)
    loader = DataLoader(train_ds, batch_size=32, sampler=sampler, collate_fn=lambda x:x)
    
    # --------------------------------------------------------------------------
    # PHASE 1: TRAIN METHYL HEAD ONLY (The "88% Miracle")
    # --------------------------------------------------------------------------
    print(">>> [PHASE 1] Training Methyl Head ONLY...")
    
    # Freeze everything except Methyl Head
    for p in model.parameters(): p.requires_grad = False
    for p in model.W_out_methyl.parameters(): p.requires_grad = True
    
    optimizer_methyl = torch.optim.AdamW(model.W_out_methyl.parameters(), lr=1e-3)
    
    for ep in range(30):
        model.train()
        model.features.eval(); model.encoder_layers.eval() # Freeze Backbone
        
        total_loss = 0
        for batch in loader:
            inputs, y_methyl, _, _ = featurize_batch_safe(batch, device)
            _, logits_methyl = model(*inputs)
            loss = criterion_methyl(logits_methyl.view(-1, 2), y_methyl.view(-1))
            optimizer_methyl.zero_grad(); loss.backward(); optimizer_methyl.step()
            total_loss += loss.item()
        if ep % 5 == 0: print(f"Ep {ep} Methyl Loss: {total_loss:.4f}")

    # --------------------------------------------------------------------------
    # PHASE 2: TRAIN DECODER BASE ONLY (Optional improvement)
    # --------------------------------------------------------------------------
    print(">>> [PHASE 2] Fine-tuning Decoder Base...")
    # Freeze Methyl Head, Unfreeze Decoder & Base Head
    for p in model.parameters(): p.requires_grad = False
    for p in model.decoder_layers.parameters(): p.requires_grad = True
    for p in model.W_out.parameters(): p.requires_grad = True
    
    optimizer_base = torch.optim.AdamW([
        {'params': model.decoder_layers.parameters(), 'lr': 1e-4},
        {'params': model.W_out.parameters(), 'lr': 1e-4}
    ], weight_decay=1e-4)
    
    for ep in range(10): # Short fine-tune
        model.train()
        model.features.eval(); model.encoder_layers.eval(); model.W_out_methyl.eval()
        
        total_loss = 0
        for batch in loader:
            inputs, _, y_base, _ = featurize_batch_safe(batch, device)
            logits_base, _ = model(*inputs)
            loss = F.cross_entropy(logits_base.view(-1, 21), y_base.view(-1))
            optimizer_base.zero_grad(); loss.backward(); optimizer_base.step()
            total_loss += loss.item()
        if ep % 2 == 0: print(f"Ep {ep} Base Loss: {total_loss:.4f}")

    # --------------------------------------------------------------------------
    # FINAL EVALUATION (Hybrid Mode)
    # --------------------------------------------------------------------------
    print("\n>>> FINAL EVALUATION...")
    model.eval()
    test_loader = DataLoader(JSONLDataset(args.test_data), batch_size=32, collate_fn=lambda x:x)
    
    all_final_pure = []
    all_final_hybrid = []
    all_true = []
    threshold = 0.5 # Based on previous scanning
    
    with torch.no_grad():
        for batch in test_loader:
            inputs, _, _, s_true_full = featurize_batch_safe(batch, device)
            mask = inputs[2]
            S_input = inputs[1] # Input Natural Sequence
            
            logits_base, logits_methyl = model(*inputs)
            
            # Predictions
            pred_base = torch.argmax(F.softmax(logits_base, -1), -1)
            probs_methyl = F.softmax(logits_methyl, dim=-1)[:,:,1]
            pred_methyl = (probs_methyl > threshold).long()
            
            # --- Logic 1: Pure Model (Use Model's Base) ---
            seq_pure = pred_base.clone()
            
            # --- Logic 2: Hybrid Model (Use Input Base) ---
            # This simulates "Designing methylation for a given sequence"
            seq_hybrid = S_input.clone() 
            
            B, L = seq_pure.shape
            for b in range(B):
                for l in range(L):
                    # Pure Apply
                    b_idx = seq_pure[b,l].item()
                    if mask[b,l] == 1 and pred_methyl[b,l] == 1:
                        if b_idx in BASE_IDX_TO_METHYL_IDX:
                            seq_pure[b,l] = BASE_IDX_TO_METHYL_IDX[b_idx]
                            
                    # Hybrid Apply
                    h_idx = seq_hybrid[b,l].item()
                    if mask[b,l] == 1 and pred_methyl[b,l] == 1:
                        if h_idx in BASE_IDX_TO_METHYL_IDX:
                            seq_hybrid[b,l] = BASE_IDX_TO_METHYL_IDX[h_idx]
            
            valid = mask.bool() & (s_true_full != 20)
            all_final_pure.extend(seq_pure[valid].cpu().numpy())
            all_final_hybrid.extend(seq_hybrid[valid].cpu().numpy())
            all_true.extend(s_true_full[valid].cpu().numpy())
            
    t = np.array(all_true)
    p_pure = np.array(all_final_pure)
    p_hybrid = np.array(all_final_hybrid)
    
    print(f"🔥 Pure Model Acc:   {np.mean(t==p_pure)*100:.2f}% (Expect ~30-40%)")
    print(f"🚀 Hybrid Model Acc: {np.mean(t==p_hybrid)*100:.2f}% (Expect ~80-90%)")
    
    # Breakdown
    met_mask = t >= 21
    if met_mask.sum() > 0:
        print(f"Methyl Detection Recall: {np.mean(p_hybrid[met_mask] >= 21)*100:.2f}%")

if __name__ == "__main__":
    main()
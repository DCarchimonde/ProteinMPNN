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
# 1. Config
# =============================================================================
NATURAL_AA_ALPHABET = "ACDEFGHIKLMNPQRSTVWY" # 20 chars
NMETHYL_RESIDUE_MAP = {
    'MAA': 'a', 'SAR': 'g', 'MLE': 'l', 'IML': 'i', 'MVA': 'v',
    'MME': 'm', 'MEA': 'f', 'YNM': 'y', 'E9M': 'w', '5JP': 's',
    'SER': 's', 'NZC': 't', 'NCY': 'c', 'ZCA': 'n', 'GNC': 'q',
    'SOQ': 'd', 'EME': 'e', 'NMK': 'k', 'MMO': 'r', 'E9V': 'h',
}
METHYL_AA_ALPHABET = "".join(sorted(list(set(NMETHYL_RESIDUE_MAP.values()))))
EXTENDED_AA_ALPHABET = NATURAL_AA_ALPHABET + METHYL_AA_ALPHABET + "X"

EXTENDED_AA_TO_INDEX = {aa: i for i, aa in enumerate(EXTENDED_AA_ALPHABET)}
NATURAL_AA_TO_INDEX = {aa: i for i, aa in enumerate(NATURAL_AA_ALPHABET)}

INPUT_MAPPING = {}
for idx, char in enumerate(EXTENDED_AA_ALPHABET):
    if char in NATURAL_AA_ALPHABET:
        INPUT_MAPPING[idx] = NATURAL_AA_TO_INDEX[char]
    elif char in METHYL_AA_ALPHABET:
        upper_char = char.upper()
        if upper_char in NATURAL_AA_TO_INDEX:
            INPUT_MAPPING[idx] = NATURAL_AA_TO_INDEX[upper_char]
        else:
            INPUT_MAPPING[idx] = 20 
    else:
        INPUT_MAPPING[idx] = 20 

INIT_MAPPING = {}
for m_char in METHYL_AA_ALPHABET:
    m_idx = EXTENDED_AA_TO_INDEX[m_char]
    n_char = m_char.upper()
    n_idx = EXTENDED_AA_TO_INDEX[n_char]
    INIT_MAPPING[m_idx] = n_idx

print(f">>> Config Loaded. Vocab Size: {len(EXTENDED_AA_ALPHABET)}")

# =============================================================================
# 2. Model (Using Safe Gather Functions)
# =============================================================================
def gather_nodes(nodes, neighbor_idx):
    # nodes: [B, L, H]
    # neighbor_idx: [B, L, K]
    # Output: [B, L, K, H]
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
        d_onehot = torch.nn.functional.one_hot(d, 66)
        return self.linear(d_onehot.float())

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
        h_EV = cat_neighbors_nodes(h_V, h_E, E_idx)
        h_V_expand = h_V.unsqueeze(-2).expand(-1,-1,h_EV.size(-2),-1); h_EV = torch.cat([h_V_expand, h_EV], -1)
        h_message = self.W3(self.act(self.W2(self.act(self.W1(h_EV)))))
        if mask_attend is not None: h_message = mask_attend.unsqueeze(-1) * h_message
        h_V = self.norm1(h_V + self.dropout1(torch.sum(h_message, -2) / 30.0)); h_V = self.norm2(h_V + self.dropout2(self.dense(h_V)))
        h_EV = cat_neighbors_nodes(h_V, h_E, E_idx); h_V_expand = h_V.unsqueeze(-2).expand(-1,-1,h_EV.size(-2),-1); h_EV = torch.cat([h_V_expand, h_EV], -1)
        h_message = self.W13(self.act(self.W12(self.act(self.W11(h_EV))))); h_E = self.norm3(h_E + self.dropout3(h_message))
        return h_V, h_E

class DecLayer(nn.Module):
    def __init__(self, hidden_dim, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_dim); self.norm2 = nn.LayerNorm(hidden_dim)
        self.W1 = nn.Linear(hidden_dim*3, hidden_dim); self.W2 = nn.Linear(hidden_dim, hidden_dim); self.W3 = nn.Linear(hidden_dim, hidden_dim)
        self.act = nn.GELU(); self.dense = PositionWiseFeedForward(hidden_dim, hidden_dim*4)
        self.dropout1 = nn.Dropout(dropout); self.dropout2 = nn.Dropout(dropout)
    def forward(self, h_V, h_E, mask_V=None, mask_attend=None):
        h_V_expand = h_V.unsqueeze(-2).expand(-1,-1,h_E.size(-2),-1); h_EV = torch.cat([h_V_expand, h_E], -1)
        h_message = self.W3(self.act(self.W2(self.act(self.W1(h_EV)))))
        if mask_attend is not None: h_message = mask_attend.unsqueeze(-1) * h_message
        h_V = self.norm1(h_V + self.dropout1(torch.sum(h_message, -2) / 30.0)); h_V = self.norm2(h_V + self.dropout2(self.dense(h_V)))
        return h_V

class ProteinMPNN_Extended(nn.Module):
    def __init__(self, hidden_dim=128, k_neighbors=48, num_out_classes=41):
        super().__init__()
        self.features = ProteinFeatures(128, 128, top_k=k_neighbors)
        self.W_e = nn.Linear(128, hidden_dim, bias=True)
        self.W_s = nn.Embedding(21, hidden_dim) # Fixed to 21 input
        self.encoder_layers = nn.ModuleList([EncLayer(hidden_dim) for _ in range(3)])
        self.decoder_layers = nn.ModuleList([DecLayer(hidden_dim) for _ in range(3)])
        self.W_out = nn.Linear(hidden_dim, num_out_classes, bias=True) # Extended output

    def forward(self, X, S_natural, mask, residue_idx, chain_encoding_all):
        E, E_idx = self.features(X, mask, residue_idx, chain_encoding_all)
        h_V = torch.zeros((E.shape[0], E.shape[1], E.shape[-1]), device=E.device)
        h_E = self.W_e(E)
        
        # Mask setup
        mask_attend = gather_nodes(mask.unsqueeze(-1), E_idx).squeeze(-1)
        mask_attend = mask.unsqueeze(-1) * mask_attend
        
        for layer in self.encoder_layers:
            h_V, h_E = layer(h_V, h_E, E_idx, mask, mask_attend)
            
        h_S = self.W_s(S_natural)
        
        # [FIX] Safe gathering using helper function
        h_S_neighbors = gather_nodes(h_S, E_idx)
        h_ES = torch.cat([h_S_neighbors, h_E], -1) # 256 dim
        
        # Encoder context
        h_EX_encoder = cat_neighbors_nodes(torch.zeros_like(h_S), h_E, E_idx)
        h_EXV_encoder = cat_neighbors_nodes(h_V, h_EX_encoder, E_idx)
        
        # Decoding order
        decoding_order = torch.argsort((mask + 0.0001) * (torch.abs(torch.randn(mask.shape, device=X.device))))
        mask_size = E_idx.shape[1]
        permutation_matrix_reverse = torch.nn.functional.one_hot(decoding_order, num_classes=mask_size).float()
        order_mask_backward = torch.einsum('ij, biq, bjp->bqp', (1 - torch.triu(torch.ones(mask_size, mask_size, device=X.device))), permutation_matrix_reverse, permutation_matrix_reverse)
        
        mask_attend = gather_nodes(order_mask_backward, E_idx)
        mask_1D = mask.view([mask.size(0), mask.size(1), 1, 1])
        mask_bw = mask_1D * mask_attend
        mask_fw = mask_1D * (1. - mask_attend)
        h_EXV_encoder_fw = mask_fw * h_EXV_encoder
        
        for layer in self.decoder_layers:
            # [FIX] Safe gathering for decoder context
            h_V_neighbors = gather_nodes(h_V, E_idx)
            h_ESV = torch.cat([h_V_neighbors, h_ES], -1)
            
            h_ESV = mask_bw * h_ESV + h_EXV_encoder_fw
            h_V = layer(h_V, h_ESV, mask)
            
        logits = self.W_out(h_V)
        return logits

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
    X = np.zeros([B, L_max, 4, 3]); 
    S_natural = np.zeros([B, L_max], dtype=int) 
    S_extended = np.zeros([B, L_max], dtype=int) 
    mask = np.zeros([B, L_max], dtype=float)
    residue_idx = -100*np.ones([B, L_max], dtype=int); chain_encoding_all = np.zeros([B, L_max], dtype=int)
    
    for i, b in enumerate(batch):
        seq = b.get('seq_chain_A', '')
        l = len(seq)
        for j, char in enumerate(seq):
            ext_idx = EXTENDED_AA_TO_INDEX.get(char, EXTENDED_AA_TO_INDEX['X'])
            S_extended[i, j] = ext_idx
            nat_idx = INPUT_MAPPING.get(ext_idx, 20)
            S_natural[i, j] = nat_idx
            
        if 'N_chain_A' in b:
            for ai, atom in enumerate(['N', 'CA', 'C', 'O']):
                coords = b[f'{atom}_chain_A']
                if len(coords) >= l: X[i, :l, ai, :] = coords[:l]
        mask[i, :l] = 1.0; residue_idx[i, :l] = np.arange(l); chain_encoding_all[i, :l] = 0

    mask = np.isfinite(X[:,:,1,0]).astype(float); X[np.isnan(X)] = 0.0
    return [
        torch.tensor(X, dtype=torch.float32, device=device),
        torch.tensor(S_natural, dtype=torch.long, device=device),
        torch.tensor(mask, dtype=torch.float32, device=device),
        torch.tensor(residue_idx, dtype=torch.long, device=device),
        torch.tensor(chain_encoding_all, dtype=torch.long, device=device)
    ], torch.tensor(S_extended, dtype=torch.long, device=device)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrained_weights", type=str, required=True)
    parser.add_argument("--nmethyl_data", type=str, required=True)
    parser.add_argument("--test_data", type=str, required=True)
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    vocab_size = len(EXTENDED_AA_ALPHABET)
    model = ProteinMPNN_Extended(hidden_dim=128, k_neighbors=48, num_out_classes=vocab_size).to(device)
    
    print(">>> Loading Weights & Performing Surgery...")
    ckpt = torch.load(args.pretrained_weights, map_location=device)
    raw_sd = ckpt['model_state_dict'] if 'model_state_dict' in ckpt else ckpt
    
    model_sd = model.state_dict()
    transfer_sd = {k: v for k, v in raw_sd.items() if k in model_sd and v.shape == model_sd[k].shape}
    msg = model.load_state_dict(transfer_sd, strict=False)
    print(f"✅ Backbone Loaded.")
    
    if 'W_out.weight' in raw_sd:
        w_out_pretrained = raw_sd['W_out.weight'] 
        w_bias_pretrained = raw_sd['W_out.bias']
        with torch.no_grad():
            for i in range(20):
                model.W_out.weight.data[i] = w_out_pretrained[i]
                model.W_out.bias.data[i] = w_bias_pretrained[i]
            model.W_out.weight.data[vocab_size-1] = w_out_pretrained[20]
            print(">>> Injecting Methyl Weights from Parents...")
            for methyl_idx, natural_idx in INIT_MAPPING.items():
                model.W_out.weight.data[methyl_idx] = w_out_pretrained[natural_idx]
                model.W_out.bias.data[methyl_idx] = w_bias_pretrained[natural_idx]
                model.W_out.weight.data[methyl_idx] += torch.randn_like(model.W_out.weight.data[methyl_idx]) * 0.01

    class_weights = torch.ones(vocab_size, device=device)
    for i in range(20, vocab_size-1): class_weights[i] = 10.0 
        
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    train_ds = JSONLDataset(args.nmethyl_data)
    
    weights = []
    for d in train_ds.data:
        has_m = any(c in METHYL_AA_ALPHABET for c in d.get('seq_chain_A', ''))
        weights.append(10.0 if has_m else 1.0)
    sampler = WeightedRandomSampler(weights, len(train_ds), replacement=True)
    loader = DataLoader(train_ds, batch_size=16, sampler=sampler, collate_fn=lambda x:x)
    
    print(">>> Training (Integrated Pipeline)...")
    for ep in range(50):
        model.train()
        total_loss = 0
        for batch in loader:
            inputs, targets = featurize_batch_safe(batch, device)
            logits = model(*inputs)
            loss = F.cross_entropy(logits.view(-1, vocab_size), targets.view(-1), weight=class_weights)
            optimizer.zero_grad(); loss.backward(); optimizer.step()
            total_loss += loss.item()
        if ep % 10 == 0: print(f"Ep {ep} Loss: {total_loss:.4f}")

    print("\n>>> FINAL EVALUATION...")
    model.eval()
    test_loader = DataLoader(JSONLDataset(args.test_data), batch_size=32, collate_fn=lambda x:x)
    all_p, all_t = [], []
    with torch.no_grad():
        for batch in test_loader:
            inputs, targets = featurize_batch_safe(batch, device)
            mask = inputs[2]
            logits = model(*inputs)
            preds = torch.argmax(F.softmax(logits, -1), -1)
            valid = mask.bool() & (targets != EXTENDED_AA_TO_INDEX['X'])
            all_p.extend(preds[valid].cpu().numpy())
            all_t.extend(targets[valid].cpu().numpy())
    t = np.array(all_t); p = np.array(all_p)
    print(f"🔥 FINAL ACCURACY: {np.mean(t==p)*100:.2f}%")
    
    nat_mask = t < 20
    if nat_mask.sum() > 0: print(f"Natural AA Acc: {np.mean(t[nat_mask]==p[nat_mask])*100:.2f}%")
    met_mask = (t >= 20) & (t < vocab_size-1)
    if met_mask.sum() > 0: print(f"Methyl AA Acc:  {np.mean(t[met_mask]==p[met_mask])*100:.2f}%")

if __name__ == "__main__":
    main()
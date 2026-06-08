import argparse
import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
try:
    from torch.amp import GradScaler, autocast
except ImportError:
    from torch.cuda.amp import GradScaler, autocast

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 导入刚才创建的纯净版
from model_utils_vanilla import ProteinMPNN 
try:
    from nmethyl.utils.nmethyl_config import (
        EXTENDED_AA_ALPHABET, NATURAL_AA_ALPHABET,
        NMETHYL_TO_NATURAL_MAPPING, EXTENDED_AA_TO_INDEX
    )
except ImportError:
    EXTENDED_AA_ALPHABET = "ACDEFGHIKLMNPQRSTVWYXacdefghiklmnpqrstvwy"
    NATURAL_AA_ALPHABET = "ACDEFGHIKLMNPQRSTVWY"
    NMETHYL_TO_NATURAL_MAPPING = {i+21: i for i in range(20)}
    EXTENDED_AA_TO_INDEX = {aa: i for i, aa in enumerate(EXTENDED_AA_ALPHABET)}

# --- 核心：包装器，负责动态手术 ---
class SurgeryWrapper(nn.Module):
    def __init__(self, original_model):
        super().__init__()
        self.backbone = original_model
        # 冻结 backbone
        for p in self.backbone.parameters():
            p.requires_grad = False
            
        # 1. Base Head (继承)
        self.W_out_base = nn.Linear(128, 20)
        with torch.no_grad():
            self.W_out_base.weight.copy_(original_model.W_out.weight[:20])
            self.W_out_base.bias.copy_(original_model.W_out.bias[:20])
        # 冻结 Base Head
        for p in self.W_out_base.parameters(): p.requires_grad = False
            
        # 2. Methyl Head (新建，只有这里需要训练)
        self.W_out_methyl = nn.Linear(128, 2)
        
        # 3. Embedding 扩充
        old_emb = original_model.W_s
        new_emb = nn.Embedding(len(EXTENDED_AA_ALPHABET), 128)
        with torch.no_grad():
            new_emb.weight[:21] = old_emb.weight[:21]
            for m, n in NMETHYL_TO_NATURAL_MAPPING.items():
                if (m + 20) < len(EXTENDED_AA_ALPHABET):
                    new_emb.weight[m+20] = old_emb.weight[n]
        self.backbone.W_s = new_emb 
        self.backbone.W_s.weight.requires_grad = True 
        
    def forward(self, X, S, mask, chain_M, residue_idx, chain_encoding_all):
        m = self.backbone
        E, E_idx = m.features(X, mask, residue_idx, chain_encoding_all)
        h_V = torch.zeros((E.shape[0], E.shape[1], E.shape[-1]), device=E.device)
        h_E = m.W_e(E)

        mask_attend = self.gather_nodes(mask.unsqueeze(-1),  E_idx).squeeze(-1)
        mask_attend = mask.unsqueeze(-1) * mask_attend
        for layer in m.encoder_layers:
            h_V, h_E = layer(h_V, h_E, E_idx, mask, mask_attend)

        h_S = m.W_s(S)
        h_ES = self.cat_neighbors_nodes(h_S, h_E, E_idx)
        h_EX_encoder = self.cat_neighbors_nodes(torch.zeros_like(h_S), h_E, E_idx)
        h_EXV_encoder = self.cat_neighbors_nodes(h_V, h_EX_encoder, E_idx)

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

        for layer in m.decoder_layers:
            h_ESV = self.cat_neighbors_nodes(h_V, h_ES, E_idx)
            h_ESV = mask_bw * h_ESV + h_EXV_encoder_fw
            h_V = layer(h_V, h_ESV, mask)
            
        logits_base = self.W_out_base(h_V)
        logits_methyl = self.W_out_methyl(h_V)
        return logits_base, logits_methyl

    def gather_nodes(self, nodes, neighbor_idx):
        neighbors_flat = neighbor_idx.view((neighbor_idx.shape[0], -1))
        neighbors_flat = neighbors_flat.unsqueeze(-1).expand(-1, -1, nodes.size(2))
        return torch.gather(nodes, 1, neighbors_flat).view(list(neighbor_idx.shape)[:3] + [-1])

    def cat_neighbors_nodes(self, h_nodes, h_neighbors, E_idx):
        h_nodes = self.gather_nodes(h_nodes, E_idx)
        return torch.cat([h_neighbors, h_nodes], -1)

# ... (Helper Functions) ...
def compute_hbond_mask(X, mask):
    B, L, _, _ = X.shape
    N = X[:, :, 0, :]; O = X[:, :, 3, :]
    dist = torch.norm(N.unsqueeze(2) - O.unsqueeze(1), p=2, dim=-1)
    diag = torch.eye(L, device=X.device).unsqueeze(0)
    is_hb = (dist < 3.5) & (dist > 0.5) & (diag < 0.5)
    return is_hb.any(dim=2).float() * mask

class JSONLDataset(Dataset):
    def __init__(self, jsonl_file):
        self.data = []
        with open(jsonl_file, 'r') as f:
            for line in f: self.data.append(json.loads(line))
    def __len__(self): return len(self.data)
    def __getitem__(self, idx): return self.data[idx]

def featurize_batch(batch, device):
    B = len(batch); L_max = max([len(b['seq']) for b in batch])
    X = np.full([B, L_max, 4, 3], np.nan); S = np.zeros([B, L_max], dtype=int)
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
    
    # [FIXED HERE] Explicit Cast to float32/long
    X = torch.from_numpy(X).to(dtype=torch.float32, device=device)
    S = torch.from_numpy(S).to(dtype=torch.long, device=device)
    mask = torch.from_numpy(mask).to(dtype=torch.float32, device=device)
    chain_M = torch.from_numpy(chain_M).to(dtype=torch.float32, device=device)
    residue_idx = torch.from_numpy(residue_idx).to(dtype=torch.long, device=device)
    chain_encoding_all = torch.from_numpy(chain_encoding_all).to(dtype=torch.long, device=device)
    
    return [X, S, mask, chain_M, residue_idx, chain_encoding_all]

def evaluate(model, loader, device):
    model.eval()
    print(">>> Final Evaluation (Hybrid + Rule)...")
    all_p, all_t = [], []
    with torch.no_grad():
        for batch in loader:
            inputs = featurize_batch(batch, device)
            X, S, mask = inputs[0], inputs[1], inputs[2]
            lb, lm = model(*inputs)
            
            pb = F.softmax(lb, -1); pred_b = torch.argmax(pb, -1)
            pm = F.softmax(lm, -1)[:,:,1]
            hb = compute_hbond_mask(X, mask)
            pm = pm * (1.0 - hb) 
            is_m = pm > 0.5
            
            final = pred_b.clone()
            for b in range(X.shape[0]):
                for l in range(X.shape[1]):
                    if mask[b,l] and is_m[b,l]:
                        base = pred_b[b,l].item()
                        for m_id, n_id in NMETHYL_TO_NATURAL_MAPPING.items():
                            if n_id == base: final[b,l] = m_id + 20; break
            
            valid = mask.bool() & (S != 20)
            all_p.extend(final[valid].cpu().numpy()); all_t.extend(S[valid].cpu().numpy())
    
    acc = np.mean(np.array(all_p) == np.array(all_t))
    print(f"\n🏆 FINAL ACCURACY: {acc*100:.2f}%")
    return acc

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrained_weights", type=str, required=True)
    parser.add_argument("--nmethyl_data", type=str, required=True)
    parser.add_argument("--test_data", type=str, required=True)
    parser.add_argument("--mode", type=str, default="train")
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. Load Original Model
    base_model = ProteinMPNN(node_features=128, edge_features=128, hidden_dim=128)
    ckpt = torch.load(args.pretrained_weights, map_location=device)
    base_model.load_state_dict(ckpt['model_state_dict']) 
    print("✅ Pretrained weights loaded successfully.")
    
    # 2. Wrap it
    model = SurgeryWrapper(base_model).to(device)
    
    # 3. Train
    train_ds = JSONLDataset(args.nmethyl_data)
    loader = DataLoader(train_ds, batch_size=8, shuffle=True, collate_fn=lambda x:x)
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3)
    scaler = GradScaler()
    
    print(">>> Training Methyl Head Only...")
    for ep in range(30):
        model.train()
        for batch in loader:
            inputs = featurize_batch(batch, device)
            S = inputs[1]
            with autocast(device_type='cuda', dtype=torch.float16):
                lb, lm = model(*inputs)
                target_m = (S >= 20).long()
                loss = F.cross_entropy(lm.view(-1,2), target_m.view(-1), weight=torch.tensor([1., 5.], device=device))
            scaler.scale(loss).backward()
            scaler.step(optimizer); scaler.update(); optimizer.zero_grad()
        if ep % 5 == 0: print(f"Ep {ep} done.")
        
    torch.save(model.state_dict(), "final_hybrid_model.pt")
    evaluate(model, DataLoader(JSONLDataset(args.test_data), batch_size=8, collate_fn=lambda x:x), device)

if __name__ == "__main__":
    main()
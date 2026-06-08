import argparse
import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
import json
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

# AMP
try:
    from torch.amp import GradScaler, autocast
except ImportError:
    from torch.cuda.amp import GradScaler, autocast

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# --- 配置 ---
try:
    from nmethyl.utils.nmethyl_config import (
        EXTENDED_AA_ALPHABET, NATURAL_AA_ALPHABET,
        NMETHYL_TO_NATURAL_MAPPING, EXTENDED_AA_TO_INDEX
    )
    print(f"[Config] Loaded. Vocab Size: {len(EXTENDED_AA_ALPHABET)}")
except ImportError:
    EXTENDED_AA_ALPHABET = "ACDEFGHIKLMNPQRSTVWYXacdefghiklmnpqrstvwy"
    NATURAL_AA_ALPHABET = "ACDEFGHIKLMNPQRSTVWY"
    NMETHYL_TO_NATURAL_MAPPING = {i+21: i for i in range(20)}
    EXTENDED_AA_TO_INDEX = {aa: i for i, aa in enumerate(EXTENDED_AA_ALPHABET)}
    print(f"[Config] Default. Vocab Size: {len(EXTENDED_AA_ALPHABET)}")

# =============================================================================
# 1. 物理规则引擎：氢键检测
# =============================================================================
def compute_hbond_mask(X, mask):
    B, L, _, _ = X.shape
    N_atoms = X[:, :, 0, :] 
    O_atoms = X[:, :, 3, :] 
    dist = torch.norm(N_atoms.unsqueeze(2) - O_atoms.unsqueeze(1), p=2, dim=-1)
    diag_mask = torch.eye(L, device=X.device).unsqueeze(0).expand(B, -1, -1)
    is_hbond = (dist < 3.5) & (dist > 0.5) & (diag_mask < 0.5)
    has_hbond = is_hbond.any(dim=2).float()
    return has_hbond * mask 

# =============================================================================
# 2. 模型定义
# =============================================================================
def gather_nodes(nodes, neighbor_idx):
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

class VanillaProteinFeatures(nn.Module):
    def __init__(self, edge_features, node_features, num_positional_embeddings=16, num_rbf=16, top_k=30):
        super().__init__()
        self.top_k = top_k
        self.num_rbf = num_rbf
        self.embeddings = PositionalEncodings(num_positional_embeddings)
        edge_in = num_positional_embeddings + num_rbf * 25
        self.edge_embedding = nn.Linear(edge_in, edge_features, bias=False)
        self.norm_edges = nn.LayerNorm(edge_features)

    def _rbf(self, D):
        device = D.device
        D_min, D_max, D_count = 2., 22., self.num_rbf
        D_mu = torch.linspace(D_min, D_max, D_count, device=device).view([1,1,1,-1])
        D_sigma = (D_max - D_min) / D_count
        return torch.exp(-((D.unsqueeze(-1) - D_mu) / D_sigma)**2)

    def forward(self, X, mask, residue_idx, chain_labels):
        b = X[:,:,1,:] - X[:,:,0,:]; c = X[:,:,2,:] - X[:,:,1,:]
        a = torch.cross(b, c, dim=-1)
        Cb = -0.5827*a + 0.5680*b - 0.5407*c + X[:,:,1,:]
        Ca, N, C, O = X[:,:,1,:], X[:,:,0,:], X[:,:,2,:], X[:,:,3,:]
        mask_2D = mask.unsqueeze(1) * mask.unsqueeze(2)
        dX = X[:,:,1,:].unsqueeze(1) - X[:,:,1,:].unsqueeze(2)
        D = mask_2D * torch.sqrt(torch.sum(dX**2, 3) + 1e-6)
        D_max, _ = torch.max(D, -1, keepdim=True)
        D_adjust = D + (1. - mask_2D) * D_max
        D_neighbors, E_idx = torch.topk(D_adjust, np.minimum(self.top_k, X.shape[1]), dim=-1, largest=False)
        RBF_list = []
        RBF_list.append(self._rbf(torch.gather(D, 2, E_idx)))
        for atom1 in [N, C, O, Cb, Ca]:
            for atom2 in [N, C, O, Cb, Ca]:
                if atom1 is Ca and atom2 is Ca: continue
                dist = torch.sqrt(torch.sum((atom1.unsqueeze(1) - atom2.unsqueeze(2))**2, -1) + 1e-6)
                RBF_list.append(self._rbf(torch.gather(dist, 2, E_idx)))
        RBF_all = torch.cat(tuple(RBF_list), dim=-1)
        offset = residue_idx.unsqueeze(1) - residue_idx.unsqueeze(2)
        offset = torch.gather(offset, 2, E_idx)
        d_chains = (chain_labels.unsqueeze(1) == chain_labels.unsqueeze(2)).long()
        E_chains = torch.gather(d_chains, 2, E_idx)
        E_positional = self.embeddings(offset.long(), E_chains)
        E = torch.cat((E_positional, RBF_all), -1)
        E = self.edge_embedding(E)
        return self.norm_edges(E), E_idx

class ProteinMPNN_Hybrid(nn.Module):
    def __init__(self, hidden_dim=128):
        super().__init__()
        self.features = VanillaProteinFeatures(128, 128, top_k=48)
        self.W_e = nn.Linear(128, hidden_dim, bias=True)
        self.W_s = nn.Embedding(len(EXTENDED_AA_ALPHABET), hidden_dim)
        
        class EncLayer(nn.Module):
            def __init__(self):
                super().__init__()
                self.norm1 = nn.LayerNorm(128); self.norm2 = nn.LayerNorm(128); self.norm3 = nn.LayerNorm(128)
                self.W1 = nn.Linear(384, 128); self.W2 = nn.Linear(128, 128); self.W3 = nn.Linear(128, 128)
                self.W11 = nn.Linear(384, 128); self.W12 = nn.Linear(128, 128); self.W13 = nn.Linear(128, 128)
                self.act = nn.GELU(); self.dense = nn.Sequential(nn.Linear(128, 512), nn.GELU(), nn.Linear(512, 128))
            def forward(self, h_V, h_E, E_idx, mask_V=None):
                h_EV = cat_neighbors_nodes(h_V, h_E, E_idx)
                h_V_ex = h_V.unsqueeze(-2).expand(-1,-1,h_EV.size(-2),-1)
                h_message = self.W3(self.act(self.W2(self.act(self.W1(torch.cat([h_V_ex, h_EV], -1))))))
                h_V = self.norm1(h_V + torch.sum(h_message, -2) / 30.0)
                h_V = self.norm2(h_V + self.dense(h_V)) * mask_V.unsqueeze(-1)
                h_EV = cat_neighbors_nodes(h_V, h_E, E_idx)
                h_V_ex = h_V.unsqueeze(-2).expand(-1,-1,h_EV.size(-2),-1)
                h_message = self.W13(self.act(self.W12(self.act(self.W11(torch.cat([h_V_ex, h_EV], -1))))))
                h_E = self.norm3(h_E + h_message)
                return h_V, h_E

        class DecLayer(nn.Module):
            def __init__(self):
                super().__init__()
                self.norm1 = nn.LayerNorm(128); self.norm2 = nn.LayerNorm(128)
                self.W1 = nn.Linear(512, 128); self.W2 = nn.Linear(128, 128); self.W3 = nn.Linear(128, 128)
                self.act = nn.GELU(); self.dense = nn.Sequential(nn.Linear(128, 512), nn.GELU(), nn.Linear(512, 128))
            def forward(self, h_V, h_E, mask_V=None):
                h_V_ex = h_V.unsqueeze(-2).expand(-1,-1,h_E.size(-2),-1)
                h_message = self.W3(self.act(self.W2(self.act(self.W1(torch.cat([h_V_ex, h_E], -1))))))
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
        
        for layer in self.encoder_layers:
            h_V, h_E = layer(h_V, h_E, E_idx, mask)

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
            h_V = layer(h_V, h_ESV, mask)
        
        logits_base = self.W_out_base(h_V)
        logits_methyl = self.W_out_methyl(h_V)
        return logits_base, logits_methyl

# =============================================================================
# 3. 训练与测试 (Logic Fix)
# =============================================================================
class JSONLDataset(Dataset):
    def __init__(self, jsonl_file):
        self.data = []
        with open(jsonl_file, 'r') as f:
            for line in f: self.data.append(json.loads(line))
    def __len__(self): return len(self.data)
    def __getitem__(self, idx): return self.data[idx]

def featurize_batch_standalone(batch, device):
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
            if f'N_chain_{c_id}' in b:
                coords_N = b[f'N_chain_{c_id}']; coords_CA = b[f'CA_chain_{c_id}']; coords_C = b[f'C_chain_{c_id}']; coords_O = b[f'O_chain_{c_id}']
            else:
                coords_N, coords_CA, coords_C, coords_O = [], [], [], []
            coords_N = np.array(coords_N); coords_CA = np.array(coords_CA); coords_C = np.array(coords_C); coords_O = np.array(coords_O)
            l = len(seq)
            def safe_assign(source, atom_idx):
                if len(source) > 0:
                    v_len = min(l, len(source))
                    X[i, l_p:l_p+v_len, atom_idx, :] = source[:v_len]
            safe_assign(coords_N, 0); safe_assign(coords_CA, 1); safe_assign(coords_C, 2); safe_assign(coords_O, 3)
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

def train_epoch(model, loader, optimizer, device, scaler, accum_steps):
    model.train()
    total_loss = 0; steps = 0
    optimizer.zero_grad()
    for i, batch in enumerate(loader):
        f = featurize_batch_standalone(batch, device)
        if f is None: continue
        X, S, mask, chain_M, residue_idx, chain_encoding_all = f
        
        with autocast(device_type='cuda', dtype=torch.float16):
            logits_base, logits_methyl = model(X, S, mask, chain_M, residue_idx, chain_encoding_all)
            
            targets_base = S.clone()
            offset = 20
            for m, n in NMETHYL_TO_NATURAL_MAPPING.items():
                targets_base[targets_base == (m + offset)] = n
            targets_base[targets_base >= 20] = -100
            
            targets_methyl = (S >= 20).long()
            
            loss_base = F.cross_entropy(logits_base.view(-1, 20), targets_base.view(-1), ignore_index=-100)
            loss_methyl = F.cross_entropy(logits_methyl.view(-1, 2), targets_methyl.view(-1))
            
            loss = loss_base * 0.5 + loss_methyl * 2.0
            loss = loss / accum_steps
        
        scaler.scale(loss).backward()
        if (i + 1) % accum_steps == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
        total_loss += loss.item() * accum_steps; steps += 1
    return total_loss / steps

def evaluate_with_rules(model, loader, device):
    model.eval()
    all_preds, all_targets = [], []
    print(">>> Inference with H-Bond Exclusion Rule...")
    
    with torch.no_grad():
        for batch in loader:
            f = featurize_batch_standalone(batch, device)
            if f is None: continue
            X, S, mask, chain_M, residue_idx, chain_encoding_all = f
            
            logits_base, logits_methyl = model(X, S, mask, chain_M, residue_idx, chain_encoding_all)
            probs_base = F.softmax(logits_base, dim=-1)
            pred_base = torch.argmax(probs_base, dim=-1) # 0-19
            probs_methyl = F.softmax(logits_methyl, dim=-1)[:, :, 1]
            
            hbond_mask = compute_hbond_mask(X, mask)
            probs_methyl = probs_methyl * (1.0 - hbond_mask) # 物理否决
            is_methyl = probs_methyl > 0.5
            
            final_pred = pred_base.clone()
            
            # 合并逻辑
            for b in range(X.shape[0]):
                for l in range(X.shape[1]):
                    if mask[b, l] == 1:
                        if is_methyl[b, l]:
                            base_id = pred_base[b, l].item()
                            # 逆向查找: base_id 是哪个 Methyl 的基底？
                            # NMETHYL_TO_NATURAL_MAPPING = {Me: Nat}
                            for m_id, n_id in NMETHYL_TO_NATURAL_MAPPING.items():
                                if n_id == base_id:
                                    final_pred[b, l] = m_id + 20
                                    break
            
            valid = mask.bool() & (S != EXTENDED_AA_TO_INDEX.get('X', -1))
            all_preds.extend(final_pred[valid].cpu().numpy())
            all_targets.extend(S[valid].cpu().numpy())

    # 统计
    t = np.array(all_targets); p = np.array(all_preds)
    acc = np.mean(t == p)
    print(f"\n========================================")
    print(f"🚀 FINAL HYBRID ACCURACY (Rule+Model): {acc:.4f} ({acc*100:.2f}%)")
    print(f"========================================")
    
    nat_mask = t < 20
    print(f"Base AA Acc: {np.mean(t[nat_mask] == p[nat_mask]):.4f}")
    met_mask = t >= 20
    if met_mask.sum() > 0:
        print(f"Methyl AA Acc: {np.mean(t[met_mask] == p[met_mask]):.4f}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrained_weights", type=str, required=True)
    parser.add_argument("--nmethyl_data", type=str, required=True)
    parser.add_argument("--test_data", type=str, required=True)
    parser.add_argument("--mode", type=str, default="train")
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ProteinMPNN_Hybrid(128).to(device)
    
    if args.mode == "train":
        print("Loading Pretrained (Base Weights Transfer)...")
        ckpt = torch.load(args.pretrained_weights, map_location=device)
        sd = ckpt['model_state_dict']
        
        # [KEY FIX] 手动过滤，防止尺寸不匹配
        # 1. 先过滤掉不匹配的层
        model_dict = model.state_dict()
        filtered_sd = {k: v for k, v in sd.items() if k in model_dict and v.shape == model_dict[k].shape}
        model.load_state_dict(filtered_sd, strict=False)
        
        # 2. 手动植入 W_s (Input Embeddings)
        # Pretrained: 21, Ours: 40
        print("💉 Injecting Input Embeddings (Safe Copy)...")
        with torch.no_grad():
            limit = min(21, model.W_s.num_embeddings)
            model.W_s.weight.data[:limit] = sd['W_s.weight'][:limit]

        # 3. 手动植入 W_out (Output Projection)
        if 'W_out.weight' in sd:
            print("💉 Injecting Pretrained Output Weights -> Base Head")
            with torch.no_grad():
                model.W_out_base.weight.data.copy_(sd['W_out.weight'][:20])
                model.W_out_base.bias.data.copy_(sd['W_out.bias'][:20])
        
        # 冻结 Base Head
        for p in model.W_out_base.parameters(): p.requires_grad = False
        
        train_ds = JSONLDataset(args.nmethyl_data)
        loader = DataLoader(train_ds, batch_size=8, shuffle=True, collate_fn=lambda x: x)
        optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4)
        scaler = GradScaler()
        
        print(">>> Training Methyl Head...")
        for epoch in range(50):
            loss = train_epoch(model, loader, optimizer, device, scaler, 1)
            if epoch % 10 == 0: print(f"Ep {epoch}: Loss {loss:.4f}")
        
        torch.save(model.state_dict(), "final_hybrid_model.pt")
        
        test_ds = JSONLDataset(args.test_data)
        test_loader = DataLoader(test_ds, batch_size=8, collate_fn=lambda x: x)
        evaluate_with_rules(model, test_loader, device)

if __name__ == "__main__":
    main()
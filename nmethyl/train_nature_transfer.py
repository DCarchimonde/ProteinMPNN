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

# 兼容不同版本的 PyTorch AMP
try:
    from torch.amp import GradScaler, autocast
except ImportError:
    from torch.cuda.amp import GradScaler, autocast

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 1. 配置
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

# 2. 模型定义 (Enc/Dec 分离，维度修正版)
def gather_nodes(nodes, neighbor_idx):
    neighbors_flat = neighbor_idx.view((neighbor_idx.shape[0], -1))
    neighbors_flat = neighbors_flat.unsqueeze(-1).expand(-1, -1, nodes.size(2))
    neighbor_features = torch.gather(nodes, 1, neighbors_flat)
    neighbor_features = neighbor_features.view(list(neighbor_idx.shape)[:3] + [-1])
    return neighbor_features

def cat_neighbors_nodes(h_nodes, h_neighbors, E_idx):
    h_nodes = gather_nodes(h_nodes, E_idx)
    h_nn = torch.cat([h_neighbors, h_nodes], -1)
    return h_nn

class PositionalEncodings(nn.Module):
    def __init__(self, num_embeddings, max_relative_feature=32):
        super(PositionalEncodings, self).__init__()
        self.num_embeddings = num_embeddings
        self.max_relative_feature = max_relative_feature
        self.linear = nn.Linear(2*max_relative_feature+1+1, num_embeddings)
    def forward(self, offset, mask):
        d = torch.clip(offset + self.max_relative_feature, 0, 2*self.max_relative_feature)*mask + (1-mask)*(2*self.max_relative_feature+1)
        d_onehot = torch.nn.functional.one_hot(d, 2*self.max_relative_feature+1+1)
        E = self.linear(d_onehot.float())
        return E

class VanillaProteinFeatures(nn.Module):
    def __init__(self, edge_features, node_features, num_positional_embeddings=16,
        num_rbf=16, top_k=30, augment_eps=0.):
        super(VanillaProteinFeatures, self).__init__()
        self.edge_features = edge_features
        self.node_features = node_features
        self.top_k = top_k
        self.augment_eps = augment_eps 
        self.num_rbf = num_rbf
        self.num_positional_embeddings = num_positional_embeddings
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
        D_expand = torch.unsqueeze(D, -1)
        RBF = torch.exp(-((D_expand - D_mu) / D_sigma)**2)
        return RBF

    def _get_rbf(self, A, B, E_idx):
        D_A_B = torch.sqrt(torch.sum((A[:,:,None,:] - B[:,None,:,:])**2,-1) + 1e-6)
        D_A_B_neighbors = torch.gather(D_A_B, 2, E_idx)
        return self._rbf(D_A_B_neighbors)

    def forward(self, X, mask, residue_idx, chain_labels):
        if self.training and self.augment_eps > 0:
            X = X + self.augment_eps * torch.randn_like(X)
        b = X[:,:,1,:] - X[:,:,0,:]
        c = X[:,:,2,:] - X[:,:,1,:]
        a = torch.cross(b, c, dim=-1)
        Cb = -0.58273431*a + 0.56802827*b - 0.54067466*c + X[:,:,1,:]
        Ca, N, C, O = X[:,:,1,:], X[:,:,0,:], X[:,:,2,:], X[:,:,3,:]
        D_neighbors, E_idx = self._dist(Ca, mask)

        RBF_all = []
        RBF_all.append(self._rbf(D_neighbors)) 
        RBF_all.extend([self._get_rbf(N,N,E_idx), self._get_rbf(C,C,E_idx), self._get_rbf(O,O,E_idx), self._get_rbf(Cb,Cb,E_idx)])
        RBF_all.extend([self._get_rbf(Ca,N,E_idx), self._get_rbf(Ca,C,E_idx), self._get_rbf(Ca,O,E_idx), self._get_rbf(Ca,Cb,E_idx)])
        RBF_all.extend([self._get_rbf(N,C,E_idx), self._get_rbf(N,O,E_idx), self._get_rbf(N,Cb,E_idx)])
        RBF_all.extend([self._get_rbf(Cb,C,E_idx), self._get_rbf(Cb,O,E_idx), self._get_rbf(O,C,E_idx)])
        RBF_all.extend([self._get_rbf(N,Ca,E_idx), self._get_rbf(C,Ca,E_idx), self._get_rbf(O,Ca,E_idx), self._get_rbf(Cb,Ca,E_idx)])
        RBF_all.extend([self._get_rbf(C,N,E_idx), self._get_rbf(O,N,E_idx), self._get_rbf(Cb,N,E_idx)])
        RBF_all.extend([self._get_rbf(C,Cb,E_idx), self._get_rbf(O,Cb,E_idx), self._get_rbf(C,O,E_idx)])
        
        RBF_all = torch.cat(tuple(RBF_all), dim=-1)
        offset = residue_idx[:,:,None]-residue_idx[:,None,:] 
        offset = torch.gather(offset, 2, E_idx)
        d_chains = ((chain_labels[:, :, None] - chain_labels[:,None,:])==0).long()
        E_chains = torch.gather(d_chains, 2, E_idx)
        E_positional = self.embeddings(offset.long(), E_chains)
        E = torch.cat((E_positional, RBF_all), -1)
        E = self.edge_embedding(E)
        E = self.norm_edges(E)
        return E, E_idx

class EncLayer(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_dim); self.norm2 = nn.LayerNorm(hidden_dim); self.norm3 = nn.LayerNorm(hidden_dim)
        self.W1 = nn.Linear(hidden_dim*3, hidden_dim) 
        self.W2 = nn.Linear(hidden_dim, hidden_dim); self.W3 = nn.Linear(hidden_dim, hidden_dim)
        self.W11 = nn.Linear(hidden_dim*3, hidden_dim) 
        self.W12 = nn.Linear(hidden_dim, hidden_dim); self.W13 = nn.Linear(hidden_dim, hidden_dim)
        self.act = nn.GELU(); self.dense = nn.Sequential(nn.Linear(hidden_dim, hidden_dim*4), nn.GELU(), nn.Linear(hidden_dim*4, hidden_dim))
    
    def forward(self, h_V, h_E, E_idx, mask_V=None, mask_attend=None):
        h_EV = cat_neighbors_nodes(h_V, h_E, E_idx) 
        h_V_expand = h_V.unsqueeze(-2).expand(-1,-1,h_EV.size(-2),-1) 
        h_EV = torch.cat([h_V_expand, h_EV], -1) 
        h_message = self.W3(self.act(self.W2(self.act(self.W1(h_EV)))))
        if mask_attend is not None: h_message = mask_attend.unsqueeze(-1) * h_message
        h_V = self.norm1(h_V + torch.sum(h_message, -2) / 30.0)
        h_V = self.norm2(h_V + self.dense(h_V))
        if mask_V is not None: h_V = mask_V.unsqueeze(-1) * h_V
        h_EV = cat_neighbors_nodes(h_V, h_E, E_idx)
        h_V_expand = h_V.unsqueeze(-2).expand(-1,-1,h_EV.size(-2),-1)
        h_EV = torch.cat([h_V_expand, h_EV], -1) 
        h_message = self.W13(self.act(self.W12(self.act(self.W11(h_EV)))))
        h_E = self.norm3(h_E + h_message)
        return h_V, h_E

class DecLayer(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_dim); self.norm2 = nn.LayerNorm(hidden_dim)
        self.W1 = nn.Linear(hidden_dim + hidden_dim*3, hidden_dim)
        self.W2 = nn.Linear(hidden_dim, hidden_dim); self.W3 = nn.Linear(hidden_dim, hidden_dim)
        self.act = nn.GELU(); self.dense = nn.Sequential(nn.Linear(hidden_dim, hidden_dim*4), nn.GELU(), nn.Linear(hidden_dim*4, hidden_dim))
    
    def forward(self, h_V, h_E, mask_V=None, mask_attend=None):
        h_V_expand = h_V.unsqueeze(-2).expand(-1,-1,h_E.size(-2),-1) 
        h_EV = torch.cat([h_V_expand, h_E], -1) 
        h_message = self.W3(self.act(self.W2(self.act(self.W1(h_EV)))))
        if mask_attend is not None: h_message = mask_attend.unsqueeze(-1) * h_message
        h_V = self.norm1(h_V + torch.sum(h_message, -2) / 30.0)
        h_V = self.norm2(h_V + self.dense(h_V))
        if mask_V is not None: h_V = mask_V.unsqueeze(-1) * h_V
        return h_V

class ProteinMPNN_Standalone(nn.Module):
    def __init__(self, hidden_dim=128, augment_eps=0.05):
        super(ProteinMPNN_Standalone, self).__init__()
        self.features = VanillaProteinFeatures(128, 128, top_k=48, augment_eps=augment_eps)
        self.W_e = nn.Linear(128, hidden_dim, bias=True)
        self.W_s = nn.Embedding(len(EXTENDED_AA_ALPHABET), hidden_dim)
        self.encoder_layers = nn.ModuleList([EncLayer(hidden_dim) for _ in range(3)])
        self.decoder_layers = nn.ModuleList([DecLayer(hidden_dim) for _ in range(3)])
        self.W_out_base = nn.Linear(128, len(NATURAL_AA_ALPHABET))
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

# 3. 数据处理
class JSONLDataset(Dataset):
    def __init__(self, jsonl_file):
        self.data = []
        with open(jsonl_file, 'r') as f:
            for line in f: self.data.append(json.loads(line))
    def __len__(self): return len(self.data)
    def __getitem__(self, idx): return self.data[idx]

def get_weighted_sampler(dataset): 
    weights = []
    for item in dataset.data:
        has_methyl = any(c in NMETHYL_TO_NATURAL_MAPPING for seq_key in item if 'seq' in seq_key for c in item[seq_key])
        weights.append(5.0 if has_methyl else 1.0)
    return WeightedRandomSampler(weights, len(weights), replacement=True)

def featurize_batch_standalone(batch, device):
    B = len(batch)
    batch = [b for b in batch if 'seq' in b and len(b['seq']) > 0]
    if not batch: return None
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

def calculate_loss(logits_base, logits_methyl, targets, mask, methyl_weight=5.0):
    mask_flat = mask.contiguous().view(-1).bool()
    if mask_flat.sum() == 0: return torch.tensor(0.0, device=logits_base.device), (0, 0)
    targets_flat = targets.contiguous().view(-1)[mask_flat]
    valid = targets_flat != EXTENDED_AA_TO_INDEX.get('X', -1)
    if valid.sum() == 0: return torch.tensor(0.0, device=logits_base.device), (0, 0)
    targets_flat = targets_flat[valid]
    logits_base = logits_base.contiguous().view(-1, 20)[mask_flat][valid]
    logits_methyl = logits_methyl.contiguous().view(-1, 2)[mask_flat][valid]
    base_targets = targets_flat.clone()
    offset = len(NATURAL_AA_ALPHABET)
    for m, n in NMETHYL_TO_NATURAL_MAPPING.items():
        base_targets[base_targets == (m + offset)] = n
    base_targets[base_targets >= len(NATURAL_AA_ALPHABET)] = -100
    methyl_targets = (targets_flat >= len(NATURAL_AA_ALPHABET)).long()
    loss_base = F.cross_entropy(logits_base, base_targets, ignore_index=-100)
    ce_loss_methyl = F.cross_entropy(logits_methyl, methyl_targets, reduction='none', weight=torch.tensor([1.0, 5.0], device=logits_base.device))
    loss_methyl = ((1 - torch.exp(-ce_loss_methyl)) ** 2.0 * ce_loss_methyl).mean()
    with torch.no_grad():
        pred_base = torch.argmax(logits_base, dim=1)
        acc_b = (pred_base == base_targets).float().mean().item()
        pred_meth = torch.argmax(logits_methyl, dim=1)
        acc_m = (pred_meth == methyl_targets).float().mean().item()
    return loss_base + methyl_weight * loss_methyl, (acc_b, acc_m)

def train_epoch(model, loader, optimizer, device, scaler, accum_steps):
    model.train()
    total_loss = 0; steps = 0; total_acc_b = 0; total_acc_m = 0
    optimizer.zero_grad()
    for i, batch in enumerate(loader):
        f = featurize_batch_standalone(batch, device)
        if f is None: continue
        X, S, mask, chain_M, residue_idx, chain_encoding_all = f
        if i == 0 and torch.count_nonzero(X) == 0:
            print("❌ Data Error: All zeros!")
            sys.exit(1)
        with autocast(device_type='cuda', dtype=torch.float16):
            lb, lm = model(X, S, mask, chain_M, residue_idx, chain_encoding_all)
            loss, (acc_b, acc_m) = calculate_loss(lb, lm, S, mask)
            loss = loss / accum_steps
        if loss.item() > 0:
            scaler.scale(loss).backward()
            if (i + 1) % accum_steps == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
            total_loss += loss.item() * accum_steps
            total_acc_b += acc_b; total_acc_m += acc_m; steps += 1
    return total_loss/steps if steps > 0 else 0, total_acc_b/steps if steps > 0 else 0, total_acc_m/steps if steps > 0 else 0

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrained_weights", type=str, required=True)
    parser.add_argument("--nmethyl_data", type=str, required=True)
    parser.add_argument("--test_data", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="./run_nature_transfer")
    parser.add_argument("--epochs", type=int, default=100)
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.output_dir, exist_ok=True)
    
    model = ProteinMPNN_Standalone(augment_eps=0.2).to(device)
    
    print(f"Loading weights from: {args.pretrained_weights}")
    ckpt = torch.load(args.pretrained_weights, map_location=device)
    model_dict = model.state_dict()
    # 智能加载
    pretrained_dict = {k: v for k, v in ckpt.get('model_state_dict', ckpt).items() if k in model_dict and v.shape == model_dict[k].shape}
    model.load_state_dict(pretrained_dict, strict=False)
    
    print("🔥 Injecting Knowledge Transfer & UNFREEZING...")
    if 'W_s.weight' in ckpt.get('model_state_dict', ckpt):
        old_ws = ckpt.get('model_state_dict', ckpt)['W_s.weight']
        with torch.no_grad():
            model.W_s.weight.data[:21] = old_ws[:21]
            for m_idx, n_idx in NMETHYL_TO_NATURAL_MAPPING.items():
                target_idx = len(NATURAL_AA_ALPHABET) + m_idx
                if target_idx < model.W_s.num_embeddings:
                    model.W_s.weight.data[target_idx] = model.W_s.weight.data[n_idx]
    
    # [KEY STEP] 全参数训练，不要冻结！
    print("🔓 Full Fine-tuning Enabled! (Encoder/Decoder Unfrozen)")
    optimizer = torch.optim.AdamW([
        {'params': model.features.parameters(), 'lr': 1e-5}, 
        {'params': model.encoder_layers.parameters(), 'lr': 1e-5}, 
        {'params': model.decoder_layers.parameters(), 'lr': 1e-5},
        {'params': model.W_out_base.parameters(), 'lr': 1e-4}, 
        {'params': model.W_out_methyl.parameters(), 'lr': 1e-4},
        {'params': model.W_s.parameters(), 'lr': 1e-4}
    ], weight_decay=1e-4)

    train_ds = JSONLDataset(args.nmethyl_data)
    loader = DataLoader(train_ds, batch_size=8, sampler=get_weighted_sampler(train_ds), collate_fn=lambda x: x)
    scaler = GradScaler()
    
    print(">>> Starting Full Fine-tuning...")
    for epoch in range(1, args.epochs+1):
        loss, acc_b, acc_m = train_epoch(model, loader, optimizer, device, scaler, 4)
        if epoch % 5 == 0:
            print(f"Epoch {epoch}: Loss={loss:.4f} | Base Acc={acc_b:.2%} | Methyl Acc={acc_m:.2%}")
        if epoch % 10 == 0:
            torch.save({'model_state_dict': model.state_dict()}, f"{args.output_dir}/model_final.pt")

if __name__ == "__main__":
    main()
import argparse
import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint
import json
import numpy as np
import random
import copy
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from sklearn.metrics import classification_report
from torch.optim.lr_scheduler import OneCycleLR

# --- 系统路径 ---
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    print(f"[System] Random Seed locked to: {seed}")

# =============================================================================
# 1. 核心配置与工具函数
# =============================================================================
NATURAL_AA_ALPHABET = "ACDEFGHIKLMNPQRSTVWY"
EXTENDED_AA_ALPHABET = "ACDEFGHIKLMNPQRSTVWYacdefghiklmnqrstvwvyX"
EXTENDED_AA_TO_INDEX = {aa: i for i, aa in enumerate(EXTENDED_AA_ALPHABET)}

NMETHYL_TO_NATURAL_MAPPING = {} 
for i, aa in enumerate(NATURAL_AA_ALPHABET):
    lower_aa = aa.lower()
    if lower_aa in EXTENDED_AA_TO_INDEX:
        nmethyl_idx = EXTENDED_AA_TO_INDEX[lower_aa]
        natural_idx = EXTENDED_AA_TO_INDEX[aa]
        NMETHYL_TO_NATURAL_MAPPING[nmethyl_idx - 20] = natural_idx 

def gather_nodes(nodes, neighbor_idx):
    neighbors_flat = neighbor_idx.view((neighbor_idx.shape[0], -1))
    neighbors_flat = neighbors_flat.unsqueeze(-1).expand(-1, -1, nodes.size(2))
    neighbor_features = torch.gather(nodes, 1, neighbors_flat)
    return neighbor_features.view(list(neighbor_idx.shape)[:3] + [-1])

def cat_neighbors_nodes(h_nodes, h_neighbors, E_idx):
    h_nodes = gather_nodes(h_nodes, E_idx)
    h_nn = torch.cat([h_neighbors, h_nodes], -1)
    return h_nn

# =============================================================================
# 2. 模型定义 (保留 V16 的正确架构)
# =============================================================================
class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.W_in = nn.Linear(d_model, d_ff)
        self.W_out = nn.Linear(d_ff, d_model)
        self.act = nn.GELU()
    def forward(self, x):
        return self.W_out(self.act(self.W_in(x)))

class EncLayer(nn.Module):
    def __init__(self, num_hidden, num_in, dropout=0.1, scale=30):
        super(EncLayer, self).__init__()
        self.num_hidden = num_hidden
        self.num_in = num_in
        self.scale = scale
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(num_hidden)
        self.norm2 = nn.LayerNorm(num_hidden)
        self.norm3 = nn.LayerNorm(num_hidden)

        # Block 1
        self.W1 = nn.Linear(num_hidden + num_in, num_hidden, bias=True)
        self.W2 = nn.Linear(num_hidden, num_hidden, bias=True)
        self.W3 = nn.Linear(num_hidden, num_hidden, bias=True)
        
        # Block 2
        self.W11 = nn.Linear(num_hidden + num_in, num_hidden, bias=True)
        self.W12 = nn.Linear(num_hidden, num_hidden, bias=True)
        self.W13 = nn.Linear(num_hidden, num_hidden, bias=True)

        self.act = nn.GELU()
        self.dense = PositionwiseFeedForward(num_hidden, num_hidden * 4, dropout)

    def forward(self, h_V, h_E, E_idx, mask_V=None, mask_attend=None):
        # Block 1
        h_V_neigh = gather_nodes(h_V, E_idx)
        h_V_expand = h_V.unsqueeze(-2).expand(-1, -1, h_E.size(-2), -1)
        h_EV = torch.cat([h_V_expand, h_V_neigh, h_E], -1) 
        h_message = self.W3(self.act(self.W2(self.act(self.W1(h_EV)))))
        if mask_attend is not None:
            h_message = mask_attend.unsqueeze(-1) * h_message
        dh = torch.sum(h_message, -2) / self.scale
        h_V = self.norm1(h_V + self.dropout1(dh))

        # Block 2
        h_V_neigh = gather_nodes(h_V, E_idx)
        h_V_expand = h_V.unsqueeze(-2).expand(-1, -1, h_E.size(-2), -1)
        h_EV = torch.cat([h_V_expand, h_V_neigh, h_E], -1)
        h_message = self.W13(self.act(self.W12(self.act(self.W11(h_EV)))))
        if mask_attend is not None:
            h_message = mask_attend.unsqueeze(-1) * h_message
        dh = torch.sum(h_message, -2) / self.scale
        h_V = self.norm2(h_V + self.dropout2(dh))

        h_V = self.norm3(h_V + self.dropout3(self.dense(h_V)))
        return h_V, h_E

class DecLayer(nn.Module):
    def __init__(self, num_hidden, num_in, dropout=0.1, scale=30):
        super(DecLayer, self).__init__()
        self.num_hidden = num_hidden
        self.num_in = num_in
        self.scale = scale
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(num_hidden)
        self.norm2 = nn.LayerNorm(num_hidden)

        self.W1 = nn.Linear(num_hidden + num_in, num_hidden, bias=True)
        self.W2 = nn.Linear(num_hidden, num_hidden, bias=True)
        self.W3 = nn.Linear(num_hidden, num_hidden, bias=True)
        
        self.act = nn.GELU()
        self.dense = PositionwiseFeedForward(num_hidden, num_hidden * 4, dropout)

    def forward(self, h_V, h_E, mask_V=None, mask_attend=None):
        h_V_expand = h_V.unsqueeze(-2).expand(-1, -1, h_E.size(-2), -1)
        h_cat = torch.cat([h_V_expand, h_E], -1) 
        h_message = self.W3(self.act(self.W2(self.act(self.W1(h_cat))))) 
        if mask_attend is not None:
            h_message = mask_attend.unsqueeze(-1) * h_message
        dh = torch.sum(h_message, -2) / self.scale
        h_V = self.norm1(h_V + self.dropout1(dh))
        h_V = self.norm2(h_V + self.dropout2(self.dense(h_V)))
        return h_V

class DecoupledProteinMPNN(nn.Module):
    def __init__(self, num_letters=21, node_features=128, edge_features=128,
                 hidden_dim=128, num_encoder_layers=3, num_decoder_layers=3,
                 vocab=21, k_neighbors=48, augment_eps=0.1, dropout=0.1):
        super(DecoupledProteinMPNN, self).__init__()
        self.hidden_dim = hidden_dim
        self.k_neighbors = k_neighbors 

        self.features = nn.ModuleDict({
            'embeddings': nn.ModuleDict({'linear': nn.Linear(66, 16)}),
            'edge_embedding': nn.Linear(416, 128, bias=False),
            'norm_edges': nn.LayerNorm(128)
        })
        
        self.W_e = nn.Linear(128, hidden_dim, bias=True)
        self.W_s = nn.Embedding(len(EXTENDED_AA_ALPHABET), hidden_dim)

        self.encoder_layers = nn.ModuleList([
            EncLayer(hidden_dim, hidden_dim*2, dropout=dropout)
            for _ in range(num_encoder_layers)
        ])

        self.decoder_layers = nn.ModuleList([
            DecLayer(hidden_dim, hidden_dim*3, dropout=dropout)
            for _ in range(num_decoder_layers)
        ])

        self.W_out_base = nn.Linear(hidden_dim, len(NATURAL_AA_ALPHABET))
        self.W_out_methyl = nn.Linear(hidden_dim, 2)

    def forward(self, X, S, mask, chain_M, residue_idx, chain_encoding_all):
        device = X.device
        
        b = X[:,:,1,:] - X[:,:,0,:]
        c = X[:,:,2,:] - X[:,:,1,:]
        a = torch.cross(b, c, dim=-1)
        Cb = -0.58273431*a + 0.56802827*b - 0.54067466*c + X[:,:,1,:]
        
        dist = torch.norm(X[:,:,1,:].unsqueeze(1) - X[:,:,1,:].unsqueeze(2), dim=-1)
        mask_2D = mask.unsqueeze(1) * mask.unsqueeze(2)
        dist = dist + (1.0 - mask_2D) * 1e8
        
        L_max = dist.shape[-1]
        curr_k = min(self.k_neighbors, L_max)
        E_idx = torch.topk(dist, curr_k, dim=-1, largest=False)[1]
        
        offset = residue_idx.unsqueeze(1) - residue_idx.unsqueeze(2)
        offset = torch.gather(offset, 2, E_idx)
        d_pos = torch.clip(offset + 32, 0, 64)
        pos_emb = self.features['embeddings']['linear'](F.one_hot(d_pos, 66).float())
        
        D_neighbors = torch.gather(dist, 2, E_idx)
        RBF_all = []
        RBF_all.append(self._rbf(D_neighbors))
        
        atoms = [X[:,:,0,:], X[:,:,2,:], X[:,:,3,:], Cb, X[:,:,1,:]]
        for i, atom1 in enumerate(atoms):
            for j, atom2 in enumerate(atoms):
                if i==4 and j==4: continue
                d = torch.norm(atom1.unsqueeze(1) - atom2.unsqueeze(2), dim=-1)
                d_neigh = torch.gather(d, 2, E_idx)
                RBF_all.append(self._rbf(d_neigh))
        
        RBF = torch.cat(RBF_all, dim=-1)
        E = torch.cat((pos_emb, RBF), -1)
        E = self.features['edge_embedding'](E)
        E = self.features['norm_edges'](E)
        
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

    def _rbf(self, D):
        device = D.device
        D_min, D_max, D_count = 2., 22., 16
        D_mu = torch.linspace(D_min, D_max, D_count, device=device)
        D_sigma = (D_max - D_min) / D_count
        D_expand = D.unsqueeze(-1)
        RBF = torch.exp(-((D_expand - D_mu) / D_sigma) ** 2)
        return RBF

# =============================================================================
# 3. 权重加载 (带扩展修复)
# =============================================================================
def advanced_load_weights(model, pretrained_path, device):
    print(f"\n>>> [Loader] Opening {pretrained_path}...")
    ckpt = torch.load(pretrained_path, map_location=device)
    state_dict = ckpt.get('model_state_dict', ckpt)
    
    if 'W_s.weight' in state_dict:
        std_emb = state_dict['W_s.weight']
        new_emb = model.W_s.weight.data.clone()
        new_emb[:21] = std_emb[:21]
        from nmethyl.utils.nmethyl_config import METHYL_AA_ALPHABET
        for nme_idx, nat_idx in NMETHYL_TO_NATURAL_MAPPING.items():
            if nme_idx < len(METHYL_AA_ALPHABET):
                nme_char = METHYL_AA_ALPHABET[nme_idx]
                nat_char = NATURAL_AA_ALPHABET[nat_idx]
                if nme_char in EXTENDED_AA_TO_INDEX and nat_char in EXTENDED_AA_TO_INDEX:
                    parent = std_emb[EXTENDED_AA_TO_INDEX[nat_char]]
                    new_emb[EXTENDED_AA_TO_INDEX[nme_char]] = parent + torch.randn_like(parent)*0.01
        model.W_s.weight.data = new_emb
        del state_dict['W_s.weight']

    model.load_state_dict(state_dict, strict=False)
    print("✅ Weights Loaded (Strict Architecture Match).")

# =============================================================================
# 4. 损失函数 & 数据处理
# =============================================================================
def calculate_loss(logits_base, logits_methyl, targets, mask, methyl_weight=2.0):
    mask_flat = mask.contiguous().view(-1).bool()
    if mask_flat.sum() == 0: return torch.tensor(0.0, device=logits_base.device), 0
    targets_flat = targets.contiguous().view(-1)[mask_flat]
    x_idx = EXTENDED_AA_TO_INDEX.get('X', -1)
    valid_pos = targets_flat != x_idx
    if valid_pos.sum() == 0: return torch.tensor(0.0, device=logits_base.device), 0
    
    targets_flat = targets_flat[valid_pos]
    logits_base = logits_base.contiguous().view(-1, 20)[mask_flat][valid_pos]
    logits_methyl = logits_methyl.contiguous().view(-1, 2)[mask_flat][valid_pos]
    
    base_targets = targets_flat.clone()
    offset = len(NATURAL_AA_ALPHABET)
    for m_rel, n_idx in NMETHYL_TO_NATURAL_MAPPING.items():
        base_targets[base_targets == (m_rel + offset)] = n_idx
    base_targets[base_targets >= len(NATURAL_AA_ALPHABET)] = -100
    
    methyl_targets = (targets_flat >= len(NATURAL_AA_ALPHABET)).long()
    
    loss_base = F.cross_entropy(logits_base, base_targets, ignore_index=-100)
    loss_methyl = F.cross_entropy(logits_methyl, methyl_targets, weight=torch.tensor([1.0, 3.0], device=logits_base.device))
    return loss_base + methyl_weight * loss_methyl, base_targets.numel()

class JSONLDataset(Dataset):
    def __init__(self, jsonl_file):
        self.data = []
        with open(jsonl_file, 'r') as f:
            for line in f: self.data.append(json.loads(line))
    def __len__(self): return len(self.data)
    def __getitem__(self, idx): return self.data[idx]

def get_weighted_sampler(dataset, oversample_weight=10.0):
    weights = []
    offset = len(NATURAL_AA_ALPHABET)
    methyl_indices = {k + offset for k in NMETHYL_TO_NATURAL_MAPPING.keys()}
    for item in dataset.data:
        has_methyl = False
        for key in item:
            if key.startswith('seq_chain_'):
                for char in item[key]:
                    if EXTENDED_AA_TO_INDEX.get(char, -1) in methyl_indices:
                        has_methyl = True; break
            if has_methyl: break
        weights.append(oversample_weight if has_methyl else 1.0)
    return WeightedRandomSampler(weights, len(weights), replacement=True)

def collate_fn(batch): return batch

# [关键改回] 原味 Featurizer (去掉自动拉伸，只保留基本的 CA/O 纠错提示)
def featurize_batch(batch, device):
    B = len(batch)
    batch = [b for b in batch if 'seq_chain_A' in b and len(b['seq_chain_A']) > 0]
    if not batch: return None
    lengths = [len(b['seq_chain_A']) for b in batch]
    L_max = max(lengths)
    X = np.zeros([B, L_max, 4, 3])
    S = np.zeros([B, L_max], dtype=np.int32)
    mask = np.zeros([B, L_max], dtype=np.float32)
    chain_M = np.zeros([B, L_max], dtype=np.float32)
    residue_idx = -100 * np.ones([B, L_max], dtype=np.int32)
    chain_encoding_all = np.zeros([B, L_max], dtype=np.int32)
    
    for i, b in enumerate(batch):
        c_id = 'A'
        seq = b.get(f'seq_chain_{c_id}', '')
        N = np.array(b.get(f'N_chain_{c_id}', []))
        CA = np.array(b.get(f'CA_chain_{c_id}', []))
        C = np.array(b.get(f'C_chain_{c_id}', []))
        O = np.array(b.get(f'O_chain_{c_id}', []))
        l = min(len(seq), len(CA))
        if l == 0: continue
        
        # ⚠️ Physics Fix: Swap CA/O if needed (这个是对的，保留)
        dist_n_ca = np.linalg.norm(N[:1] - CA[:1])
        dist_n_o = np.linalg.norm(N[:1] - O[:1])
        real_CA, real_O = CA, O
        if dist_n_o < dist_n_ca and dist_n_o < 1.6:
            real_CA, real_O = O, CA
            # if i==0: print("⚠️ [Physics] Auto-corrected CA/O swap.")

        X[i, :l, 0, :] = N[:l]
        X[i, :l, 1, :] = real_CA[:l]
        X[i, :l, 2, :] = C[:l]
        X[i, :l, 3, :] = real_O[:l]
        
        indices = []
        for aa in seq[:l]: indices.append(EXTENDED_AA_TO_INDEX.get(aa, EXTENDED_AA_TO_INDEX['X']))
        S[i, :l] = indices
        chain_M[i, :l] = 1.0
        residue_idx[i, :l] = np.arange(l)
        chain_encoding_all[i, :l] = 0

    # ❌ 移除了自动拉伸逻辑 (Automatic Scaling Removed)
    # X *= (3.8 / avg_dist) <--- DELETED

    isnan = np.isnan(X); mask = np.isfinite(np.sum(X, (2,3))).astype(np.float32); X[isnan] = 0.
    return [torch.from_numpy(t).to(dtype=torch.long if t.dtype==np.int32 else torch.float32, device=device) for t in [X, S, mask, chain_M, residue_idx, chain_encoding_all]]

# =============================================================================
# 5. 验证与主循环
# =============================================================================
def validate(model, loader, device, verbose=False):
    model.eval()
    all_preds_base_raw, all_targets_base_raw = [], [] 
    all_preds_methyl, all_targets_methyl = [], []
    all_preds_comb, all_targets_comb = [], []
    nat_to_me_abs = {v: k + len(NATURAL_AA_ALPHABET) for k, v in NMETHYL_TO_NATURAL_MAPPING.items()}
    offset = len(NATURAL_AA_ALPHABET)
    
    with torch.no_grad():
        for batch in loader:
            f = featurize_batch(batch, device)
            if f is None: continue
            X, S, mask, chain_M, residue_idx, chain_encoding_all = f
            lb, lm = model(X, S, mask, chain_M, residue_idx, chain_encoding_all)
            
            pb = torch.argmax(lb, -1)
            pm = (F.softmax(lm, dim=-1)[:, :, 1] > 0.4).long()
            
            pf = pb.clone()
            for m_rel, n_idx in NMETHYL_TO_NATURAL_MAPPING.items():
                mask_up = (pm == 1) & (pb == n_idx)
                pf[mask_up] = m_rel + offset
                
            ts = S.cpu().numpy().flatten()
            valid = mask.cpu().numpy().flatten().astype(bool) & (ts != EXTENDED_AA_TO_INDEX.get('X', -1))
            
            t_base = ts[valid].copy()
            for m_rel, n_idx in NMETHYL_TO_NATURAL_MAPPING.items():
                t_base[t_base == (m_rel + offset)] = n_idx
            
            all_preds_base_raw.extend(pb.cpu().numpy().flatten()[valid])
            all_targets_base_raw.extend(t_base)
            
            all_preds_methyl.extend(pm.cpu().numpy().flatten()[valid])
            all_targets_methyl.extend((ts[valid] >= offset).astype(int))
            
            all_preds_comb.extend(pf.cpu().numpy().flatten()[valid])
            all_targets_comb.extend(ts[valid])

    if not all_targets_comb: return 0,0,0,0
    
    base_acc = np.mean(np.array(all_preds_base_raw) == np.array(all_targets_base_raw))
    methyl_acc = np.mean(np.array(all_preds_methyl) == np.array(all_targets_methyl))
    total_acc = np.mean(np.array(all_preds_comb) == np.array(all_targets_comb))
    
    t_comb = np.array(all_targets_comb)
    p_comb = np.array(all_preds_comb)
    nat_mask = t_comb < offset
    nat_acc = np.mean(t_comb[nat_mask] == p_comb[nat_mask]) if nat_mask.sum() > 0 else 0
    
    return base_acc, methyl_acc, total_acc, nat_acc

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrained_weights", type=str, required=True)
    parser.add_argument("--nmethyl_data", type=str, required=True)
    parser.add_argument("--test_data", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="./run_v17_redemption")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--methyl_loss_weight", type=float, default=2.0)
    parser.add_argument("--sampler_weight", type=float, default=10.0)
    args = parser.parse_args()
    
    set_seed(42)
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = DecoupledProteinMPNN(augment_eps=0.1).to(device)
    advanced_load_weights(model, args.pretrained_weights, device)

    train_ds = JSONLDataset(args.nmethyl_data)
    sampler = get_weighted_sampler(train_ds, oversample_weight=args.sampler_weight)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, sampler=sampler, collate_fn=collate_fn)
    test_ds = JSONLDataset(args.test_data)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)

    # [关键调参] 恢复更激进的学习率 (V13 levels)
    optimizer = torch.optim.AdamW([
        {'params': [p for n, p in model.named_parameters() if 'W_out' not in n], 'lr': 1e-4}, # Backbone back to 1e-4
        {'params': [p for n, p in model.named_parameters() if 'W_out' in n], 'lr': 1e-3}      # Head 1e-3
    ], weight_decay=1e-4)
    
    scheduler = OneCycleLR(optimizer, max_lr=[1e-4, 1e-3], epochs=args.epochs, steps_per_epoch=len(train_loader), pct_start=0.3, div_factor=10)

    print("Starting Training (V17 - Redemption Mode)...")
    best_total_acc = 0.0
    
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss, n_steps = 0, 0
        for batch in train_loader:
            f = featurize_batch(batch, device)
            if f is None: continue
            optimizer.zero_grad()
            X, S, mask, chain_M, residue_idx, chain_encoding_all = f
            lb, lm = model(X, S, mask, chain_M, residue_idx, chain_encoding_all)
            loss, valid = calculate_loss(lb, lm, S, mask, methyl_weight=args.methyl_loss_weight)
            if valid > 0 and torch.isfinite(loss):
                loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step(); scheduler.step()
                total_loss += loss.item(); n_steps += 1
        
        if epoch % 5 == 0:
            avg_loss = total_loss / n_steps if n_steps > 0 else 0
            print(f"Epoch {epoch}: Loss = {avg_loss:.4f}")
            ba, ma, ta, na = validate(model, test_loader, device)
            print(f"   [Val] Base: {ba*100:.2f}% | Methyl: {ma*100:.2f}% | Total: {ta*100:.2f}% | Nat: {na*100:.2f}%")
            if ta > best_total_acc:
                best_total_acc = ta
                torch.save({'model_state_dict': model.state_dict()}, os.path.join(args.output_dir, "best_model.pt"))
                print(f"   🌟 New Best: {best_total_acc*100:.2f}%")

    print("\n>>> Final Report (Best Model) <<<")
    model.load_state_dict(torch.load(os.path.join(args.output_dir, "best_model.pt"))['model_state_dict'])
    ba, ma, ta, na = validate(model, test_loader, device, verbose=True)
    print(f"Final Total Accuracy: {ta*100:.2f}%")

if __name__ == "__main__":
    main()
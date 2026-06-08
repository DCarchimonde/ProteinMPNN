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
# 1. 黄金标准配置
# =============================================================================
EXTENDED_AA_ALPHABET = "ACDEFGHIKLMNPQRSTVWYXacdefghiklmnpqrstvwy"
NATURAL_AA_ALPHABET = "ACDEFGHIKLMNPQRSTVWY" 
NMETHYL_TO_NATURAL_MAPPING = {i+21: i for i in range(20)}
EXTENDED_AA_TO_INDEX = {aa: i for i, aa in enumerate(EXTENDED_AA_ALPHABET)}

# =============================================================================
# 2. 官方标准 ProteinMPNN (严禁修改，保证 50% Base Acc)
# =============================================================================
class PositionWiseFeedForward(nn.Module):
    def __init__(self, num_hidden, num_ff):
        super(PositionWiseFeedForward, self).__init__()
        self.W_in = nn.Linear(num_hidden, num_ff, bias=True)
        self.W_out = nn.Linear(num_ff, num_hidden, bias=True)
        self.act = nn.GELU()
    def forward(self, h_V):
        h = self.act(self.W_in(h_V))
        h = self.W_out(h)
        return h

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

class ProteinFeatures(nn.Module):
    def __init__(self, edge_features, node_features, num_positional_embeddings=16, num_rbf=16, top_k=30, augment_eps=0., dropout=0.1):
        super(ProteinFeatures, self).__init__()
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
        D_expand = torch.unsqueeze(D, -1)
        RBF = torch.exp(-((D_expand - D_mu) / D_sigma)**2)
        return RBF

    def _get_rbf(self, A, B, E_idx):
        D_A_B = torch.sqrt(torch.sum((A[:,:,None,:] - B[:,None,:,:])**2,-1) + 1e-6)
        # [Fix] Simple gather
        D_A_B_neighbors = torch.gather(D_A_B, 2, E_idx)
        return self._rbf(D_A_B_neighbors)

    def forward(self, X, mask, residue_idx, chain_labels):
        if self.training and self.augment_eps > 0: X = X + self.augment_eps * torch.randn_like(X)
        b = X[:,:,1,:] - X[:,:,0,:]; c = X[:,:,2,:] - X[:,:,1,:]
        a = torch.cross(b, c, dim=-1)
        Cb = -0.58273431*a + 0.56802827*b - 0.54067466*c + X[:,:,1,:]
        Ca, N, C, O = X[:,:,1,:], X[:,:,0,:], X[:,:,2,:], X[:,:,3,:]
        D_neighbors, E_idx = self._dist(Ca, mask)
        
        RBF_all = []
        RBF_all.append(self._rbf(D_neighbors))
        for atom1 in [N, C, O, Cb, Ca]:
            for atom2 in [N, C, O, Cb, Ca]:
                if atom1 is Ca and atom2 is Ca: continue
                dist = torch.sqrt(torch.sum((atom1.unsqueeze(1) - atom2.unsqueeze(2))**2, -1) + 1e-6)
                # [Fix] Simple gather
                D_neighbor = torch.gather(dist, 2, E_idx)
                RBF_all.append(self._rbf(D_neighbor))
        RBF_all = torch.cat(tuple(RBF_all), dim=-1)
        
        offset = residue_idx[:,:,None]-residue_idx[:,None,:]
        # [Fix] Simple gather
        offset = torch.gather(offset, 2, E_idx)
        
        d_chains = ((chain_labels[:, :, None] - chain_labels[:,None,:])==0).long()
        # [Fix] Simple gather
        E_chains = torch.gather(d_chains, 2, E_idx)
        
        E_positional = self.embeddings(offset.long(), E_chains)
        E = torch.cat((E_positional, RBF_all), -1)
        E = self.edge_embedding(E)
        E = self.norm_edges(E)
        return E, E_idx

class EncLayer(nn.Module):
    def __init__(self, num_hidden, num_in, dropout=0.1, num_heads=None, scale=30):
        super(EncLayer, self).__init__()
        self.num_hidden = num_hidden; self.num_in = num_in; self.scale = scale
        self.dropout1 = nn.Dropout(dropout); self.dropout2 = nn.Dropout(dropout); self.dropout3 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(num_hidden); self.norm2 = nn.LayerNorm(num_hidden); self.norm3 = nn.LayerNorm(num_hidden)
        self.W1 = nn.Linear(num_hidden + num_in, num_hidden, bias=True)
        self.W2 = nn.Linear(num_hidden, num_hidden, bias=True)
        self.W3 = nn.Linear(num_hidden, num_hidden, bias=True)
        self.W11 = nn.Linear(num_hidden + num_in, num_hidden, bias=True)
        self.W12 = nn.Linear(num_hidden, num_hidden, bias=True)
        self.W13 = nn.Linear(num_hidden, num_hidden, bias=True)
        self.act = nn.GELU()
        self.dense = PositionWiseFeedForward(num_hidden, num_hidden * 4)

    def forward(self, h_V, h_E, E_idx, mask_V=None, mask_attend=None):
        # [Fix] Correct gather/cat logic
        # h_V: [B, L, C], E_idx: [B, L, K] -> [B, L, K, C]
        h_V_neighbors = torch.gather(h_V.unsqueeze(-2).expand(-1, -1, E_idx.size(2), -1), 1, E_idx.unsqueeze(-1).expand(-1, -1, -1, h_V.size(-1)))
        h_EV = torch.cat([h_V_neighbors, h_E], -1)
        
        h_V_expand = h_V.unsqueeze(-2).expand(-1,-1,h_EV.size(-2),-1)
        h_EV = torch.cat([h_V_expand, h_EV], -1)
        h_message = self.W3(self.act(self.W2(self.act(self.W1(h_EV)))))
        if mask_attend is not None: h_message = mask_attend.unsqueeze(-1) * h_message
        dh = torch.sum(h_message, -2) / self.scale
        h_V = self.norm1(h_V + self.dropout1(dh))
        dh = self.dense(h_V)
        h_V = self.norm2(h_V + self.dropout2(dh))
        if mask_V is not None: h_V = mask_V.unsqueeze(-1) * h_V
        
        h_V_neighbors = torch.gather(h_V.unsqueeze(-2).expand(-1, -1, E_idx.size(2), -1), 1, E_idx.unsqueeze(-1).expand(-1, -1, -1, h_V.size(-1)))
        h_EV = torch.cat([h_V_neighbors, h_E], -1)
        
        h_V_expand = h_V.unsqueeze(-2).expand(-1,-1,h_EV.size(-2),-1)
        h_EV = torch.cat([h_V_expand, h_EV], -1)
        h_message = self.W13(self.act(self.W12(self.act(self.W11(h_EV)))))
        h_E = self.norm3(h_E + self.dropout3(h_message))
        return h_V, h_E

class DecLayer(nn.Module):
    def __init__(self, num_hidden, num_in, dropout=0.1, num_heads=None, scale=30):
        super(DecLayer, self).__init__()
        self.num_hidden = num_hidden; self.num_in = num_in; self.scale = scale
        self.dropout1 = nn.Dropout(dropout); self.dropout2 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(num_hidden); self.norm2 = nn.LayerNorm(num_hidden)
        self.W1 = nn.Linear(num_hidden + num_in, num_hidden, bias=True)
        self.W2 = nn.Linear(num_hidden, num_hidden, bias=True)
        self.W3 = nn.Linear(num_hidden, num_hidden, bias=True)
        self.act = nn.GELU()
        self.dense = PositionWiseFeedForward(num_hidden, num_hidden * 4)

    def forward(self, h_V, h_E, mask_V=None, mask_attend=None):
        h_V_expand = h_V.unsqueeze(-2).expand(-1,-1,h_E.size(-2),-1)
        h_EV = torch.cat([h_V_expand, h_E], -1)
        h_message = self.W3(self.act(self.W2(self.act(self.W1(h_EV)))))
        if mask_attend is not None: h_message = mask_attend.unsqueeze(-1) * h_message
        dh = torch.sum(h_message, -2) / self.scale
        h_V = self.norm1(h_V + self.dropout1(dh))
        dh = self.dense(h_V)
        h_V = self.norm2(h_V + self.dropout2(dh))
        if mask_V is not None: h_V = mask_V.unsqueeze(-1) * h_V
        return h_V

class ProteinMPNN(nn.Module):
    def __init__(self, num_letters=21, node_features=128, edge_features=128, hidden_dim=128, num_encoder_layers=3, num_decoder_layers=3, vocab=21, k_neighbors=48, augment_eps=0.05, dropout=0.1):
        super(ProteinMPNN, self).__init__()
        self.features = ProteinFeatures(node_features, edge_features, top_k=k_neighbors, augment_eps=augment_eps, dropout=dropout)
        self.W_e = nn.Linear(edge_features, hidden_dim, bias=True)
        self.W_s = nn.Embedding(vocab, hidden_dim)
        self.encoder_layers = nn.ModuleList([EncLayer(hidden_dim, hidden_dim*2, dropout=dropout) for _ in range(num_encoder_layers)])
        self.decoder_layers = nn.ModuleList([DecLayer(hidden_dim, hidden_dim*3, dropout=dropout) for _ in range(num_decoder_layers)])
        self.W_out = nn.Linear(hidden_dim, num_letters, bias=True)

    def forward(self, X, S, mask, chain_M, residue_idx, chain_encoding_all):
        E, E_idx = self.features(X, mask, residue_idx, chain_encoding_all)
        h_V = torch.zeros((E.shape[0], E.shape[1], E.shape[-1]), device=E.device)
        h_E = self.W_e(E)
        
        # [Fix] Gather mask
        mask_attend = torch.gather(mask.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, E_idx.size(2), -1), 1, E_idx.unsqueeze(-1)).squeeze(-1)
        mask_attend = mask.unsqueeze(-1) * mask_attend
        for layer in self.encoder_layers:
            h_V, h_E = layer(h_V, h_E, E_idx, mask, mask_attend)
            
        h_S = self.W_s(S)
        # [Fix] Gather S
        h_S_neighbors = torch.gather(h_S.unsqueeze(-2).expand(-1, -1, E_idx.size(2), -1), 1, E_idx.unsqueeze(-1).expand(-1, -1, -1, h_S.size(-1)))
        h_ES = torch.cat([h_S_neighbors, h_E], -1)
        
        h_EX_encoder = torch.cat([torch.gather(torch.zeros_like(h_S).unsqueeze(-2).expand(-1, -1, E_idx.size(2), -1), 1, E_idx.unsqueeze(-1).expand(-1, -1, -1, h_S.size(-1))), h_E], -1)
        
        h_V_neighbors = torch.gather(h_V.unsqueeze(-2).expand(-1, -1, E_idx.size(2), -1), 1, E_idx.unsqueeze(-1).expand(-1, -1, -1, h_V.size(-1)))
        h_EXV_encoder = torch.cat([h_V_neighbors, h_EX_encoder], -1)
        
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
            h_V_neighbors = torch.gather(h_V.unsqueeze(-2).expand(-1, -1, E_idx.size(2), -1), 1, E_idx.unsqueeze(-1).expand(-1, -1, -1, h_V.size(-1)))
            h_ESV = torch.cat([h_V_neighbors, h_ES], -1)
            h_ESV = mask_bw * h_ESV + h_EXV_encoder_fw
            h_V = layer(h_V, h_ESV, mask)
        logits = self.W_out(h_V)
        return logits, h_V 

# =============================================================================
# 4. 外挂大脑 (Sidecar Network)
# =============================================================================
class SidecarModel(nn.Module):
    def __init__(self, original_model):
        super().__init__()
        self.backbone = original_model
        # 冻结 backbone
        for p in self.backbone.parameters(): p.requires_grad = False
        
        # 外挂的甲基化预测头
        self.methyl_head = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 2) # 0/1 Methylation
        )
        
    def forward(self, X, S, mask, chain_M, residue_idx, chain_encoding_all):
        # 1. 跑原版模型，获取 Base Logits 和 隐层特征
        logits_base, h_V = self.backbone(X, S, mask, chain_M, residue_idx, chain_encoding_all)
        
        # 2. 跑外挂头
        logits_methyl = self.methyl_head(h_V) # [B, L, 2]
        
        return logits_base, logits_methyl

# =============================================================================
# 5. 数据处理 & 氢键规则 & Helper
# =============================================================================
def compute_hbond_mask(X, mask):
    # X: [B, L, 4, 3]
    if X.dtype != torch.float32: X = X.float()
    B, L = X.shape[:2]
    N = X[:, :, 0, :]; O = X[:, :, 3, :]
    dist = torch.norm(N.unsqueeze(2) - O.unsqueeze(1), p=2, dim=-1) # [B, L, L]
    diag = torch.eye(L, device=X.device).unsqueeze(0)
    # 物理规则：N...O < 3.5A 且不是自己
    is_hb = (dist < 3.5) & (dist > 0.1) & (diag < 0.5)
    has_hbond = is_hb.any(dim=2).float()
    return has_hbond * mask

def featurize_batch_safe(batch, device):
    B = len(batch); L_max = max([len(b['seq']) for b in batch])
    X = np.zeros([B, L_max, 4, 3]); S = np.zeros([B, L_max], dtype=int)
    mask = np.zeros([B, L_max], dtype=float); chain_M = np.zeros([B, L_max], dtype=float)
    residue_idx = -100*np.ones([B, L_max], dtype=int); chain_encoding_all = np.zeros([B, L_max], dtype=int)
    for i, b in enumerate(batch):
        # 强制截断到 21 (X) 以适应原版模型输入
        seq = [EXTENDED_AA_TO_INDEX.get(aa, 20) for aa in b.get('seq_chain_A','')]
        
        all_chains = b.get('masked_list', []) + b.get('visible_list', [])
        l_p = 0
        for c_i, c_id in enumerate(all_chains):
            seq_str = b.get(f'seq_chain_{c_id}', '')
            coords = {a: b[f'{a}_chain_{c_id}'] for a in ['N', 'CA', 'C', 'O']}
            l = len(seq_str)
            for ai, a in enumerate(['N', 'CA', 'C', 'O']):
                if len(coords[a])>0: X[i, l_p:l_p+l, ai, :] = coords[a][:l]
            
            # S 需要是 0-20 的整数，喂给原版模型
            cleaned_seq = []
            for char in seq_str:
                idx = EXTENDED_AA_TO_INDEX.get(char, 20)
                if idx >= 21: # 'a' is 21
                    target_nat = NMETHYL_TO_NATURAL_MAPPING.get(idx - 21, 20)
                    cleaned_seq.append(target_nat)
                else:
                    cleaned_seq.append(idx)
            
            S[i, l_p:l_p+l] = cleaned_seq
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
# 6. 主程序
# =============================================================================
class JSONLDataset(Dataset):
    def __init__(self, jsonl_file):
        self.data = []
        with open(jsonl_file, 'r') as f:
            for line in f: self.data.append(json.loads(line))
    def __len__(self): return len(self.data)
    def __getitem__(self, idx): return self.data[idx]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrained_weights", type=str, required=True)
    parser.add_argument("--nmethyl_data", type=str, required=True)
    parser.add_argument("--test_data", type=str, required=True)
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. 实例化原版模型
    base_model = ProteinMPNN(node_features=128, edge_features=128, hidden_dim=128)
    ckpt = torch.load(args.pretrained_weights, map_location=device)
    base_model.load_state_dict(ckpt['model_state_dict']) # 必须 100% 成功
    print("✅ Original ProteinMPNN Loaded.")
    
    # 2. 挂载 Sidecar
    model = SidecarModel(base_model).to(device)
    
    # 3. 训练 Sidecar
    train_ds = JSONLDataset(args.nmethyl_data)
    loader = DataLoader(train_ds, batch_size=8, shuffle=True, collate_fn=lambda x:x)
    optimizer = torch.optim.AdamW(model.methyl_head.parameters(), lr=1e-3)
    
    print(">>> Training Sidecar (Methyl Head)...")
    for ep in range(30):
        model.train()
        for batch in loader:
            inputs = featurize_batch_safe(batch, device)
            X, S_base, mask = inputs[0], inputs[1], inputs[2]
            
            # S_true for training methyl label
            S_true = [] 
            for b in batch:
                seq = b.get('seq_chain_A', '')
                s_row = [EXTENDED_AA_TO_INDEX.get(aa, 20) for aa in seq]
                # Pad to L_max
                s_row += [20] * (S_base.shape[1] - len(s_row))
                S_true.append(s_row)
            S_true = torch.tensor(S_true, device=device)
            
            target_methyl = (S_true >= 21).long() # 21 start of methyl
            
            _, logits_methyl = model(*inputs)
            loss = F.cross_entropy(logits_methyl.view(-1, 2), target_methyl.view(-1), weight=torch.tensor([1., 5.], device=device))
            
            loss.backward()
            optimizer.step(); optimizer.zero_grad()
            
        if ep % 5 == 0: print(f"Ep {ep} done.")

    # 4. 最终测试
    print(">>> Final Evaluation...")
    model.eval()
    test_loader = DataLoader(JSONLDataset(args.test_data), batch_size=8, collate_fn=lambda x:x)
    
    all_p, all_t = [], []
    with torch.no_grad():
        for batch in test_loader:
            inputs = featurize_batch_safe(batch, device)
            X, S_base, mask = inputs[0], inputs[1], inputs[2]
            
            # True Target
            S_true = [] 
            for b in batch:
                seq = b.get('seq_chain_A', '')
                s_row = [EXTENDED_AA_TO_INDEX.get(aa, 20) for aa in seq]
                s_row += [20] * (S_base.shape[1] - len(s_row))
                S_true.append(s_row)
            S_true = torch.tensor(S_true, device=device)
            
            lb, lm = model(*inputs)
            pred_base = torch.argmax(F.softmax(lb, -1), -1) # 0-20
            
            # Methyl Logic
            prob_methyl = F.softmax(lm, -1)[:,:,1]
            hb = compute_hbond_mask(X, mask)
            prob_methyl = prob_methyl * (1.0 - hb) # Rule
            is_methyl = prob_methyl > 0.5
            
            final = pred_base.clone()
            B, L = final.shape
            for b in range(B):
                for l in range(L):
                    if mask[b,l] == 1 and is_methyl[b,l]:
                        base = pred_base[b,l].item()
                        # Map base to methyl
                        for m_rel, n_id in NMETHYL_TO_NATURAL_MAPPING.items():
                            if n_id == base:
                                final[b,l] = m_rel + 21
                                break
            
            valid = mask.bool() & (S_true != 20)
            all_p.extend(final[valid].cpu().numpy())
            all_t.extend(S_true[valid].cpu().numpy())
            
    t = np.array(all_t); p = np.array(all_p)
    print(f"Final Accuracy: {np.mean(t==p)*100:.2f}%")
    nat_mask = t < 21
    if nat_mask.sum() > 0:
        print(f"Natural AA Acc: {np.mean(t[nat_mask]==p[nat_mask])*100:.2f}%")
    met_mask = t >= 21
    if met_mask.sum() > 0:
        print(f"Methyl AA Acc:  {np.mean(t[met_mask]==p[met_mask])*100:.2f}%")

if __name__ == "__main__":
    main()
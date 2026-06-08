import os
import sys
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import json
import numpy as np
from torch.utils.data import Dataset, DataLoader 
from tqdm import tqdm

os.environ["OMP_NUM_THREADS"] = "1"

# =============================================================================
# 1. 基础配置
# =============================================================================
NATURAL_AA_ALPHABET = "ACDEFGHIKLMNPQRSTVWY"
EXTENDED_AA_ALPHABET = "ACDEFGHIKLMNPQRSTVWYacdefghiklmnqrstvwvyX"
EXTENDED_AA_TO_INDEX = {aa: i for i, aa in enumerate(EXTENDED_AA_ALPHABET)}

NMETHYL_TO_NATURAL_MAPPING = {} 
for i, aa in enumerate(NATURAL_AA_ALPHABET):
    lower_aa = aa.lower()
    if lower_aa in EXTENDED_AA_TO_INDEX:
        NMETHYL_TO_NATURAL_MAPPING[EXTENDED_AA_TO_INDEX[lower_aa] - 20] = EXTENDED_AA_TO_INDEX[aa]

# =============================================================================
# 2. 模型结构 (RobustHierarchicalProteinMPNN 完美保留)
# =============================================================================
def gather_nodes(nodes, neighbor_idx):
    neighbors_flat = neighbor_idx.view((neighbor_idx.shape[0], -1))
    return torch.gather(nodes, 1, neighbors_flat.unsqueeze(-1).expand(-1, -1, nodes.size(2))).view(list(neighbor_idx.shape)[:3] + [-1])

def cat_neighbors_nodes(h_nodes, h_neighbors, E_idx):
    return torch.cat([h_neighbors, gather_nodes(h_nodes, E_idx)], -1)

class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.W_in = nn.Linear(d_model, d_ff); self.W_out = nn.Linear(d_ff, d_model); self.act = nn.GELU()
    def forward(self, x): return self.W_out(self.act(self.W_in(x)))

class EncLayer(nn.Module):
    def __init__(self, num_hidden, num_in, dropout=0.1, scale=30):
        super().__init__()
        self.scale = scale
        self.dropout1, self.dropout2 = nn.Dropout(dropout), nn.Dropout(dropout)
        self.norm1, self.norm2 = nn.LayerNorm(num_hidden), nn.LayerNorm(num_hidden)
        self.W1, self.W2, self.W3 = nn.Linear(num_hidden+num_in, num_hidden), nn.Linear(num_hidden, num_hidden), nn.Linear(num_hidden, num_hidden)
        self.act, self.dense = nn.GELU(), PositionwiseFeedForward(num_hidden, num_hidden*4, dropout)
    def forward(self, h_V, h_E, E_idx, mask_V=None, mask_attend=None):
        h_EV = torch.cat([h_V.unsqueeze(-2).expand(-1,-1,h_E.size(-2),-1), gather_nodes(h_V, E_idx), h_E], -1) 
        h_message = self.W3(self.act(self.W2(self.act(self.W1(h_EV)))))
        if mask_attend is not None: h_message = mask_attend.unsqueeze(-1) * h_message
        h_V = self.norm1(h_V + self.dropout1(torch.sum(h_message, -2)/self.scale))
        return self.norm2(h_V + self.dropout2(self.dense(h_V))), h_E

class DecLayer(nn.Module):
    def __init__(self, num_hidden, num_in, dropout=0.1, scale=30):
        super().__init__()
        self.scale = scale
        self.dropout1, self.dropout2 = nn.Dropout(dropout), nn.Dropout(dropout)
        self.norm1, self.norm2 = nn.LayerNorm(num_hidden), nn.LayerNorm(num_hidden)
        self.W1, self.W2, self.W3 = nn.Linear(num_hidden+num_in, num_hidden), nn.Linear(num_hidden, num_hidden), nn.Linear(num_hidden, num_hidden)
        self.act, self.dense = nn.GELU(), PositionwiseFeedForward(num_hidden, num_hidden*4, dropout)
    def forward(self, h_V, h_E, mask_V=None, mask_attend=None):
        h_cat = torch.cat([h_V.unsqueeze(-2).expand(-1,-1,h_E.size(-2),-1), h_E], -1) 
        h_message = self.W3(self.act(self.W2(self.act(self.W1(h_cat))))) 
        if mask_attend is not None: h_message = mask_attend.unsqueeze(-1) * h_message
        h_V = self.norm1(h_V + self.dropout1(torch.sum(h_message, -2)/self.scale))
        return self.norm2(h_V + self.dropout2(self.dense(h_V)))

class RobustHierarchicalProteinMPNN(nn.Module):
    def __init__(self, hidden_dim=128, k_neighbors=48, augment_eps=0.0, dropout=0.1): 
        super().__init__()
        self.hidden_dim, self.k_neighbors, self.augment_eps = hidden_dim, k_neighbors, augment_eps
        self.features = nn.ModuleDict({'embeddings': nn.ModuleDict({'linear': nn.Linear(66, 16)}), 'edge_embedding': nn.Linear(416, 128, bias=False), 'norm_edges': nn.LayerNorm(128)})
        self.W_e, self.W_s = nn.Linear(128, hidden_dim, bias=True), nn.Embedding(len(EXTENDED_AA_ALPHABET), hidden_dim)
        self.encoder_layers = nn.ModuleList([EncLayer(hidden_dim, hidden_dim*2, dropout=dropout) for _ in range(3)])
        self.decoder_layers = nn.ModuleList([DecLayer(hidden_dim, hidden_dim*3, dropout=dropout) for _ in range(3)])
        self.W_out_base = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, len(NATURAL_AA_ALPHABET)))
        self.experts = nn.ModuleList([nn.Linear(hidden_dim, 1) for _ in range(len(NATURAL_AA_ALPHABET))])

    def forward(self, X, S, mask, chain_M, residue_idx, chain_encoding_all):
        b, c = X[:,:,1,:] - X[:,:,0,:], X[:,:,2,:] - X[:,:,1,:]
        a = torch.cross(b, c, dim=-1)
        Cb = -0.58273431*a + 0.56802827*b - 0.54067466*c + X[:,:,1,:]
        dist = torch.norm(X[:,:,1,:].unsqueeze(1) - X[:,:,1,:].unsqueeze(2), dim=-1) + (1.0 - (mask.unsqueeze(1)*mask.unsqueeze(2)))*1e8
        E_idx = torch.topk(dist, min(self.k_neighbors, dist.shape[-1]), dim=-1, largest=False)[1]
        offset = torch.gather(residue_idx.unsqueeze(1) - residue_idx.unsqueeze(2), 2, E_idx)
        pos_emb = self.features['embeddings']['linear'](F.one_hot(torch.clip(offset + 32, 0, 64), 66).float())
        RBF_all = [self._rbf(torch.gather(dist, 2, E_idx))]
        for i, a1 in enumerate([X[:,:,0,:], X[:,:,2,:], X[:,:,3,:], Cb, X[:,:,1,:]]):
            for j, a2 in enumerate([X[:,:,0,:], X[:,:,2,:], X[:,:,3,:], Cb, X[:,:,1,:]]):
                if i!=4 or j!=4: RBF_all.append(self._rbf(torch.gather(torch.norm(a1.unsqueeze(1)-a2.unsqueeze(2), dim=-1), 2, E_idx)))
        E = self.features['norm_edges'](self.features['edge_embedding'](torch.cat((pos_emb, torch.cat(RBF_all, dim=-1)), -1)))
        h_V, h_E = torch.zeros((E.shape[0], E.shape[1], E.shape[-1]), device=E.device), self.W_e(E)
        mask_attend = mask.unsqueeze(-1) * gather_nodes(mask.unsqueeze(-1), E_idx).squeeze(-1)
        for layer in self.encoder_layers: h_V, h_E = layer(h_V, h_E, E_idx, mask, mask_attend)
        h_ES = cat_neighbors_nodes(self.W_s(S), h_E, E_idx)
        h_EXV_encoder = cat_neighbors_nodes(h_V, cat_neighbors_nodes(torch.zeros_like(self.W_s(S)), h_E, E_idx), E_idx)
        permutation_matrix_reverse = torch.nn.functional.one_hot(torch.argsort(chain_M*mask + 0.0001), num_classes=E_idx.shape[1]).float()
        order_mask_backward = torch.einsum('ij, biq, bjp->bqp', (1 - torch.triu(torch.ones(E_idx.shape[1], E_idx.shape[1], device=X.device))), permutation_matrix_reverse, permutation_matrix_reverse)
        mask_attend = torch.gather(order_mask_backward, 2, E_idx).unsqueeze(-1)
        mask_bw, mask_fw = mask.view([mask.size(0), mask.size(1), 1, 1]) * mask_attend, mask.view([mask.size(0), mask.size(1), 1, 1]) * (1. - mask_attend)
        for layer in self.decoder_layers: h_V = layer(h_V, mask_bw * cat_neighbors_nodes(h_V, h_ES, E_idx) + mask_fw * h_EXV_encoder, mask)
        
        logits_base = self.W_out_base(h_V)
        logits_experts = torch.cat([expert(h_V) for expert in self.experts], dim=-1)
        return logits_base, logits_experts

    def _rbf(self, D): return torch.exp(-((D.unsqueeze(-1) - torch.linspace(2., 22., 16, device=D.device)) / ((22.-2.)/16)) ** 2)

# =============================================================================
# 3. 防弹加载与数据处理
# =============================================================================
def bulletproof_load_weights(model, checkpoint_path, device):
    ckpt = torch.load(checkpoint_path, map_location=device)
    state_dict = ckpt.get('model_state_dict', ckpt)
    if 'W_out.weight' in state_dict:
        state_dict['W_out_base.3.weight'] = state_dict.pop('W_out.weight')
        state_dict['W_out_base.3.bias'] = state_dict.pop('W_out.bias')
    elif 'module.W_out.weight' in state_dict:
        state_dict['W_out_base.3.weight'] = state_dict.pop('module.W_out.weight')
        state_dict['W_out_base.3.bias'] = state_dict.pop('module.W_out.bias')
        
    model_state = model.state_dict()
    new_state_dict = {}
    for k, v in state_dict.items():
        clean_k = k.replace('module.', '')
        if clean_k in model_state and v.shape == model_state[clean_k].shape:
            new_state_dict[clean_k] = v
        elif clean_k in model_state and v.shape != model_state[clean_k].shape:
            new_v = model_state[clean_k].clone()
            slices = tuple(slice(0, min(dim_v, dim_m)) for dim_v, dim_m in zip(v.shape, new_v.shape))
            new_v[slices] = v[slices]
            new_state_dict[clean_k] = new_v
    model.load_state_dict(new_state_dict, strict=False)

def featurize_batch(batch, device):
    B = len(batch)
    batch = [b for b in batch if 'seq_chain_A' in b and len(b['seq_chain_A']) > 0]
    if not batch: return None
    L_max = max([len(b['seq_chain_A']) for b in batch])
    X, S = np.zeros([B, L_max, 4, 3]), np.zeros([B, L_max], dtype=np.int32)
    residue_idx, chain_M, chain_encoding_all = -100*np.ones([B, L_max], dtype=np.int32), np.ones([B, L_max], dtype=np.float32), np.zeros([B, L_max], dtype=np.int32)
    
    for i, b in enumerate(batch):
        all_chains = b.get('masked_list', []) + b.get('visible_list', [])
        if not all_chains: all_chains = ['A']
        l_p = 0
        for c_i, c_id in enumerate(all_chains):
            seq = b.get(f'seq_chain_{c_id}', '')
            N, CA, C, O = np.array(b.get(f'N_chain_{c_id}', [])), np.array(b.get(f'CA_chain_{c_id}', [])), np.array(b.get(f'C_chain_{c_id}', [])), np.array(b.get(f'O_chain_{c_id}', []))
            l = min(len(seq), len(CA))
            if l == 0: continue
            if len(N) > 0 and len(CA) > 0 and len(O) > 0:
                if np.linalg.norm(N[:1] - O[:1]) < np.linalg.norm(N[:1] - CA[:1]) and np.linalg.norm(N[:1] - O[:1]) < 1.6: 
                    CA, O = O, CA
            X[i, l_p:l_p+l, 0, :], X[i, l_p:l_p+l, 1, :], X[i, l_p:l_p+l, 2, :], X[i, l_p:l_p+l, 3, :] = N[:l], CA[:l], C[:l], O[:l]
            S[i, l_p:l_p+l] = [EXTENDED_AA_TO_INDEX.get(aa, EXTENDED_AA_TO_INDEX.get('X', 40)) for aa in seq[:l]]
            if c_id in b.get('masked_list', []): chain_M[i, l_p:l_p+l] = 1.0
            residue_idx[i, l_p:l_p+l] = np.arange(l) + c_i * 100
            chain_encoding_all[i, l_p:l_p+l] = c_i
            l_p += l
    mask = np.isfinite(np.sum(X, (2,3))).astype(np.float32)
    X[np.isnan(X)] = 0.
    return [torch.from_numpy(t).to(dtype=torch.long if t.dtype==np.int32 else torch.float32, device=device) for t in [X, S, mask, chain_M, residue_idx, chain_encoding_all]]

class JSONLDataset(Dataset):
    def __init__(self, jsonl_file): self.data = [json.loads(line) for line in open(jsonl_file, 'r')]
    def __len__(self): return len(self.data)
    def __getitem__(self, idx): return self.data[idx]

# =============================================================================
# 4. 微调核心逻辑 (只微调专家头)
# =============================================================================
def finetune_experts(model, train_loader, val_loader, device, save_path, epochs=5):
    print("\n🔒 [STEP 1] 冻结主干网络，保护原始 38% 准确率...")
    for name, param in model.named_parameters():
        if 'experts' not in name:
            param.requires_grad = False  # 锁死所有非专家头的权重
        else:
            param.requires_grad = True   # 只允许专家头更新
            
    # 只把 experts 传给优化器
    optimizer = torch.optim.Adam(model.experts.parameters(), lr=1e-3)
    
    # 【核心魔法】：因为正样本(甲基化)只有17%，负样本83%。我们给正样本加 5 倍的权重 (83/17 ≈ 5)
    # 强迫模型重视甲基化！
    pos_weight = torch.tensor([5.0]).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    
    offset = len(NATURAL_AA_ALPHABET) # 20
    x_idx = EXTENDED_AA_TO_INDEX.get('X', 40)
    
    best_f1 = 0.0
    
    print("\n🚀 [STEP 2] 开始为期 5 个 Epoch 的专家头专项补习...")
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} Training"):
            f = featurize_batch(batch, device)
            if f is None: continue
            X, S, mask, chain_M, residue_idx, chain_encoding_all = f
            
            optimizer.zero_grad()
            # 前向传播
            _, l_experts = model(X, S, mask, chain_M, residue_idx, chain_encoding_all)
            
            # 1. 还原出真实的 Base AA
            true_base_idx = S.clone()
            for m_rel, n_idx in NMETHYL_TO_NATURAL_MAPPING.items():
                true_base_idx[true_base_idx == (m_rel + offset)] = n_idx
            true_base_idx[true_base_idx >= 20] = 0 # 安全保护
            
            # 2. 获取目标专家的 logit
            expert_logit = torch.gather(l_experts, -1, true_base_idx.unsqueeze(-1)).squeeze(-1)
            
            # 3. 构建 0/1 标签
            tgts = S.view(-1)
            expert_logit_flat = expert_logit.view(-1)
            mask_flat = mask.view(-1).bool()
            valid = mask_flat & (tgts != x_idx)
            
            if not valid.any(): continue
            
            true_binary_labels = (tgts[valid] >= offset).float()
            logits_valid = expert_logit_flat[valid]
            
            # 4. 计算 Loss 并反向传播
            loss = criterion(logits_valid, true_binary_labels)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
        # ==================== 验证环节 ====================
        model.eval()
        true_labels, pred_probs = [], []
        with torch.no_grad():
            for batch in val_loader:
                f = featurize_batch(batch, device)
                if f is None: continue
                X, S, mask, chain_M, residue_idx, chain_encoding_all = f
                _, l_experts = model(X, S, mask, chain_M, residue_idx, chain_encoding_all)
                
                true_base_idx = S.clone()
                for m_rel, n_idx in NMETHYL_TO_NATURAL_MAPPING.items():
                    true_base_idx[true_base_idx == (m_rel + offset)] = n_idx
                true_base_idx[true_base_idx >= 20] = 0
                
                expert_logit = torch.gather(l_experts, -1, true_base_idx.unsqueeze(-1)).squeeze(-1)
                prob_methyl = torch.sigmoid(expert_logit)
                
                tgts = S.cpu().numpy().flatten()
                p_methyl = prob_methyl.cpu().numpy().flatten()
                valid = mask.cpu().numpy().flatten().astype(bool) & (tgts != x_idx)
                
                true_labels.extend((tgts[valid] >= offset).astype(int))
                pred_probs.extend(p_methyl[valid])
                
        # 计算 0.5 阈值下的 F1 作为保存标准
        t_l = np.array(true_labels)
        p_p = np.array(pred_probs) > 0.5
        TP = np.sum((t_l == 1) & (p_p == 1))
        FP = np.sum((t_l == 0) & (p_p == 1))
        FN = np.sum((t_l == 1) & (p_p == 0))
        precision = TP / (TP + FP) if (TP + FP) > 0 else 0
        recall = TP / (TP + FN) if (TP + FN) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        print(f"✅ Epoch {epoch+1} 结束 | Train Loss: {total_loss/len(train_loader):.4f} | Val F1-Score (Thr=0.5): {f1*100:.2f}")
        
        if f1 > best_f1:
            best_f1 = f1
            # 提取原权重的 model_state_dict 来保存
            torch.save({'model_state_dict': model.state_dict()}, save_path)
            print(f"⭐ F1提升！保存最佳微调模型至: {save_path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True, help="原始的 V28 模型路径")
    parser.add_argument("--train_data", type=str, required=True, help="你的训练集 jsonl")
    parser.add_argument("--test_data", type=str, required=True, help="你的测试集 jsonl")
    parser.add_argument("--save_path", type=str, default="./best_finetuned_experts.pt")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = RobustHierarchicalProteinMPNN(augment_eps=0.0).to(device) 
    
    # 1. 完美加载原始权重
    bulletproof_load_weights(model, args.model_path, device)
    
    # 2. 准备数据 (用训练集补习，用测试集验证)
    train_ds = JSONLDataset(args.train_data)
    test_ds = JSONLDataset(args.test_data)
    train_loader = DataLoader(train_ds, batch_size=16, shuffle=True, collate_fn=lambda x:x)
    test_loader = DataLoader(test_ds, batch_size=8, shuffle=False, collate_fn=lambda x:x)

    # 3. 开启微调！
    finetune_experts(model, train_loader, test_loader, device, args.save_path, epochs=5)

if __name__ == "__main__":
    main()
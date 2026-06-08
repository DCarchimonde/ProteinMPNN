import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json
import random
import os
import copy

# ============================================================================
# 1. 核心模型架构 (保持稳定版)
# ============================================================================

class PositionalEncodings(nn.Module):
    def __init__(self, num_embeddings, max_relative_feature=32):
        super(PositionalEncodings, self).__init__()
        self.num_embeddings = num_embeddings
        self.max_relative_feature = max_relative_feature
        self.linear = nn.Linear(2*max_relative_feature+2, num_embeddings)

    def forward(self, offset, mask):
        d = torch.clip(offset + self.max_relative_feature, 0, 2*self.max_relative_feature)
        d_onehot = F.one_hot(d, 2*self.max_relative_feature+1).float() 
        padding = torch.ones(d_onehot.shape[:-1] + (1,), device=d_onehot.device)
        E_in = torch.cat([d_onehot, padding], dim=-1) 
        return self.linear(E_in)

class ProteinFeatures(nn.Module):
    def __init__(self, edge_features, node_features, num_positional_embeddings=16,
        num_rbf=16, top_k=30, augment_eps=0.0):
        super(ProteinFeatures, self).__init__()
        self.edge_features = edge_features
        self.node_features = node_features
        self.top_k = top_k
        self.augment_eps = augment_eps 
        self.num_rbf = num_rbf
        self.num_positional_embeddings = num_positional_embeddings

        self.embeddings = PositionalEncodings(num_positional_embeddings)
        node_in, edge_in = 6, num_positional_embeddings + num_rbf*25
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
        D_min, D_max, D_count = 0., 20., self.num_rbf
        D_mu = torch.linspace(D_min, D_max, D_count, device=device)
        D_mu = D_mu.view([1,1,1,-1])
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
        atoms = [X[:,:,0,:], X[:,:,1,:], X[:,:,2,:], X[:,:,3,:], Cb]
        D_neighbors, E_idx = self._dist(X[:,:,1,:], mask) 
        RBF_all = []
        for atom1 in atoms:
            for atom2 in atoms:
                RBF_all.append(self._get_rbf(atom1, atom2, E_idx))
        RBF_all = torch.cat(tuple(RBF_all), dim=-1)
        offset = residue_idx[:,:,None] - residue_idx[:,None,:]
        offset = torch.gather(offset, 2, E_idx)
        E_positional = self.embeddings(offset.long(), mask)
        E = torch.cat((E_positional, RBF_all), dim=-1)
        E = self.edge_embedding(E)
        E = self.norm_edges(E)
        return E, E_idx

class EncLayer(nn.Module):
    def __init__(self, num_hidden, num_in, dropout=0.1, scale=30, num_heads=8):
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
        self.W1 = nn.Linear(num_hidden + num_in, num_hidden, bias=True)
        self.W2 = nn.Linear(num_hidden, num_hidden, bias=True)
        self.W3 = nn.Linear(num_hidden, num_hidden, bias=True)
        self.W11 = nn.Linear(num_hidden + num_in, num_hidden, bias=True)
        self.W12 = nn.Linear(num_hidden, num_hidden, bias=True)
        self.W13 = nn.Linear(num_hidden, num_hidden, bias=True)
        self.act = nn.GELU()
        self.dense = nn.Sequential(
            nn.Linear(num_hidden, num_hidden*4),
            nn.GELU(),
            nn.Linear(num_hidden*4, num_hidden)
        )

    def forward(self, h_V, h_E, E_idx, mask_V=None, mask_attend=None):
        B, L, K = E_idx.shape
        neighbor_idx = E_idx.view(B, -1).unsqueeze(-1).expand(-1, -1, h_V.size(-1))
        h_V_src = torch.gather(h_V, 1, neighbor_idx).view(B, L, K, -1)
        h_V_dst = h_V.unsqueeze(2).expand(-1,-1, K, -1)
        h_EV = torch.cat([h_V_dst, h_E, h_V_src], dim=-1)
        h_message = self.W3(self.act(self.W2(self.act(self.W1(h_EV)))))
        if mask_attend is not None:
            h_message = mask_attend.unsqueeze(-1) * h_message
        dh = torch.sum(h_message, dim=2) / self.scale
        h_V = self.norm1(h_V + self.dropout1(dh))
        dh = self.dense(h_V)
        h_V = self.norm2(h_V + self.dropout2(dh))
        if mask_V is not None:
            mask_V = mask_V.unsqueeze(-1)
            h_V = mask_V * h_V
        h_EV = torch.cat([h_V.unsqueeze(2).expand(-1,-1, K, -1), h_E, h_V_src], dim=-1)
        h_message = self.W13(self.act(self.W12(self.act(self.W11(h_EV)))))
        h_E = self.norm3(h_E + self.dropout3(h_message))
        return h_V, h_E

class DecLayer(nn.Module):
    def __init__(self, num_hidden, num_in, dropout=0.1, scale=30, num_heads=8):
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
        self.dense = nn.Sequential(
            nn.Linear(num_hidden, num_hidden*4),
            nn.GELU(),
            nn.Linear(num_hidden*4, num_hidden)
        )

    def forward(self, h_V, h_E, mask_V=None, mask_attend=None):
        h_EV = torch.cat([h_V.unsqueeze(2).expand(-1,-1,h_E.size(2),-1), h_E], dim=-1)
        h_message = self.W3(self.act(self.W2(self.act(self.W1(h_EV)))))
        if mask_attend is not None:
            h_message = mask_attend.unsqueeze(-1) * h_message
        dh = torch.sum(h_message, dim=2) / self.scale
        h_V = self.norm1(h_V + self.dropout1(dh))
        dh = self.dense(h_V)
        h_V = self.norm2(h_V + self.dropout2(dh))
        if mask_V is not None:
            mask_V = mask_V.unsqueeze(-1)
            h_V = mask_V * h_V
        return h_V

class ProteinMPNN_Hybrid(nn.Module):
    def __init__(self, num_letters=21, node_features=128, edge_features=128, hidden_dim=128, num_encoder_layers=3, num_decoder_layers=3, k_neighbors=48):
        super(ProteinMPNN_Hybrid, self).__init__()
        self.features = ProteinFeatures(edge_features, node_features, top_k=k_neighbors)
        self.W_e = nn.Linear(edge_features, hidden_dim, bias=True)
        self.W_s = nn.Embedding(num_letters, hidden_dim)
        self.encoder_layers = nn.ModuleList([EncLayer(hidden_dim, hidden_dim*2) for _ in range(num_encoder_layers)])
        self.decoder_layers = nn.ModuleList([DecLayer(hidden_dim, hidden_dim*3) for _ in range(num_decoder_layers)])
        self.W_out = nn.Linear(hidden_dim, num_letters, bias=True)
        self.W_methyl = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 2)
        )

    def _gather(self, nodes, idx):
        B, L, C = nodes.shape
        neighbors = idx.view(B, -1) 
        neighbors = neighbors.unsqueeze(-1).expand(-1, -1, C) 
        return torch.gather(nodes, 1, neighbors).view(B, L, -1, C)

    def calc_hbond_mask(self, X, mask):
        # 物理外挂: 计算骨架氢键 N-H...O
        N = X[:, :, 0, :]
        O = X[:, :, 3, :]
        dist = torch.norm(N.unsqueeze(2) - O.unsqueeze(1), dim=-1)
        L = X.shape[1]
        indices = torch.arange(L, device=X.device)
        non_neighbor = torch.abs(indices.unsqueeze(1) - indices.unsqueeze(0)) > 1
        is_hbond_pair = (dist < 3.5) & non_neighbor.unsqueeze(0)
        mask_2d = mask.unsqueeze(1) * mask.unsqueeze(2)
        is_hbond_pair = is_hbond_pair & mask_2d.bool()
        has_hbond = is_hbond_pair.any(dim=2).float() 
        return has_hbond

    def forward(self, X, S, mask, chain_encoding_all):
        E, E_idx = self.features(X, mask, torch.arange(X.shape[1], device=X.device).unsqueeze(0).repeat(X.shape[0], 1), chain_encoding_all)
        h_V = torch.zeros((E.shape[0], E.shape[1], E.shape[-1]), device=E.device)
        h_E = self.W_e(E)
        mask_neighbors = self._gather(mask.unsqueeze(-1), E_idx) 
        mask_attend = mask.unsqueeze(-1).unsqueeze(-1) * mask_neighbors 
        mask_attend = mask_attend.squeeze(-1) 
        for layer in self.encoder_layers:
            h_V, h_E = layer(h_V, h_E, E_idx, mask, mask_attend)
        h_S = self.W_s(S)
        h_V_src = self._gather(h_V, E_idx) 
        h_S_src = self._gather(h_S, E_idx) 
        h_ES = torch.cat([h_E, h_V_src, h_S_src], dim=-1) 
        for layer in self.decoder_layers:
            h_V = layer(h_V, h_ES, mask)
        logits_aa = self.W_out(h_V)
        logits_met = self.W_methyl(h_V)
        return logits_aa, logits_met

# ============================================================================
# 2. 数据处理与混合策略训练
# ============================================================================

NATURAL_AA = "ACDEFGHIKLMNPQRSTVWYX"
NAT_TO_IDX = {aa: i for i, aa in enumerate(NATURAL_AA)}

def featurize_batch_safe(batch_json, device):
    B = len(batch_json)
    L_max = max([len(b.get('seq_chain_A', b.get('seq', ''))) for b in batch_json])
    X = torch.zeros([B, L_max, 4, 3], device=device)
    S = torch.zeros([B, L_max], dtype=torch.long, device=device)
    Y_met = torch.zeros([B, L_max], dtype=torch.long, device=device)
    Mask = torch.zeros([B, L_max], device=device)
    valid = []
    
    for i, b in enumerate(batch_json):
        seq_str = b.get('seq_chain_A', b.get('seq', ''))
        if not seq_str: continue
        l = len(seq_str)
        prefix = '_chain_A' if 'N_chain_A' in b else ''
        if f'N{prefix}' not in b: continue 
        valid.append(i)
        
        for atom_idx, atom_name in enumerate(['N', 'CA', 'C', 'O']):
            coords = b[f'{atom_name}{prefix}']
            X[i, :l, atom_idx, :] = torch.tensor(coords, device=device)
        
        # 坐标中心化
        center = X[i, :l, 1, :].mean(dim=0)
        X[i, :l, :, :] = X[i, :l, :, :] - center

        Mask[i, :l] = 1.0
        
        for j, char in enumerate(seq_str):
            is_lower = char.islower()
            upper_char = char.upper()
            if upper_char in NAT_TO_IDX:
                S[i, j] = NAT_TO_IDX[upper_char]
                Y_met[i, j] = 1 if is_lower else 0 
            else:
                S[i, j] = 20
                
    if not valid: return None, None, None, None
    return X, S, Mask, Y_met

def run_training():
    PRETRAINED_PATH = "vanilla_model_weights/v_48_020.pt"
    TRAIN_DATA = "nmethyl_data/training_set/train.jsonl"
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    LR = 1e-4 
    EPOCHS = 50
    BATCH_SIZE = 8
    
    print(f"Using device: {DEVICE}")
    model = ProteinMPNN_Hybrid(k_neighbors=48).to(DEVICE)
    
    try:
        ckpt = torch.load(PRETRAINED_PATH, map_location=DEVICE)
        state_dict = ckpt['model_state_dict']
        model_state = model.state_dict()
        pretrained_dict = {k: v for k, v in state_dict.items() if k in model_state and v.shape == model_state[k].shape}
        model_state.update(pretrained_dict)
        model.load_state_dict(model_state)
        print(f"✅ 成功继承 {len(pretrained_dict)} 层权重。")
    except Exception as e:
        print(f"❌ 权重加载失败: {e}")
        return

    print("🔥 全模型解冻训练 + 混合遮罩策略...")
    for param in model.parameters():
        param.requires_grad = True
    
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    
    class_weights = torch.tensor([1.0, 10.0], device=DEVICE)
    criterion_met = nn.CrossEntropyLoss(weight=class_weights, reduction='none')
    criterion_aa = nn.CrossEntropyLoss(reduction='none')
    
    print("Loading dataset...")
    with open(TRAIN_DATA, 'r') as f:
        full_data = [json.loads(line) for line in f]
    print(f"Dataset size: {len(full_data)}")

    # 简单的单位检查
    with torch.no_grad():
        X_check, _, _, _ = featurize_batch_safe(full_data[:1], DEVICE)
        dist_check = torch.norm(X_check[0,0,1] - X_check[0,1,1]) # CA-CA 距离
        print(f"🔍 [数据自检] 相邻 CA 距离约为: {dist_check:.2f}")
        if dist_check < 1.0:
            print("⚠️ 警告！你的坐标好像是【纳米】单位！(太小了)")
            print("⚠️ ProteinMPNN 需要【埃】(Angstrom)。请将 coordinates * 10")
        else:
            print("✅ 坐标单位看起来正常 (Angstrom)。")
    
    for epoch in range(EPOCHS):
        model.train()
        random.shuffle(full_data)
        
        total_loss = 0
        count = 0
        
        for i in range(0, len(full_data), BATCH_SIZE):
            batch = full_data[i : i+BATCH_SIZE]
            X, S, Mask, Y_met = featurize_batch_safe(batch, DEVICE)
            if X is None: continue 
            
            chain_encoding = torch.ones_like(S)
            optimizer.zero_grad()
            
            # === 🧬 核心策略：混合遮罩 (Curriculum Learning) ===
            # 不要总是全 Unknown (20)，也不要总是全真实序列。
            # 我们随机选择一个比例 mask_prob (0.1 ~ 1.0)
            # 1.0 = 地狱模式 (全 Unknown), 0.1 = 简单模式 (大部分已知)
            
            mask_prob = random.uniform(0.1, 1.0) # 每一批次的难度随机
            
            # 生成 Mask
            rand_mask = torch.rand(S.shape, device=DEVICE) < mask_prob
            
            # 制作输入 S_input: 
            # 如果位置被 mask 选中 -> 变成 20 (Unknown)
            # 没选中 -> 保持真实 AA
            S_input = S.clone()
            S_input[rand_mask] = 20
            
            # Forward
            logits_aa, logits_met = model(X, S_input, Mask, chain_encoding)
            
            loss_met = criterion_met(logits_met.permute(0,2,1), Y_met)
            loss_aa = criterion_aa(logits_aa.permute(0,2,1), S)
            
            loss_met = (loss_met * Mask).sum() / (Mask.sum() + 1e-6)
            
            # 只计算被 Mask 掉的那部分的 AA Loss (类似 BERT)
            # 这样模型专注于去"猜"那些看不见的部分
            # 注意：这里我们简单点，算全部的 Loss 也可以，但加上 rand_mask 更准
            loss_aa = (loss_aa * Mask).sum() / (Mask.sum() + 1e-6)
            
            loss = 0.5 * loss_aa + 1.0 * loss_met 
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            count += 1
            
            if count % 100 == 0:
                print(f"Ep {epoch} Batch {count} | Loss: {loss.item():.3f} (Prob: {mask_prob:.2f})")
                
        # === 盲测验证 (严格考核) ===
        if epoch % 1 == 0:
            model.eval()
            total_blind_base = 0
            total_blind_final = 0
            val_count = 0
            val_sample = full_data[:32]
            
            with torch.no_grad():
                for i in range(0, len(val_sample), BATCH_SIZE):
                    batch = val_sample[i : i+BATCH_SIZE]
                    X_val, S_val, Mask_val, Y_val = featurize_batch_safe(batch, DEVICE)
                    if X_val is None: continue

                    # 验证时：全盲 (Full Unknown)
                    S_blind = torch.full_like(S_val, 20) 
                    chain_enc = torch.ones_like(S_val)
                    
                    logits_aa, logits_met = model(X_val, S_blind, Mask_val, chain_enc)
                    
                    # 1. Base Acc
                    pred_aa = torch.argmax(logits_aa, dim=-1)
                    correct_base = (pred_aa == S_val)
                    
                    # 2. Methyl Acc + 物理外挂
                    probs_met = F.softmax(logits_met, dim=-1)[:, :, 1]
                    hbond = model.calc_hbond_mask(X_val, Mask_val)
                    pred_met = (probs_met * (1.0-hbond) > 0.5).long()
                    correct_met = (pred_met == Y_val)
                    
                    # 3. Final Acc
                    correct_final = correct_base & correct_met
                    
                    acc_base = (correct_base.float() * Mask_val).sum() / (Mask_val.sum() + 1e-6)
                    acc_final = (correct_final.float() * Mask_val).sum() / (Mask_val.sum() + 1e-6)
                    
                    total_blind_base += acc_base.item()
                    total_blind_final += acc_final.item()
                    val_count += 1
            
            avg_base = total_blind_base / val_count
            avg_final = total_blind_final / val_count
            print(f"=== Epoch {epoch} Result (Blind) ===")
            print(f"   👀 Base Acc:  {avg_base:.3f}")
            print(f"   🔥 Final Acc: {avg_final:.3f}")
            print("==================================")

if __name__ == "__main__":
    run_training()
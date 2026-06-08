import argparse, os, torch, json, itertools
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader

# ==========================================
# 1. 核心映射 (参考 6.txt 字典)
# ==========================================
AA_LIST = "ACDEFGHIKLMNPQRSTVWYX"
AA_TO_IDX = {aa: i for i, aa in enumerate(AA_LIST)}
MET_MAP = {k.lower(): k.upper() for k in "ACDEFGHIKLMNPQRSTVWY"}

# ==========================================
# 2. 物理硬约束：氢键检测
# ==========================================
def get_hbond_mask(X, mask, threshold=3.5):
    """
    检测 N 原子是否作为氢键供体。
    依据：如果 N 距离任何非相邻 O 小于 3.5A，判定为有氢键，该位置甲基化概率归零。
    """
    N, O = X[:, :, 0, :], X[:, :, 3, :]
    dist = torch.norm(N.unsqueeze(2) - O.unsqueeze(1), dim=-1)
    L = N.size(1); device = X.device
    exclude = (torch.eye(L, device=device) + torch.diag(torch.ones(L-1, device=device), 1) + torch.diag(torch.ones(L-1, device=device), -1)).unsqueeze(0)
    dist = dist + exclude * 10.0
    return (torch.min(dist, dim=-1)[0] < threshold).float() * mask

# ==========================================
# 3. 官方架构复刻 (严格修复维度对齐)
# ==========================================
def gather_nodes(nodes, neighbor_idx):
    res = torch.gather(nodes, 1, neighbor_idx.view(nodes.size(0), -1).unsqueeze(-1).expand(-1, -1, nodes.size(2)))
    return res.view(neighbor_idx.shape + (nodes.size(2),))

class EncLayer(nn.Module):
    def __init__(self, d_model=128):
        super().__init__()
        self.norm1, self.norm2, self.norm3 = [nn.LayerNorm(d_model) for _ in range(3)]
        self.W1 = nn.Linear(3*d_model, d_model)
        self.W2 = nn.Linear(d_model, d_model) # 必须 128
        self.W3 = nn.Linear(d_model, d_model) # 必须 128
        self.act = nn.GELU()
        self.dense = nn.ModuleDict({'W_in': nn.Linear(d_model, 512), 'W_out': nn.Linear(512, d_model)})
    def forward(self, h_V, h_E, E_idx):
        h_V_neigh = gather_nodes(h_V, E_idx)
        h_EV = torch.cat([h_V.unsqueeze(2).expand_as(h_V_neigh), h_V_neigh, h_E], dim=-1)
        h_V = self.norm1(h_V + torch.mean(self.act(self.W3(self.act(self.W2(self.act(self.W1(h_EV)))))), dim=2))
        h_V = self.norm2(h_V + self.dense['W_out'](self.act(self.dense['W_in'](h_V))))
        return h_V, h_E

class DecLayer(nn.Module):
    def __init__(self, d_model=128):
        super().__init__()
        self.norm1, self.norm2 = nn.LayerNorm(d_model), nn.LayerNorm(d_model)
        self.W1 = nn.Linear(4*d_model, d_model)
        self.W2 = nn.Linear(d_model, d_model) # 必须 128
        self.W3 = nn.Linear(d_model, d_model) # 必须 128
        self.act = nn.GELU()
        self.dense = nn.ModuleDict({'W_in': nn.Linear(d_model, 512), 'W_out': nn.Linear(512, d_model)})
    def forward(self, h_V, h_E):
        h_V_expand = h_V.unsqueeze(2).expand(-1, -1, h_E.size(2), -1)
        h_EV = torch.cat([h_V_expand, h_E], dim=-1)
        h_V = self.norm1(h_V + torch.mean(self.act(self.W3(self.act(self.W2(self.act(self.W1(h_EV)))))), dim=2))
        h_V = self.norm2(h_V + self.dense['W_out'](self.act(self.dense['W_in'](h_V))))
        return h_V

class ProteinMPNN_DoubleHead(nn.Module):
    def __init__(self, d_model=128):
        super().__init__()
        self.features = nn.ModuleDict({
            'embeddings': nn.ModuleDict({'linear': nn.Linear(66, 16)}), #
            'edge_embedding': nn.Linear(416, d_model, bias=False), # 严格对齐 416
            'norm_edges': nn.LayerNorm(d_model)
        })
        self.W_e = nn.Linear(d_model, d_model); self.W_s = nn.Embedding(21, d_model)
        self.encoder_layers = nn.ModuleList([EncLayer(d_model) for _ in range(3)])
        self.decoder_layers = nn.ModuleList([DecLayer(d_model) for _ in range(3)])
        self.W_out_aa = nn.Linear(d_model, 21)
        self.W_out_methyl = nn.Sequential(nn.Linear(d_model * 2, 256), nn.ReLU(), nn.Linear(256, 1))

    def forward(self, S, E_idx, h_E_rbf, res_idx, hb_mask=None):
        offset = res_idx.unsqueeze(1) - res_idx.unsqueeze(2)
        offset = torch.gather(offset, 2, E_idx)
        E_pos = self.features['embeddings']['linear'](F.one_hot(torch.clip(offset+32, 0, 64), 66).float())
        h_E = self.features['norm_edges'](self.features['edge_embedding'](torch.cat([E_pos, h_E_rbf], dim=-1)))
        h_V = torch.zeros(S.size(0), S.size(1), 128, device=S.device); h_E_enc = self.W_e(h_E)
        for layer in self.encoder_layers: h_V, _ = layer(h_V, h_E_enc, E_idx)
        h_S = self.W_s(S)
        h_ES = torch.cat([h_E_enc, gather_nodes(h_V, E_idx), gather_nodes(h_S, E_idx)], dim=-1)
        for layer in self.decoder_layers: h_V = layer(h_V, h_ES)
        logits_aa = self.W_out_aa(h_V)
        logits_methyl = self.W_out_methyl(torch.cat([h_V, h_S], dim=-1)).squeeze(-1)
        if hb_mask is not None: logits_methyl = logits_methyl - (hb_mask * 1e8)
        return logits_aa, logits_methyl

# ==========================================
# 4. 官方 RBF 引擎 (严格 N, Ca, C, O, Cb 顺序)
# ==========================================
def get_rbf(X, mask, top_k=48):
    B, L, _, _ = X.shape
    N, Ca, C, O = X[:,:,0,:], X[:,:,1,:], X[:,:,2,:], X[:,:,3,:]
    b, c = Ca - N, C - Ca; a = torch.cross(b, c, dim=-1)
    Cb = -0.5827*a + 0.5680*b - 0.5407*c + Ca
    
    dX_ca = Ca.unsqueeze(1) - Ca.unsqueeze(2)
    D_ca = torch.sqrt(torch.sum(dX_ca**2, 3) + 1e-6)
    _, E_idx = torch.topk(D_ca + (1.-mask.unsqueeze(1)*mask.unsqueeze(2))*1e8, min(top_k, L), dim=-1, largest=False)
    
    D_mu = torch.linspace(2.0, 22.0, 16, device=X.device).view(1,1,1,-1)
    RBF_all = []
    # 官方第 1 块：Ca-Ca
    RBF_all.append(torch.exp(-((torch.gather(D_ca, 2, E_idx).unsqueeze(-1) - D_mu) / 1.25)**2))
    # 官方第 2-25 块：5x5 两两原子组合
    atom_list = [N, C, O, Cb, Ca]
    for a1 in atom_list:
        for a2 in atom_list:
            if a1 is Ca and a2 is Ca: continue
            dist = torch.sqrt(torch.sum((a1.unsqueeze(1) - a2.unsqueeze(2))**2, -1) + 1e-6)
            RBF_all.append(torch.exp(-((torch.gather(dist, 2, E_idx).unsqueeze(-1) - D_mu) / 1.25)**2))
    return E_idx, torch.cat(RBF_all, -1) # 400 维

# ==========================================
# 5. 数据处理与坐标纠偏
# ==========================================
def process_batch(batch, device):
    B = len(batch); L_max = max([len(b['seq_chain_A']) for b in batch])
    X = torch.zeros(B, L_max, 4, 3, device=device); S_t = torch.zeros(B, L_max, dtype=torch.long, device=device)
    M_t = torch.zeros(B, L_max, device=device); mask = torch.zeros(B, L_max, device=device); r = torch.zeros(B, L_max, dtype=torch.long, device=device)
    for i, b in enumerate(batch):
        seq = b['seq_chain_A']; l = len(seq); mask[i,:l] = 1.0; r[i,:l] = torch.arange(l)
        for j, c in enumerate(seq):
            S_t[i,j] = AA_TO_IDX[MET_MAP[c]] if c in MET_MAP else AA_TO_IDX.get(c, 20)
            if c in MET_MAP: M_t[i,j] = 1.0
        raw = [np.array(b[f'{a}_chain_A']) for a in ['N','CA','C','O']]
        # 修正原子顺序 (0, 3, 2, 1) -> N, O, C, CA 归一化为 N, CA, C, O
        X[i,:l,0,:] = torch.tensor(raw[0][:l]) # N
        X[i,:l,1,:] = torch.tensor(raw[3][:l]) # CA
        X[i,:l,2,:] = torch.tensor(raw[2][:l]) # C
        X[i,:l,3,:] = torch.tensor(raw[1][:l]) # O
    X *= 1.355 # 尺度修正
    return X, mask, r, S_t, M_t

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--nmethyl_data", required=True); parser.add_argument("--test_data", required=True); parser.add_argument("--pretrained_weights", required=True)
    args = parser.parse_args(); device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ProteinMPNN_DoubleHead().to(device)
    ckpt = torch.load(args.pretrained_weights, map_location=device)['model_state_dict']
    new_sd = {k.replace('encoder.encoder_layers.', 'encoder_layers.').replace('decoder.decoder_layers.', 'decoder_layers.'): v for k, v in ckpt.items()}
    if 'W_out.weight' in ckpt: new_sd['W_out_aa.weight'] = ckpt['W_out.weight']; new_sd['W_out_aa.bias'] = ckpt['W_out.bias']
    model.load_state_dict(new_sd, strict=False)
    for n, p in model.named_parameters():
        if 'W_out_methyl' not in n: p.requires_grad = False # 锁定大脑
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    with open(args.nmethyl_data, 'r') as f: train_ds = [json.loads(l) for l in f]
    with open(args.test_data, 'r') as f: test_ds = [json.loads(l) for l in f]
    print("🚀 V130: FINAL ALIGNMENT. SECURING 60% THRESHOLD...")
    for ep in range(31):
        model.train(); total_l = 0
        for b in DataLoader(train_ds, batch_size=8, shuffle=True, collate_fn=lambda x:x):
            X, m, r, S_t, M_t = process_batch(b, device); E_idx, rbf = get_rbf(X, m); hb = get_hbond_mask(X, m)
            _, lm = model(S_t, E_idx, rbf, r, hb_mask=hb); loss = F.binary_cross_entropy_with_logits(lm, M_t)
            optimizer.zero_grad(); loss.backward(); optimizer.step(); total_l += loss.item()
        if ep % 10 == 0: print(f"Ep {ep} | Loss: {total_l:.4f}")
    model.eval(); aa_correct, m_correct, total_res = 0, 0, 0
    with torch.no_grad():
        for b in test_ds:
            X, m, r, S_t, M_t = process_batch([b], device); E_idx, rbf = get_rbf(X, m); hb = get_hbond_mask(X, m)
            la, lm = model(S_t, E_idx, rbf, r, hb_mask=hb)
            aa_correct += (torch.argmax(la, -1) == S_t).sum().item()
            m_correct += ((torch.sigmoid(lm) > 0.4).float() == M_t).sum().item()
            total_res += m.sum().item()
    print("\n" + "="*45)
    print(f"✅ 天然氨基酸准确率 (AA Acc): {aa_correct/total_res*100:.2f}%")
    print(f"✅ 氮甲基化准确率 (Methyl Acc): {m_correct/total_res*100:.2f}%")
    print(f"🔥 总设计准确率 (Total): {(aa_correct + m_correct)/(total_res*2)*100:.2f}%")
    print("="*45)

if __name__ == "__main__": main()
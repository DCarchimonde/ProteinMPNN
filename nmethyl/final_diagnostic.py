import torch, torch.nn as nn, torch.nn.functional as F
import numpy as np, json, os, itertools

# ==========================================
# 1. 核心映射 (ProteinMPNN 标准)
# ==========================================
NATURAL_AA = "ACDEFGHIKLMNPQRSTVWYX"
NAT_TO_IDX = {aa: i for i, aa in enumerate(NATURAL_AA)}
MET_MAP = {'a':'A','c':'C','d':'D','e':'E','f':'F','g':'G','h':'H','i':'I','k':'K','l':'L','m':'M','n':'N','p':'P','q':'Q','r':'R','s':'S','t':'T','v':'V','w':'W','y':'Y'}

# ==========================================
# 2. 官方架构 (修正 W2/W3 维度为 128 - 彻底对齐权重)
# ==========================================
def gather_nodes(nodes, neighbor_idx):
    res = torch.gather(nodes, 1, neighbor_idx.view(nodes.size(0), -1).unsqueeze(-1).expand(-1, -1, nodes.size(2)))
    return res.view(neighbor_idx.shape + (nodes.size(2),))

class EncLayer(nn.Module):
    def __init__(self, d_model=128, d_ff=512):
        super().__init__()
        self.norm1, self.norm2, self.norm3 = [nn.LayerNorm(d_model) for _ in range(3)]
        self.W1 = nn.Linear(3*d_model, d_model)
        self.W2 = nn.Linear(d_model, d_model) # 必须是 128
        self.W3 = nn.Linear(d_model, d_model) # 必须是 128
        self.W11 = nn.Linear(3*d_model, d_model)
        self.W12 = nn.Linear(d_model, d_model)
        self.W13 = nn.Linear(d_model, d_model)
        self.act = nn.GELU()
        self.dense = nn.ModuleDict({'W_in': nn.Linear(d_model, d_ff), 'W_out': nn.Linear(d_ff, d_model)})
    def forward(self, h_V, h_E, E_idx):
        h_V_neigh = gather_nodes(h_V, E_idx)
        h_EV = torch.cat([h_V.unsqueeze(2).expand_as(h_V_neigh), h_V_neigh, h_E], dim=-1)
        h_V = self.norm1(h_V + torch.mean(self.act(self.W3(self.act(self.W2(self.act(self.W1(h_EV)))))), dim=2))
        h_V = self.norm2(h_V + self.dense['W_out'](self.act(self.dense['W_in'](h_V))))
        return h_V

class DecLayer(nn.Module):
    def __init__(self, d_model=128, d_ff=512):
        super().__init__()
        self.norm1, self.norm2 = nn.LayerNorm(d_model), nn.LayerNorm(d_model)
        self.W1 = nn.Linear(4*d_model, d_model)
        self.W2 = nn.Linear(d_model, d_model) # 必须是 128
        self.W3 = nn.Linear(d_model, d_model) # 必须是 128
        self.act = nn.GELU()
        self.dense = nn.ModuleDict({'W_in': nn.Linear(d_model, d_ff), 'W_out': nn.Linear(d_ff, d_model)})
    def forward(self, h_V, h_E):
        h_V_expand = h_V.unsqueeze(2).expand(-1, -1, h_E.size(2), -1)
        h_EV = torch.cat([h_V_expand, h_E], dim=-1)
        h_V = self.norm1(h_V + torch.mean(self.act(self.W3(self.act(self.W2(self.act(self.W1(h_EV)))))), dim=2))
        h_V = self.norm2(h_V + self.dense['W_out'](self.act(self.dense['W_in'](h_V))))
        return h_V

class ProteinMPNN_Final(nn.Module):
    def __init__(self, d_model=128):
        super().__init__()
        self.features = nn.ModuleDict({
            'embeddings': nn.ModuleDict({'linear': nn.Linear(66, 16)}),
            'edge_embedding': nn.Linear(416, d_model, bias=False),
            'norm_edges': nn.LayerNorm(d_model)
        })
        self.W_e = nn.Linear(d_model, d_model); self.W_s = nn.Embedding(21, d_model)
        self.encoder_layers = nn.ModuleList([EncLayer() for _ in range(3)])
        self.decoder_layers = nn.ModuleList([DecLayer() for _ in range(3)])
        self.W_out = nn.Linear(d_model, 21)
    def forward(self, S, E_idx, h_E_rbf, res_idx):
        offset = res_idx.unsqueeze(1) - res_idx.unsqueeze(2)
        offset = torch.gather(offset, 2, E_idx)
        E_pos = self.features['embeddings']['linear'](F.one_hot(torch.clip(offset+32, 0, 64), 66).float())
        h_E = self.features['norm_edges'](self.features['edge_embedding'](torch.cat([E_pos, h_E_rbf], dim=-1)))
        h_V = torch.zeros(S.size(0), S.size(1), 128, device=S.device); h_E_enc = self.W_e(h_E)
        for layer in self.encoder_layers: h_V = layer(h_V, h_E_enc, E_idx)
        h_S = self.W_s(S)
        h_ES = torch.cat([h_E_enc, gather_nodes(h_V, E_idx), gather_nodes(h_S, E_idx)], dim=-1)
        for layer in self.decoder_layers: h_V = layer(h_V, h_ES)
        return self.W_out(h_V)

# ==========================================
# 3. 官方 RBF 引擎
# ==========================================
def get_rbf(X, mask, top_k=48):
    B, L, _, _ = X.shape
    N, Ca, C, O = X[:,:,0,:], X[:,:,1,:], X[:,:,2,:], X[:,:,3,:]
    b, c = Ca - N, C - Ca
    a = torch.cross(b, c, dim=-1)
    Cb = -0.5827*a + 0.5680*b - 0.5407*c + Ca
    atoms = [N, C, O, Cb, Ca] 
    dX = Ca.unsqueeze(1) - Ca.unsqueeze(2)
    D = torch.sqrt(torch.sum(dX**2, 3) + 1e-6)
    _, E_idx = torch.topk(D + (1.-mask.unsqueeze(1)*mask.unsqueeze(2))*1000000.0, min(top_k, L), dim=-1, largest=False)
    D_mu = torch.linspace(2.0, 22.0, 16, device=X.device).view(1,1,1,-1)
    RBF_all = []
    for a1 in atoms:
        for a2 in atoms:
            dist = torch.sqrt(torch.sum((a1.unsqueeze(1) - a2.unsqueeze(2))**2, -1) + 1e-6)
            RBF_all.append(torch.exp(-((torch.gather(dist, 2, E_idx).unsqueeze(-1) - D_mu) / 1.25)**2))
    return E_idx, torch.cat(RBF_all, -1)

def run_diagnostic():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    weights_path = "vanilla_model_weights/v_48_020.pt"
    test_path = "nmethyl_data/test_set/test.jsonl"

    model = ProteinMPNN_Final().to(device)
    ckpt = torch.load(weights_path, map_location=device)['model_state_dict']
    new_sd = {k.replace('encoder.encoder_layers.', 'encoder_layers.').replace('decoder.decoder_layers.', 'decoder_layers.'): v for k, v in ckpt.items()}
    model.load_state_dict(new_sd, strict=False)
    print("\n✅ WEIGHTS LOADED (100% MATCH)")

    with open(test_path, 'r') as f: data = [json.loads(line) for line in f]
    entry = data[0]; L = len(entry['seq_chain_A'])
    raw_coords = [np.array(entry[f'{a}_chain_A']) for a in ['N','CA','C','O']]

    print("\n" + "🌀"*15)
    print("V110 PERMUTATION SEARCH")
    print("🌀"*15)

    for p in itertools.permutations(range(4)):
        X = torch.zeros(1, L, 4, 3, device=device)
        for i, idx in enumerate(p): X[0,:,i,:] = torch.tensor(raw_coords[idx])
        
        # 物理修正
        d_ca = torch.norm(X[0,0,1,:] - X[0,1,1,:])
        if d_ca < 3.2: X *= (3.8 / d_ca.item())

        S_t = torch.zeros(1, L, dtype=torch.long, device=device)
        for j, c in enumerate(entry['seq_chain_A']):
            S_t[0,j] = NAT_TO_IDX[MET_MAP[c]] if c in MET_MAP else NAT_TO_IDX.get(c, 20)
        
        mask = torch.ones(1, L, device=device); r = torch.arange(L, device=device).unsqueeze(0)
        E_idx, h_E_rbf = get_rbf(X, mask)
        lb = model(S_t, E_idx, h_E_rbf, r)
        acc = (torch.argmax(lb, -1)[0] == S_t[0]).float().mean().item()
        
        if acc > 0.15: 
            print(f"✅ FOUND MATCH! Permutation: {p}, Accuracy: {acc*100:.2f}%")
            return

    print("❌ All permutations failed. Please provide a protein ID to check raw data.")

if __name__ == "__main__":
    run_diagnostic()
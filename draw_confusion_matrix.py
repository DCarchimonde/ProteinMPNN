import argparse
import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader, Dataset
import warnings

warnings.filterwarnings("ignore")
os.environ["OMP_NUM_THREADS"] = "1"

# =============================================================================
# SCI Q1 顶刊美学设置
# =============================================================================
sns.set_theme(style="ticks", context="paper", font_scale=1.5)
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans', 'sans-serif']
plt.rcParams['axes.linewidth'] = 2.0
plt.rcParams['savefig.dpi'] = 300

# 引入官方字典
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(current_dir)
sys.path.append(parent_dir)

try:
    from nmethyl.utils.nmethyl_config import (
        EXTENDED_AA_ALPHABET, NATURAL_AA_ALPHABET,
        NMETHYL_TO_NATURAL_MAPPING, EXTENDED_AA_TO_INDEX
    )
except ImportError as e:
    print(f"❌ ImportError: {e}")
    sys.exit(1)

# =============================================================================
# 核心架构 (仅需要专家头部分)
# =============================================================================
def gather_nodes(nodes, neighbor_idx):
    neighbors_flat = neighbor_idx.view((neighbor_idx.shape[0], -1))
    neighbors_flat = neighbors_flat.unsqueeze(-1).expand(-1, -1, nodes.size(2))
    return torch.gather(nodes, 1, neighbors_flat).view(list(neighbor_idx.shape)[:3] + [-1])

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

class Model_Decoupled_Expert(nn.Module):
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
        import torch.utils.checkpoint as checkpoint
        for layer in self.encoder_layers: h_V, h_E = checkpoint.checkpoint(layer, h_V, h_E, E_idx, mask, mask_attend)
        h_ES = cat_neighbors_nodes(self.W_s(S), h_E, E_idx)
        h_EXV_encoder = cat_neighbors_nodes(h_V, cat_neighbors_nodes(torch.zeros_like(self.W_s(S)), h_E, E_idx), E_idx)
        permutation_matrix_reverse = torch.nn.functional.one_hot(torch.argsort(chain_M*mask + 0.0001), num_classes=E_idx.shape[1]).float()
        order_mask_backward = torch.einsum('ij, biq, bjp->bqp', (1 - torch.triu(torch.ones(E_idx.shape[1], E_idx.shape[1], device=X.device))), permutation_matrix_reverse, permutation_matrix_reverse)
        mask_attend = torch.gather(order_mask_backward, 2, E_idx).unsqueeze(-1)
        mask_bw, mask_fw = mask.view([mask.size(0), mask.size(1), 1, 1]) * mask_attend, mask.view([mask.size(0), mask.size(1), 1, 1]) * (1. - mask_attend)
        for layer in self.decoder_layers: h_V = layer(h_V, mask_bw * cat_neighbors_nodes(h_V, h_ES, E_idx) + mask_fw * h_EXV_encoder, mask)
        
        logits_base = self.W_out_base(h_V)
        expert_outputs = [expert(h_V) for expert in self.experts]
        return logits_base, torch.cat(expert_outputs, dim=-1)

    def _rbf(self, D): return torch.exp(-((D.unsqueeze(-1) - torch.linspace(2., 22., 16, device=D.device)) / ((22.-2.)/16)) ** 2)

def bulletproof_load_weights(model, checkpoint_path, device):
    try: ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    except: ckpt = torch.load(checkpoint_path, map_location=device)
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
        if clean_k in model_state:
            if v.shape != model_state[clean_k].shape:
                new_v = model_state[clean_k].clone()
                slices = tuple(slice(0, min(dim_v, dim_m)) for dim_v, dim_m in zip(v.shape, new_v.shape))
                new_v[slices] = v[slices]
                new_state_dict[clean_k] = new_v
            else:
                new_state_dict[clean_k] = v
    model.load_state_dict(new_state_dict, strict=False)

# =============================================================================
# 数据处理
# =============================================================================
class JSONLDataset(Dataset):
    def __init__(self, jsonl_file): self.data = [json.loads(line) for line in open(jsonl_file, 'r')]
    def __len__(self): return len(self.data)
    def __getitem__(self, idx): return self.data[idx]

def collate_fn(batch): return batch

def featurize_batch(batch, device):
    B = len(batch)
    batch = [b for b in batch if 'seq_chain_A' in b and len(b['seq_chain_A']) > 0]
    if not batch: return None
    lengths = [sum([len(b.get(f'seq_chain_{c}', '')) for c in (b.get('masked_list', []) + b.get('visible_list', ['A']))]) for b in batch]
    L_max = max(lengths)
    if L_max == 0: return None
    X, S = np.zeros([B, L_max, 4, 3]), np.zeros([B, L_max], dtype=np.int32)
    mask, chain_M = np.zeros([B, L_max], dtype=np.float32), np.zeros([B, L_max], dtype=np.float32)
    residue_idx, chain_encoding_all = -100*np.ones([B, L_max], dtype=np.int32), np.zeros([B, L_max], dtype=np.int32)
    for i, b in enumerate(batch):
        all_chains = b.get('masked_list', []) + b.get('visible_list', [])
        if not all_chains: all_chains = ['A']
        l_p = 0
        for c_i, c_id in enumerate(all_chains):
            seq = b.get(f'seq_chain_{c_id}', '')
            N, CA, C, O = [np.array(b.get(f'{atom}_chain_{c_id}', [])) for atom in ['N', 'CA', 'C', 'O']]
            l = min(len(seq), len(CA))
            if l == 0: continue
            if len(N) > 0 and len(CA) > 0 and len(O) > 0 and np.linalg.norm(N[:1] - O[:1]) < np.linalg.norm(N[:1] - CA[:1]) and np.linalg.norm(N[:1] - O[:1]) < 1.6: CA, O = O, CA
            l = min(l, len(N), len(C), len(O))
            if l == 0: continue
            X[i, l_p:l_p+l, 0, :], X[i, l_p:l_p+l, 1, :], X[i, l_p:l_p+l, 2, :], X[i, l_p:l_p+l, 3, :] = N[:l], CA[:l], C[:l], O[:l]
            S[i, l_p:l_p+l] = [EXTENDED_AA_TO_INDEX.get(aa, 40) for aa in seq[:l]]
            if c_id in b.get('masked_list', []): chain_M[i, l_p:l_p+l] = 1.0
            residue_idx[i, l_p:l_p+l] = np.arange(l) + c_i * 100
            chain_encoding_all[i, l_p:l_p+l] = c_i
            l_p += l
    mask = np.isfinite(np.sum(X, (2,3))).astype(np.float32)
    X[np.isnan(X)] = 0.
    return [torch.from_numpy(t).to(dtype=torch.long if t.dtype==np.int32 else torch.float32, device=device) for t in [X, S, mask, chain_M, residue_idx, chain_encoding_all]]

# =============================================================================
# 提取混淆矩阵数据并画图
# =============================================================================
def draw_cm(model, dataloader, device, threshold):
    model.eval()
    all_targets_comb = []
    all_probs_experts_dec = []
    
    x_idx, offset = 40, 20

    print("🔍 正在计算测试集的真实预测结果...")
    with torch.no_grad():
        for batch in dataloader:
            f = featurize_batch(batch, device)
            if f is None: continue
            X, S, mask, chain_M, residue_idx, chain_encoding_all = f
            
            _, l_experts_dec = model(X, S, mask, chain_M, residue_idx, chain_encoding_all)
            
            true_base_idx = S.clone()
            for m_rel, n_idx in NMETHYL_TO_NATURAL_MAPPING.items():
                true_base_idx[true_base_idx == (m_rel + offset)] = n_idx
            true_base_idx[true_base_idx >= 20] = 0
            
            expert_logit_dec = torch.gather(l_experts_dec, -1, true_base_idx.unsqueeze(-1)).squeeze(-1)
            prob_methyl_dec = torch.sigmoid(expert_logit_dec)
            
            tgts, p_methyl_dec = S.cpu().numpy().flatten(), prob_methyl_dec.cpu().numpy().flatten()
            mask_flat = mask.cpu().numpy().flatten().astype(bool)
            
            valid = mask_flat & (tgts != x_idx)
            all_targets_comb.extend(tgts[valid])
            all_probs_experts_dec.extend(p_methyl_dec[valid])

    all_targets_comb = np.array(all_targets_comb)
    all_probs_experts_dec = np.array(all_probs_experts_dec)
    
    true_labels = (all_targets_comb >= offset).astype(int)
    pred_labels = (all_probs_experts_dec > threshold).astype(int)
    
    # 获取混淆矩阵
    cm = confusion_matrix(true_labels, pred_labels)
    cm_perc = cm / cm.sum() * 100
    
    # 构建包含数字和百分比的标签
    labels = np.asarray([f"{count}\n({perc:.1f}%)" for count, perc in zip(cm.flatten(), cm_perc.flatten())]).reshape(2, 2)
    
    print("🎨 正在绘制高逼格混淆矩阵图...")
    plt.figure(figsize=(7, 6))
    
    # 使用高级紫/蓝色系，符合 AIDD 审美
    ax = sns.heatmap(cm, annot=labels, fmt='', cmap='Purples', cbar=False,
                     xticklabels=['Canonical', 'Methylated'],
                     yticklabels=['Canonical', 'Methylated'],
                     annot_kws={"size": 18, "weight": "bold"})
    
    # 细节修饰
    plt.title('Expert Module Confusion Matrix\n(Threshold = 0.60)', fontweight='bold', fontsize=18, pad=15)
    plt.xlabel('Predicted Label', fontweight='bold', fontsize=16)
    plt.ylabel('True Label', fontweight='bold', fontsize=16)
    plt.xticks(fontsize=14, fontweight='bold')
    plt.yticks(fontsize=14, fontweight='bold', rotation=0)
    
    # 加上边框
    for _, spine in ax.spines.items():
        spine.set_visible(True)
        spine.set_linewidth(2)

    plt.tight_layout()
    plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✅ 混淆矩阵 confusion_matrix.png 绘制完成！")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--test_monomers", type=str, required=True)
    parser.add_argument("--threshold", type=float, default=0.60)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Model_Decoupled_Expert(augment_eps=0.0).to(device)
    bulletproof_load_weights(model, args.model_path, device)
    
    monomer_loader = DataLoader(JSONLDataset(args.test_monomers), batch_size=8, shuffle=False, collate_fn=collate_fn)
    draw_cm(model, monomer_loader, device, args.threshold)

if __name__ == "__main__":
    main()
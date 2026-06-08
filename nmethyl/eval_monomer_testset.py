import os
import sys
os.environ["OMP_NUM_THREADS"] = "1"
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import json
import numpy as np
import pandas as pd  # ✨ 新增：用于直接导出精美表格
from torch.utils.data import Dataset, DataLoader 

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
        nmethyl_idx = EXTENDED_AA_TO_INDEX[lower_aa]
        natural_idx = EXTENDED_AA_TO_INDEX[aa]
        NMETHYL_TO_NATURAL_MAPPING[nmethyl_idx - 20] = natural_idx 

# =============================================================================
# 2. 模型结构 (RobustHierarchicalProteinMPNN)
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
# 3. 加载与数据处理
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
        if clean_k in model_state:
            if v.shape != model_state[clean_k].shape:
                new_v = model_state[clean_k].clone()
                slices = tuple(slice(0, min(dim_v, dim_m)) for dim_v, dim_m in zip(v.shape, new_v.shape))
                new_v[slices] = v[slices]
                new_state_dict[clean_k] = new_v
            else:
                new_state_dict[clean_k] = v
    model.load_state_dict(new_state_dict, strict=False)
    print("✅ 权重加载成功！")

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

def eval_binary_classifier(model, loader, device, thresholds):
    model.eval()
    print("🧪 正在评估甲基化探测器...")
    true_labels, pred_probs = [], []
    offset, x_idx = len(NATURAL_AA_ALPHABET), EXTENDED_AA_TO_INDEX.get('X', 40)

    with torch.no_grad():
        for batch in loader:
            f = featurize_batch(batch, device)
            if f is None: continue
            X, S, mask, chain_M, residue_idx, chain_encoding_all = f
            l_base, l_experts = model(X, S, mask, chain_M, residue_idx, chain_encoding_all)
            
            true_base_idx = S.clone()
            for m_rel, n_idx in NMETHYL_TO_NATURAL_MAPPING.items():
                true_base_idx[true_base_idx == (m_rel + offset)] = n_idx
            true_base_idx[true_base_idx >= 20] = 0
            
            expert_logit = torch.gather(l_experts, -1, true_base_idx.unsqueeze(-1)).squeeze(-1)
            prob_methyl = torch.sigmoid(expert_logit)
            
            tgts = S.cpu().numpy().flatten()
            p_methyl = prob_methyl.cpu().numpy().flatten()
            mask_flat = mask.cpu().numpy().flatten().astype(bool)
            valid = mask_flat & (tgts != x_idx) 
            
            true_labels.extend((tgts[valid] >= offset).astype(int))
            pred_probs.extend(p_methyl[valid])

    true_labels, pred_probs = np.array(true_labels), np.array(pred_probs)
    print("\n" + "="*95)
    print(f"{'Threshold':<10} | {'Binary Accuracy':<18} | {'Precision (精准率)':<18} | {'Recall (召回率)':<18} | {'F1-Score':<10}")
    print("-" * 95)
    for thresh in thresholds:
        pred_labels = (pred_probs > thresh).astype(int)
        TP = np.sum((true_labels == 1) & (pred_labels == 1))
        TN = np.sum((true_labels == 0) & (pred_labels == 0))
        FP = np.sum((true_labels == 0) & (pred_labels == 1))
        FN = np.sum((true_labels == 1) & (pred_labels == 0))
        acc = (TP + TN) / (TP + TN + FP + FN) * 100 if (TP + TN + FP + FN) > 0 else 0
        precision = TP / (TP + FP) * 100 if (TP + FP) > 0 else 0
        recall = TP / (TP + FN) * 100 if (TP + FN) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        print(f"Thr = {thresh:<4.2f} | {acc:>14.2f}%   | {precision:>14.2f}%   | {recall:>14.2f}%   | {f1:>6.2f}")
    print("="*95)

def save_pdb_backbone(coords, seq, out_path):
    THREE_LETTER = {
        'A':'ALA', 'C':'CYS', 'D':'ASP', 'E':'GLU', 'F':'PHE',
        'G':'GLY', 'H':'HIS', 'I':'ILE', 'K':'LYS', 'L':'LEU',
        'M':'MET', 'N':'ASN', 'P':'PRO', 'Q':'GLN', 'R':'ARG',
        'S':'SER', 'T':'THR', 'V':'VAL', 'W':'TRP', 'Y':'TYR',
        'a':'ALA', 'c':'CYS', 'd':'ASP', 'e':'GLU', 'f':'PHE',
        'g':'GLY', 'h':'HIS', 'i':'ILE', 'k':'LYS', 'l':'LEU',
        'm':'MET', 'n':'ASN', 'p':'PRO', 'q':'GLN', 'r':'ARG',
        's':'SER', 't':'THR', 'v':'VAL', 'w':'TRP', 'y':'TYR', 'X':'UNK'
    }
    with open(out_path, 'w') as f:
        atom_idx = 1
        for i, (res, c) in enumerate(zip(seq, coords)):
            res_name = THREE_LETTER.get(res, 'UNK')
            f.write(f"ATOM  {atom_idx:5d}  N   {res_name} A{i+1:4d}    {c[0,0]:8.3f}{c[0,1]:8.3f}{c[0,2]:8.3f}  1.00 50.00           N  \n"); atom_idx+=1
            f.write(f"ATOM  {atom_idx:5d}  CA  {res_name} A{i+1:4d}    {c[1,0]:8.3f}{c[1,1]:8.3f}{c[1,2]:8.3f}  1.00 50.00           C  \n"); atom_idx+=1
            f.write(f"ATOM  {atom_idx:5d}  C   {res_name} A{i+1:4d}    {c[2,0]:8.3f}{c[2,1]:8.3f}{c[2,2]:8.3f}  1.00 50.00           C  \n"); atom_idx+=1
            f.write(f"ATOM  {atom_idx:5d}  O   {res_name} A{i+1:4d}    {c[3,0]:8.3f}{c[3,1]:8.3f}{c[3,2]:8.3f}  1.00 50.00           O  \n"); atom_idx+=1

# =============================================================================
# 5. ✨ 核心改装：基于0.6信任度生成并自动汇总 Master 文件 ✨
# =============================================================================
def generate_and_save_results(model, loader, device, threshold, output_dir):
    model.eval()
    os.makedirs(output_dir, exist_ok=True)
    print(f"\n🚀 严谨校准：正在以指定阈值 {threshold} 进行全量推理并实时拦截打包序列...")
    
    offset = len(NATURAL_AA_ALPHABET)
    x_idx = EXTENDED_AA_TO_INDEX.get('X', 40)

    # 用来存储给师兄的汇总数据
    master_fasta_lines = []
    master_csv_rows = []
    global_counter = 1

    with torch.no_grad():
        for batch in loader:
            f = featurize_batch(batch, device)
            if f is None: continue
            X, S, mask, chain_M, residue_idx, chain_encoding_all = f
            l_base, l_experts = model(X, S, mask, chain_M, residue_idx, chain_encoding_all)

            true_base_idx = S.clone()
            for m_rel, n_idx in NMETHYL_TO_NATURAL_MAPPING.items():
                true_base_idx[true_base_idx == (m_rel + offset)] = n_idx
            true_base_idx[true_base_idx >= 20] = 0

            expert_logit = torch.gather(l_experts, -1, true_base_idx.unsqueeze(-1)).squeeze(-1)
            prob_methyl = torch.sigmoid(expert_logit)

            pred_S_idx = true_base_idx.clone()
            methyl_mask = (prob_methyl > threshold) & (mask.bool()) & (S != x_idx)
            pred_S_idx[methyl_mask] = pred_S_idx[methyl_mask] + 20 # 完美加上小写甲基化！

            for i, b in enumerate(batch):
                name = b.get('name', f"protein_{np.random.randint(10000)}")
                seq_len = int(mask[i].sum().item())
                
                pred_indices = pred_S_idx[i, :seq_len].cpu().numpy()
                pred_seq_str = "".join([EXTENDED_AA_ALPHABET[idx] for idx in pred_indices])

                # (A) 保存原有的单体 FASTA
                fasta_path = os.path.join(output_dir, f"{name}.fa")
                with open(fasta_path, 'w') as fa_file:
                    fa_file.write(f">{name} | Predicted Threshold: {threshold}\n{pred_seq_str}\n")

                # (B) 保存原有的骨架 PDB
                pdb_path = os.path.join(output_dir, f"{name}_pred.pdb")
                coords = X[i, :seq_len].cpu().numpy()
                save_pdb_backbone(coords, pred_seq_str, pdb_path)
                
                # 🎯【核心拦截】：在这里悄悄加入发给师兄的 Master 汇总池
                master_fasta_lines.append(f">{name}\n{pred_seq_str}\n")
                master_csv_rows.append({
                    'No.': global_counter,
                    'Job_ID': name,
                    'Sequence_Length': seq_len,
                    'Sequence': pred_seq_str
                })
                global_counter += 1
                
    print(f"✅ 各个独立的 FASTA/PDB 文件已生成在目录：{output_dir}")

    # 💾 批量写入特供师兄的汇总大包
    fasta_out = "Sequences_for_AlphaFold3.fasta"
    csv_out = "Sequences_Summary_Table.csv"

    with open(fasta_out, "w", encoding="utf-8") as f_fa:
        f_fa.writelines(master_fasta_lines)

    df_summary = pd.DataFrame(master_csv_rows)
    df_summary.to_csv(csv_out, index=False, encoding="utf-8-sig")

    print(f"\n🎉【数据捕捉大成功！】已为您在当前目录下原地生成了特供包：")
    print(f"  1️⃣ 📦【Master FASTA 文件】: `{fasta_out}` (保留小写字母，供师兄跑预测)")
    print(f"  2️⃣ 📊【Excel可读表格】: `{csv_out}` (方便核对序列与长度)")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--test_data", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="./outputs_thr0.6")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = RobustHierarchicalProteinMPNN(augment_eps=0.0).to(device) 
    
    bulletproof_load_weights(model, args.model_path, device)
    test_ds = JSONLDataset(args.test_data)
    test_loader = DataLoader(test_ds, batch_size=8, shuffle=False, collate_fn=lambda x:x)

    thresholds_to_test = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.98, 0.99]
    eval_binary_classifier(model, test_loader, device, thresholds_to_test)
    
    # 强制以 0.6 信任度阈值拦截并提取最终序列
    generate_and_save_results(model, test_loader, device, threshold=0.6, output_dir=args.output_dir)

if __name__ == "__main__":
    main()
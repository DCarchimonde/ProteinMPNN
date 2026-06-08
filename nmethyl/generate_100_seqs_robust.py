import os
import sys
os.environ["OMP_NUM_THREADS"] = "1"
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import json
import numpy as np

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
        
        self.W_out_base = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, len(NATURAL_AA_ALPHABET))
        )
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
        
        # 移除 torch.utils.checkpoint.checkpoint 进行推理
        for layer in self.encoder_layers: h_V, h_E = layer(h_V, h_E, E_idx, mask, mask_attend)
        
        h_ES = cat_neighbors_nodes(self.W_s(S), h_E, E_idx)
        h_EXV_encoder = cat_neighbors_nodes(h_V, cat_neighbors_nodes(torch.zeros_like(self.W_s(S)), h_E, E_idx), E_idx)
        permutation_matrix_reverse = torch.nn.functional.one_hot(torch.argsort(chain_M*mask + 0.0001), num_classes=E_idx.shape[1]).float()
        order_mask_backward = torch.einsum('ij, biq, bjp->bqp', (1 - torch.triu(torch.ones(E_idx.shape[1], E_idx.shape[1], device=X.device))), permutation_matrix_reverse, permutation_matrix_reverse)
        mask_attend = torch.gather(order_mask_backward, 2, E_idx).unsqueeze(-1)
        mask_bw, mask_fw = mask.view([mask.size(0), mask.size(1), 1, 1]) * mask_attend, mask.view([mask.size(0), mask.size(1), 1, 1]) * (1. - mask_attend)
        
        for layer in self.decoder_layers: 
            h_ESV = cat_neighbors_nodes(h_V, h_ES, E_idx)
            h_ESV = mask_bw * h_ESV + mask_fw * h_EXV_encoder
            h_V = layer(h_V, h_ESV, mask)
        
        logits_base = self.W_out_base(h_V)
        logits_experts = torch.cat([expert(h_V) for expert in self.experts], dim=-1)
        return logits_base, logits_experts

    def _rbf(self, D): return torch.exp(-((D.unsqueeze(-1) - torch.linspace(2., 22., 16, device=D.device)) / ((22.-2.)/16)) ** 2)

# =============================================================================
# 3. 防弹权重加载器 (解决 40vs41 和 unexpected keys)
# =============================================================================
def bulletproof_load_weights(model, checkpoint_path, device):
    ckpt = torch.load(checkpoint_path, map_location=device)
    state_dict = ckpt.get('model_state_dict', ckpt)
    model_state = model.state_dict()
    new_state_dict = {}
    
    for k, v in state_dict.items():
        clean_k = k.replace('module.', '')
        if clean_k in model_state:
            # 解决词表大小不匹配 (40 vs 41)
            if v.shape != model_state[clean_k].shape:
                new_v = model_state[clean_k].clone()
                slices = tuple(slice(0, min(dim_v, dim_m)) for dim_v, dim_m in zip(v.shape, new_v.shape))
                new_v[slices] = v[slices]
                new_state_dict[clean_k] = new_v
            else:
                new_state_dict[clean_k] = v
                
    # strict=False 自动忽略冗余权重(如W11, W12等)
    model.load_state_dict(new_state_dict, strict=False)

# =============================================================================
# 4. 推理相关函数
# =============================================================================
def featurize_inference_complex(b, device):
    visible_list = b.get('visible_list', [])
    masked_list = b.get('masked_list', [])
    all_chains = masked_list + visible_list
    
    lengths = [len(b[f'seq_chain_{c}']) for c in all_chains]
    L_max = sum(lengths)
    
    X = np.zeros([1, L_max, 4, 3])
    S_true = np.zeros([1, L_max], dtype=np.int32)
    mask = np.ones([1, L_max], dtype=np.float32)
    chain_M = np.zeros([1, L_max], dtype=np.float32) 
    residue_idx = -100 * np.ones([1, L_max], dtype=np.int32)
    chain_encoding_all = np.zeros([1, L_max], dtype=np.int32)
    
    l_p = 0
    for c_i, c_id in enumerate(all_chains):
        seq = b[f'seq_chain_{c_id}']
        l = len(seq)
        
        N, CA, C, O = np.array(b[f'N_chain_{c_id}']), np.array(b[f'CA_chain_{c_id}']), np.array(b[f'C_chain_{c_id}']), np.array(b[f'O_chain_{c_id}'])
        
        if len(N) > 0 and len(CA) > 0 and len(O) > 0:
            dist_n_ca = np.linalg.norm(N[:1] - CA[:1])
            dist_n_o = np.linalg.norm(N[:1] - O[:1])
            if dist_n_o < dist_n_ca and dist_n_o < 1.6:
                CA, O = O, CA
                
        X[0, l_p:l_p+l, 0, :] = N[:l]
        X[0, l_p:l_p+l, 1, :] = CA[:l]
        X[0, l_p:l_p+l, 2, :] = C[:l]
        X[0, l_p:l_p+l, 3, :] = O[:l]
        
        indices = [EXTENDED_AA_TO_INDEX.get(aa, EXTENDED_AA_TO_INDEX['X']) for aa in seq]
        S_true[0, l_p:l_p+l] = indices
        
        if c_id in masked_list:
            chain_M[0, l_p:l_p+l] = 1.0 
            
        residue_idx[0, l_p:l_p+l] = np.arange(l) + c_i * 100
        chain_encoding_all[0, l_p:l_p+l] = c_i
        l_p += l

    tensors = [torch.from_numpy(t).to(dtype=torch.long if t.dtype==np.int32 else torch.float32, device=device) 
               for t in [X, S_true, mask, chain_M, residue_idx, chain_encoding_all]]
    return tensors

def generate_sequences(model, features, num_seqs=100, temperature=0.2, methyl_threshold=0.60):
    X, S_true, mask, chain_M, residue_idx, chain_encoding_all = features
    
    masked_positions = torch.nonzero(chain_M[0] == 1.0).squeeze(-1)
    if masked_positions.dim() == 0:
        masked_positions = masked_positions.unsqueeze(0)
        
    methyl_idx_to_nat_idx = {m + len(NATURAL_AA_ALPHABET): n for m, n in NMETHYL_TO_NATURAL_MAPPING.items()}
    nat_to_me_abs = {n: m for m, n in methyl_idx_to_nat_idx.items()}
    generated_seqs = []
    
    model.eval()
    with torch.no_grad():
        for i in range(num_seqs):
            S_pred = S_true.clone()
            S_pred[0, masked_positions] = EXTENDED_AA_TO_INDEX['X']
            
            decoding_order = masked_positions[torch.randperm(len(masked_positions))]
            
            for pos in decoding_order:
                l_base, l_experts = model(X, S_pred, mask, chain_M, residue_idx, chain_encoding_all)
                
                # 基础氨基酸预测带温度
                lb_pos = l_base[0, pos] / temperature
                prob_base = F.softmax(lb_pos, dim=-1)
                sampled_base = torch.multinomial(prob_base, 1).item()
                
                # 甲基化预测带温度
                expert_logit = l_experts[0, pos, sampled_base]
                prob_methyl = torch.sigmoid(expert_logit / temperature).item()
                sampled_methyl = 1 if prob_methyl > methyl_threshold else 0
                
                final_token = sampled_base
                if sampled_methyl == 1 and sampled_base in nat_to_me_abs:
                    final_token = nat_to_me_abs[sampled_base]
                    
                S_pred[0, pos] = final_token
                
            pep_indices = S_pred[0, masked_positions].cpu().numpy()
            pep_seq = "".join([EXTENDED_AA_ALPHABET[idx] for idx in pep_indices])
            generated_seqs.append(pep_seq)
            
    return generated_seqs

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--input_jsonl", type=str, default="./inference_complexes.jsonl")
    parser.add_argument("--output_dir", type=str, default="./generated_peptides")
    parser.add_argument("--num_seqs", type=int, default=100) 
    parser.add_argument("--temperature", type=float, default=0.2) 
    parser.add_argument("--methyl_threshold", type=float, default=0.60)
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 构建并使用防弹加载器加载模型！
    print(f"Loading Checkpoint: {args.model_path}")
    model = RobustHierarchicalProteinMPNN(augment_eps=0.0).to(device)
    bulletproof_load_weights(model, args.model_path, device)
    model.eval()
    
    print(f"\n📂 Reading from {args.input_jsonl}")
    with open(args.input_jsonl, 'r') as f:
        for line in f:
            b = json.loads(line)
            pdb_id = b['name']
            pep_chain = b['masked_list'][0]
            print(f"\n🧬 Designing for PDB: {pdb_id} (Target Chain: {pep_chain})")
            
            features = featurize_inference_complex(b, device)
            seqs = generate_sequences(model, features, num_seqs=args.num_seqs, temperature=args.temperature, methyl_threshold=args.methyl_threshold)
            
            unique_seqs = list(set(seqs))
            print(f"✅ Generated {args.num_seqs} sequences, found {len(unique_seqs)} unique designs.")
            
            out_fasta = os.path.join(args.output_dir, f"{pdb_id}_designs.fasta")
            with open(out_fasta, 'w') as out_f:
                for idx, seq in enumerate(unique_seqs):
                    out_f.write(f">{pdb_id}_design_{idx+1} | T={args.temperature} | Thr={args.methyl_threshold}\n{seq}\n")
                    
    print(f"\n🎉 完美收工！17个复合物的设计结果已全部保存在 '{args.output_dir}' 文件夹。")

if __name__ == "__main__":
    main()
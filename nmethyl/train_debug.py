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
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

# 兼容不同版本的 PyTorch AMP
try:
    from torch.amp import GradScaler, autocast
except ImportError:
    from torch.cuda.amp import GradScaler, autocast

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from model_utils import ProteinMPNN
    from nmethyl.utils.nmethyl_config import (
        EXTENDED_AA_ALPHABET, NATURAL_AA_ALPHABET,
        NMETHYL_TO_NATURAL_MAPPING, EXTENDED_AA_TO_INDEX
    )
except ImportError as e:
    print(f"Error: Missing project files. {e}")
    sys.exit(1)

# =============================================================================
# 1. 全兼容数据读取函数 (Auto-Detect Format)
# =============================================================================
def featurize_batch_robust(batch, device):
    alphabet = EXTENDED_AA_ALPHABET
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
            
            # --- [关键逻辑] 自动检测两种数据格式 ---
            # 尝试格式 1: 平铺 (N_chain_A)
            if f'N_chain_{c_id}' in b:
                coords_N = b[f'N_chain_{c_id}']
                coords_CA = b[f'CA_chain_{c_id}']
                coords_C = b[f'C_chain_{c_id}']
                coords_O = b[f'O_chain_{c_id}']
            # 尝试格式 2: 嵌套 (coords_chain_A['N']) -> 这是标准 MPNN 格式
            elif f'coords_chain_{c_id}' in b:
                coords_dict = b[f'coords_chain_{c_id}']
                # 有些 dict 键名可能是 'N_chain_A' 或简单的 'N'
                if 'N' in coords_dict:
                    coords_N = coords_dict['N']
                    coords_CA = coords_dict['CA']
                    coords_C = coords_dict['C']
                    coords_O = coords_dict['O']
                else:
                    # 最后的尝试，有些格式很怪
                    coords_N = coords_dict.get(f'N_chain_{c_id}', [])
                    coords_CA = coords_dict.get(f'CA_chain_{c_id}', [])
                    coords_C = coords_dict.get(f'C_chain_{c_id}', [])
                    coords_O = coords_dict.get(f'O_chain_{c_id}', [])
            else:
                print(f"[Warning] No coordinates found for chain {c_id} in sample {i}. Keys found: {list(b.keys())[:5]}...")
                coords_N = []
                coords_CA = []
                coords_C = []
                coords_O = []

            # 转换为 numpy
            coords_N = np.array(coords_N)
            coords_CA = np.array(coords_CA)
            coords_C = np.array(coords_C)
            coords_O = np.array(coords_O)
            
            l = len(seq)
            def safe_assign(source, atom_idx):
                if len(source) > 0:
                    v_len = min(l, len(source))
                    X[i, l_p:l_p+v_len, atom_idx, :] = source[:v_len]

            safe_assign(coords_N, 0)
            safe_assign(coords_CA, 1)
            safe_assign(coords_C, 2)
            safe_assign(coords_O, 3)
            
            indices = []
            for aa in seq[:l]:
                indices.append(EXTENDED_AA_TO_INDEX.get(aa, EXTENDED_AA_TO_INDEX['X']))
            S[i, l_p:l_p+l] = indices
            
            if c_id in b.get('masked_list', []): chain_M[i, l_p:l_p+l] = 1.0
            residue_idx[i, l_p:l_p+l] = np.arange(l) + c_i * 100
            chain_encoding_all[i, l_p:l_p+l] = c_i
            l_p += l
            
    has_N = np.isfinite(X[:, :, 0, 0])
    has_CA = np.isfinite(X[:, :, 1, 0])
    has_C = np.isfinite(X[:, :, 2, 0])
    mask = (has_N & has_CA & has_C).astype(np.float32)
    X[np.isnan(X)] = 0.0 
    
    return [torch.from_numpy(t).to(dtype=torch.long if t.dtype==np.int32 else torch.float32, device=device) for t in [X, S, mask, chain_M, residue_idx, chain_encoding_all]]

# =============================================================================
# 2. 简化的训练逻辑
# =============================================================================
class DecoupledProteinMPNN(ProteinMPNN):
    def __init__(self, hidden_dim=128, augment_eps=0.2, **kwargs):
        super().__init__(
            num_letters=21, node_features=128, edge_features=128, 
            hidden_dim=hidden_dim, vocab=21, k_neighbors=48, 
            augment_eps=augment_eps, **kwargs
        )
        self.W_s = nn.Embedding(len(EXTENDED_AA_ALPHABET), hidden_dim)
        self.W_out_base = nn.Linear(hidden_dim, len(NATURAL_AA_ALPHABET))
        self.W_out_methyl = nn.Linear(hidden_dim, 2)

    def forward(self, X, S, mask, chain_M, residue_idx, chain_encoding_all):
        E, E_idx = self.features(X, mask, residue_idx, chain_encoding_all)
        h_V = torch.zeros((E.shape[0], E.shape[1], E.shape[-1]), device=E.device)
        h_E = self.W_e(E)
        
        # 内置 gather
        def gather_nodes(nodes, neighbor_idx):
            neighbors_flat = neighbor_idx.view((neighbor_idx.shape[0], -1))
            neighbors_flat = neighbors_flat.unsqueeze(-1).expand(-1, -1, nodes.size(2))
            return torch.gather(nodes, 1, neighbors_flat).view(list(neighbor_idx.shape)[:3] + [-1])
        def cat_neighbors_nodes(h_nodes, h_neighbors, E_idx):
            h_nodes = gather_nodes(h_nodes, E_idx)
            return torch.cat([h_neighbors, h_nodes], -1)

        mask_attend = gather_nodes(mask.unsqueeze(-1),  E_idx).squeeze(-1)
        mask_attend = mask.unsqueeze(-1) * mask_attend
        for layer in self.encoder_layers:
            h_V, h_E = torch.utils.checkpoint.checkpoint(layer, h_V, h_E, E_idx, mask, mask_attend)

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
            h_V = torch.utils.checkpoint.checkpoint(layer, h_V, h_ESV, mask)
        
        logits_base = self.W_out_base(h_V)
        logits_methyl = self.W_out_methyl(h_V)
        return logits_base, logits_methyl

def calculate_loss(logits_base, logits_methyl, targets, mask, methyl_weight=5.0):
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
    for methyl_rel, natural_idx in NMETHYL_TO_NATURAL_MAPPING.items():
        base_targets[base_targets == (methyl_rel + offset)] = natural_idx
    base_targets[base_targets >= len(NATURAL_AA_ALPHABET)] = -100
    methyl_targets = (targets_flat >= len(NATURAL_AA_ALPHABET)).long()
    
    loss_base = F.cross_entropy(logits_base, base_targets, label_smoothing=0.1, ignore_index=-100)
    ce_loss_methyl = F.cross_entropy(logits_methyl, methyl_targets, reduction='none', weight=torch.tensor([1.0, 5.0], device=logits_base.device))
    loss_methyl = ((1 - torch.exp(-ce_loss_methyl)) ** 2.0 * ce_loss_methyl).mean()
    
    return loss_base + methyl_weight * loss_methyl, 1

# =============================================================================
# 3. 数据集与主函数
# =============================================================================
class JSONLDataset(Dataset):
    def __init__(self, jsonl_file):
        self.data = []
        with open(jsonl_file, 'r') as f:
            for line in f: self.data.append(json.loads(line))
    def __len__(self): return len(self.data)
    def __getitem__(self, idx): return self.data[idx]

def get_weighted_sampler(dataset): 
    weights = [1.0] * len(dataset)
    return WeightedRandomSampler(weights, len(weights), replacement=True)

def train_epoch(model, loader, optimizer, device, scaler, accum_steps):
    model.train()
    total_loss, steps = 0, 0
    optimizer.zero_grad()
    for i, batch in enumerate(loader):
        f = featurize_batch_robust(batch, device)
        if f is None: continue
        X, S, mask, chain_M, residue_idx, chain_encoding_all = f
        
        # --- [CRITICAL CHECK] ---
        if i == 0:
            nonzero_coords = torch.count_nonzero(X)
            print(f"[Sanity Check] Batch 0 - Total Non-zero Coords: {nonzero_coords.item()}")
            if nonzero_coords == 0:
                print("❌❌❌ ERROR: ALL COORDINATES ARE ZERO! CHECK YOUR JSON DATA KEYS! ❌❌❌")
                sys.exit(1)
            else:
                print("✅ Data looks good. Proceeding to train...")
        # ------------------------

        with autocast(device_type='cuda', dtype=torch.float16):
            lb, lm = model(X, S, mask, chain_M, residue_idx, chain_encoding_all)
            loss, valid = calculate_loss(lb, lm, S, mask)
            loss = loss / accum_steps
        
        if valid > 0:
            scaler.scale(loss).backward()
            if (i + 1) % accum_steps == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
            total_loss += loss.item() * accum_steps; steps += 1
    return total_loss / steps if steps > 0 else 0

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrained_weights", type=str, required=True)
    parser.add_argument("--nmethyl_data", type=str, required=True)
    parser.add_argument("--test_data", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="./run_nature_save")
    parser.add_argument("--epochs", type=int, default=100) # 先跑100轮
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.output_dir, exist_ok=True)
    
    model = DecoupledProteinMPNN(augment_eps=0.2).to(device)
    ckpt = torch.load(args.pretrained_weights, map_location=device)
    model.load_state_dict({k:v for k,v in ckpt.get('model_state_dict', ckpt).items() 
                           if k in model.state_dict() and v.shape == model.state_dict()[k].shape}, strict=False)
    
    # 移植 Embeddings
    if 'W_s.weight' in ckpt['model_state_dict']:
        old_ws = ckpt['model_state_dict']['W_s.weight']
        with torch.no_grad():
            model.W_s.weight.data[:min(len(old_ws), 21)] = old_ws[:min(len(old_ws), 21)]

    train_ds = JSONLDataset(args.nmethyl_data)
    loader = DataLoader(train_ds, batch_size=8, sampler=get_weighted_sampler(train_ds), collate_fn=lambda x: x)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    scaler = GradScaler()
    
    print(">>> Starting Training with Data Validation...")
    for epoch in range(1, args.epochs+1):
        loss = train_epoch(model, loader, optimizer, device, scaler, 4)
        print(f"Epoch {epoch}: Loss = {loss:.4f}")
        if epoch % 10 == 0:
            torch.save({'model_state_dict': model.state_dict()}, f"{args.output_dir}/model_epoch_{epoch}.pt")

if __name__ == "__main__":
    main()
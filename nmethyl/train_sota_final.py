import argparse
import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import json
import numpy as np
import random
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
try:
    from torch.amp import GradScaler, autocast
except ImportError:
    from torch.cuda.amp import GradScaler, autocast

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    # 必须从 model_utils 导入，确保用到刚才覆盖的 SOTA Features
    from model_utils import ProteinMPNN
    from nmethyl.utils.nmethyl_config import (
        EXTENDED_AA_ALPHABET, NATURAL_AA_ALPHABET,
        NMETHYL_TO_NATURAL_MAPPING, EXTENDED_AA_TO_INDEX
    )
except ImportError as e:
    print(f"Error: Missing project files. {e}")
    sys.exit(1)

# =============================================================================
# 1. 核心数据读取 (针对平铺格式 N_chain_A)
# =============================================================================
def featurize_batch_sota(batch, device):
    """
    [SOTA Data Pipeline]
    针对 head -n 1 看到的 N_chain_A 格式进行读取。
    处理 NaN 坐标以防止 Virtual Methyl 计算崩溃。
    """
    B = len(batch)
    batch = [b for b in batch if 'seq' in b and len(b['seq']) > 0]
    if not batch: return None
    lengths = [len(b['seq']) for b in batch]
    L_max = max(lengths)
    
    # 初始化为 NaN，以便后续生成 mask
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
            
            # --- [CRITICAL FIX] 直接读取 N_chain_A ---
            # 如果某个链的坐标不存在，get返回空列表
            coords_N = np.array(b.get(f'N_chain_{c_id}', []))
            coords_CA = np.array(b.get(f'CA_chain_{c_id}', []))
            coords_C = np.array(b.get(f'C_chain_{c_id}', []))
            coords_O = np.array(b.get(f'O_chain_{c_id}', []))
            
            l = len(seq)
            # 安全填充，防止坐标长度和序列长度不一致
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
            
    # [Mask Generation] 只要 CA 存在就算有效
    mask = np.isfinite(X[:, :, 1, 0]).astype(np.float32)
    
    # [NaN Handling] 
    # 将 NaN 替换为 0.0。
    # 注意：model_utils.py 里的计算已经加了 +1e-6，所以 0.0 不会引发除零错误。
    X[np.isnan(X)] = 0.0 
    
    return [torch.from_numpy(t).to(dtype=torch.long if t.dtype==np.int32 else torch.float32, device=device) for t in [X, S, mask, chain_M, residue_idx, chain_encoding_all]]

# =============================================================================
# 2. 模型与 Loss
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
# 3. 训练流程
# =============================================================================
class JSONLDataset(Dataset):
    def __init__(self, jsonl_file):
        self.data = []
        with open(jsonl_file, 'r') as f:
            for line in f: self.data.append(json.loads(line))
    def __len__(self): return len(self.data)
    def __getitem__(self, idx): return self.data[idx]

def get_weighted_sampler(dataset): 
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
        weights.append(10.0 if has_methyl else 1.0)
    return WeightedRandomSampler(weights, len(weights), replacement=True)

def train_epoch(model, loader, optimizer, device, scaler, accum_steps):
    model.train()
    total_loss, steps = 0, 0
    optimizer.zero_grad()
    for i, batch in enumerate(loader):
        f = featurize_batch_sota(batch, device)
        if f is None: continue
        X, S, mask, chain_M, residue_idx, chain_encoding_all = f
        
        # [Sanity Check] 确保数据不是空的
        if i == 0 and torch.count_nonzero(X) == 0:
            print("❌❌❌ ERROR: Data is ALL ZEROS. Stopping training.")
            sys.exit(1)

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
    parser.add_argument("--output_dir", type=str, default="./run_nature_sota")
    parser.add_argument("--epochs", type=int, default=200)
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.output_dir, exist_ok=True)
    
    model = DecoupledProteinMPNN(augment_eps=0.2).to(device)
    
    # 智能加载 (跳过不匹配的层)
    print(f"Loading weights from: {args.pretrained_weights}")
    ckpt = torch.load(args.pretrained_weights, map_location=device)
    pretrained_dict = ckpt.get('model_state_dict', ckpt)
    model_dict = model.state_dict()
    
    # 过滤掉形状不匹配的 (第一层 edge_embedding 和 最后一层 W_s)
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict and v.shape == model_dict[k].shape}
    model.load_state_dict(pretrained_dict, strict=False)
    print(f"✅ Loaded {len(pretrained_dict)} layers. SOTA architecture initialized.")

    # 移植 W_s (重要!)
    if 'W_s.weight' in ckpt.get('model_state_dict', ckpt):
        old_ws = ckpt.get('model_state_dict', ckpt)['W_s.weight']
        with torch.no_grad():
            model.W_s.weight.data[:min(len(old_ws), 21)] = old_ws[:min(len(old_ws), 21)]

    train_ds = JSONLDataset(args.nmethyl_data)
    loader = DataLoader(train_ds, batch_size=8, sampler=get_weighted_sampler(train_ds), collate_fn=lambda x: x)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    scaler = GradScaler()
    
    print(">>> Starting SOTA Training...")
    for epoch in range(1, args.epochs+1):
        loss = train_epoch(model, loader, optimizer, device, scaler, 4)
        if epoch % 5 == 0:
            print(f"Epoch {epoch}: Loss = {loss:.4f}")
        if epoch % 20 == 0:
            torch.save({'model_state_dict': model.state_dict()}, f"{args.output_dir}/best_model_sota.pt")

if __name__ == "__main__":
    main()
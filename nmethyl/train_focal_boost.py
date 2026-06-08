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
from sklearn.metrics import classification_report
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.utils.checkpoint 

# --- 系统路径 ---
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    print(f"[System] Random Seed locked to: {seed}")

try:
    from model_utils import ProteinMPNN, gather_nodes, cat_neighbors_nodes
    from nmethyl.utils.nmethyl_config import (
        EXTENDED_AA_ALPHABET, NATURAL_AA_ALPHABET,
        NMETHYL_TO_NATURAL_MAPPING, EXTENDED_AA_TO_INDEX
    )
except ImportError as e:
    print(f"Error: Missing project files. {e}")
    sys.exit(1)

# =============================================================================
# 1. 核心组件：Focal Loss
# =============================================================================
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.8, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        if self.reduction == 'mean':
            return focal_loss.mean()
        return focal_loss.sum()

# =============================================================================
# 2. 模型定义 (修复维度不匹配)
# =============================================================================
class DecoupledProteinMPNN(ProteinMPNN):
    def __init__(self, hidden_dim=128, augment_eps=0.2, **kwargs):
        # 初始 num_letters 设为 21 以匹配预训练权重
        super().__init__(
            num_letters=21, 
            hidden_dim=hidden_dim, 
            vocab=21, 
            k_neighbors=48, 
            augment_eps=augment_eps, 
            **kwargs
        )
        # 初始化基础层
        self.W_s = nn.Embedding(21, hidden_dim)
        self.W_out_base = nn.Linear(hidden_dim, len(NATURAL_AA_ALPHABET))
        self.W_out_methyl = nn.Linear(hidden_dim, 2)

    def extend_embedding_and_heads(self, new_vocab_size):
        """
        核心修复：扩充 Embedding 到 40 维，并让甲基化氨基酸初始化为对应的天然氨基酸权重
        """
        old_weight = self.W_s.weight.data # [21, 128]
        old_num, h_dim = old_weight.shape
        
        new_W_s = nn.Embedding(new_vocab_size, h_dim).to(old_weight.device)
        # 拷贝前21个天然氨基酸权重
        new_W_s.weight.data[:old_num, :] = old_weight
        
        # 让甲基化位点继承天然位点的权重（比随机初始化更快）
        offset = len(NATURAL_AA_ALPHABET)
        for m_rel, n_idx in NMETHYL_TO_NATURAL_MAPPING.items():
            m_abs_idx = m_rel + offset
            if m_abs_idx < new_vocab_size:
                new_W_s.weight.data[m_abs_idx, :] = old_weight[n_idx, :]
                
        self.W_s = new_W_s
        print(f"[Model] Embedding extended to {new_vocab_size}")

    def forward(self, X, S, mask, chain_M, residue_idx, chain_encoding_all):
        E, E_idx = self.features(X, mask, residue_idx, chain_encoding_all)
        h_V = torch.zeros((E.shape[0], E.shape[1], E.shape[-1]), device=E.device)
        h_E = self.W_e(E)

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

# =============================================================================
# 3. 损失计算
# =============================================================================
def calculate_aggressive_loss(logits_base, logits_methyl, targets, mask, methyl_weight=10.0):
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
    
    # Focal + Positive Weighting
    ce_loss_methyl = F.cross_entropy(logits_methyl, methyl_targets, reduction='none', weight=torch.tensor([1.0, 5.0], device=logits_base.device))
    pt = torch.exp(-ce_loss_methyl)
    loss_methyl_focal = ((1 - pt) ** 2.0) * ce_loss_methyl
    loss_methyl = loss_methyl_focal.mean()
    
    return loss_base + methyl_weight * loss_methyl, base_targets.numel()

# =============================================================================
# 4. 数据处理与 Featurize
# =============================================================================
class JSONLDataset(Dataset):
    def __init__(self, jsonl_file):
        self.data = []
        with open(jsonl_file, 'r') as f:
            for line in f: self.data.append(json.loads(line))
    def __len__(self): return len(self.data)
    def __getitem__(self, idx): return self.data[idx]

def get_weighted_sampler(dataset, oversample_weight=50.0):
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
        weights.append(oversample_weight if has_methyl else 1.0)
    return WeightedRandomSampler(weights, len(weights), replacement=True)

def featurize_batch(batch, device):
    B = len(batch)
    batch = [b for b in batch if 'seq' in b or any(k.startswith('seq_chain_') for k in b)]
    if not batch: return None
    
    # 获取最大长度
    lengths = []
    for b in batch:
        l = 0
        all_chains = b.get('masked_list', []) + b.get('visible_list', [])
        for c_id in all_chains:
            l += len(b.get(f'seq_chain_{c_id}', ''))
        lengths.append(l)
    L_max = max(lengths) if lengths else 1

    X = np.zeros([B, L_max, 4, 3])
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
            N = np.array(b.get(f'N_chain_{c_id}', []))
            CA = np.array(b.get(f'CA_chain_{c_id}', []))
            C = np.array(b.get(f'C_chain_{c_id}', []))
            O = np.array(b.get(f'O_chain_{c_id}', []))
            l = min(len(seq), len(CA))
            if l == 0: continue
            
            end = min(l_p + l, L_max)
            l_act = end - l_p
            X[i, l_p:end, 0, :] = N[:l_act]
            X[i, l_p:end, 1, :] = CA[:l_act]
            X[i, l_p:end, 2, :] = C[:l_act]
            X[i, l_p:end, 3, :] = O[:l_act]
            
            indices = [EXTENDED_AA_TO_INDEX.get(aa, EXTENDED_AA_TO_INDEX['X']) for aa in seq[:l_act]]
            S[i, l_p:end] = indices
            if c_id in b.get('masked_list', []): chain_M[i, l_p:end] = 1.0
            residue_idx[i, l_p:end] = np.arange(l_act) + c_i * 100
            chain_encoding_all[i, l_p:end] = c_i
            l_p += l_act

    isnan = np.isnan(X)
    mask = np.isfinite(np.sum(X, (2,3))).astype(np.float32)
    X[isnan] = 0.
    
    return [torch.from_numpy(t).to(dtype=torch.long if t.dtype==np.int32 else torch.float32, device=device) for t in [X, S, mask, chain_M, residue_idx, chain_encoding_all]]

def collate_fn(batch): return batch

# =============================================================================
# 5. 主训练流程
# =============================================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrained_weights", type=str, required=True)
    parser.add_argument("--nmethyl_data", type=str, required=True)
    parser.add_argument("--test_data", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="./run_focal_boost")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-4)
    args = parser.parse_args()
    
    set_seed(42)
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. 初始模型 (21维)
    model = DecoupledProteinMPNN(augment_eps=0.2).to(device)
    
    # 2. 加载预训练权重
    print(f"Loading weights from: {args.pretrained_weights}")
    ckpt = torch.load(args.pretrained_weights, map_location=device)
    state_dict = ckpt.get('model_state_dict', ckpt)
    model.load_state_dict(state_dict, strict=False) 

    # 3. 关键：扩充 Embedding 到扩展词表大小
    model.extend_embedding_and_heads(len(EXTENDED_AA_ALPHABET))

    train_ds = JSONLDataset(args.nmethyl_data)
    sampler = get_weighted_sampler(train_ds, oversample_weight=50.0)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, sampler=sampler, collate_fn=collate_fn)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

    print(">>> Starting Focal Loss Boost Training...")
    best_loss = float('inf')
    
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss, n_steps = 0, 0
        for batch in train_loader:
            f = featurize_batch(batch, device)
            if f is None: continue
            X, S, mask, chain_M, residue_idx, chain_encoding_all = f
            
            optimizer.zero_grad()
            lb, lm = model(X, S, mask, chain_M, residue_idx, chain_encoding_all)
            loss, valid = calculate_aggressive_loss(lb, lm, S, mask)
            
            if valid > 0 and torch.isfinite(loss):
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                total_loss += loss.item()
                n_steps += 1
        
        avg_loss = total_loss / n_steps if n_steps > 0 else 0
        scheduler.step(avg_loss)
        if epoch % 1 == 0:
            print(f"Epoch {epoch}/{args.epochs} | Loss: {avg_loss:.4f} | LR: {optimizer.param_groups[0]['lr']:.6e}")

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save({'model_state_dict': model.state_dict()}, os.path.join(args.output_dir, "best_model_focal.pt"))

    print(f">>> Training Complete. Best Loss: {best_loss:.4f}")

if __name__ == "__main__":
    main()
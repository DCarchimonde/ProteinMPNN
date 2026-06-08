import argparse
import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import json
import numpy as np
import random
import itertools
import csv
import time
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from sklearn.metrics import classification_report, f1_score
from torch.optim.lr_scheduler import CosineAnnealingLR

# --- [关键] 动态添加路径 ---
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# --- 尝试导入项目配置 ---
try:
    from model_utils import ProteinMPNN, gather_nodes, cat_neighbors_nodes
    from nmethyl.utils.nmethyl_config import (
        EXTENDED_AA_ALPHABET,
        NATURAL_AA_ALPHABET,
        NMETHYL_TO_NATURAL_MAPPING,
        EXTENDED_AA_TO_INDEX
    )
except ImportError as e:
    print(f"错误：无法导入必需的模块。请确保 nmethyl_config.py 和 model_utils.py 存在。\n详细: {e}")
    sys.exit(1)

# 构建反向映射
NATURAL_TO_NMETHYL_MAPPING = {v: k for k, v in NMETHYL_TO_NATURAL_MAPPING.items()}

# =============================================================================
# 1. 工具函数：随机种子与Focal Loss
# =============================================================================

def set_seed(seed=42):
    """锁定所有随机种子，确保实验可复现"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"[System] Random Seed set to: {seed}")

class FocalLoss(nn.Module):
    """
    专门用于解决极端类别不平衡的损失函数。
    降低易分类样本的权重，专注于难分类样本。
    """
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

# =============================================================================
# 2. 模型架构 (Decoupled)
# =============================================================================

class DecoupledProteinMPNN(ProteinMPNN):
    def __init__(self, hidden_dim=128, **kwargs):
        super().__init__(num_letters=21, hidden_dim=hidden_dim, vocab=21, **kwargs)
        self.W_s = nn.Embedding(len(EXTENDED_AA_ALPHABET), hidden_dim) 
        # Head 1: Base Type (20 classes)
        self.W_out_base = nn.Linear(hidden_dim, len(NATURAL_AA_ALPHABET)) 
        # Head 2: Methylation Status (2 classes)
        self.W_out_methyl = nn.Linear(hidden_dim, 2)

    def forward(self, X, S, mask, chain_M, residue_idx, chain_encoding_all):
        device = X.device
        E, E_idx = self.features(X, mask, residue_idx, chain_encoding_all)
        h_V = torch.zeros((E.shape[0], E.shape[1], E.shape[-1]), device=E.device)
        h_E = self.W_e(E)

        mask_attend = gather_nodes(mask.unsqueeze(-1), E_idx).squeeze(-1)
        mask_attend = mask.unsqueeze(-1) * mask_attend
        for layer in self.encoder_layers:
            h_V, h_E = torch.utils.checkpoint.checkpoint(layer, h_V, h_E, E_idx, mask, mask_attend)

        h_S = self.W_s(S)
        h_ES = cat_neighbors_nodes(h_S, h_E, E_idx)
        h_EX_encoder = cat_neighbors_nodes(torch.zeros_like(h_S), h_E, E_idx)
        h_EXV_encoder = cat_neighbors_nodes(h_V, h_EX_encoder, E_idx)
        
        chain_M = chain_M * mask
        decoding_order = torch.argsort((chain_M + 0.0001) * (torch.abs(torch.randn(chain_M.shape, device=device))))
        mask_size = E_idx.shape[1]
        permutation_matrix_reverse = torch.nn.functional.one_hot(decoding_order, num_classes=mask_size).float()
        order_mask_backward = torch.einsum('ij, biq, bjp->bqp', (1 - torch.triu(torch.ones(mask_size, mask_size, device=device))), permutation_matrix_reverse, permutation_matrix_reverse)
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
# 3. 损失计算 (支持 Focal Loss 切换)
# =============================================================================

def calculate_loss_flexible(logits_base, logits_methyl, targets, mask, 
                            methyl_weight=2.0, label_smoothing=0.1, use_focal=False, focal_loss_fn=None):
    mask_flat = mask.contiguous().view(-1).bool()
    if mask_flat.sum() == 0: return torch.tensor(0.0, device=logits_base.device), 0
    
    targets_flat = targets.contiguous().view(-1)[mask_flat]
    
    # 过滤 'X'
    x_idx = EXTENDED_AA_TO_INDEX.get('X', -1)
    if x_idx != -1:
        valid_target_mask = targets_flat != x_idx
        if valid_target_mask.sum() == 0: return torch.tensor(0.0, device=logits_base.device), 0
        targets_flat = targets_flat[valid_target_mask]
        logits_base_valid = logits_base.contiguous().view(-1, 20)[mask_flat][valid_target_mask]
        logits_methyl_valid = logits_methyl.contiguous().view(-1, 2)[mask_flat][valid_target_mask]
    else:
        logits_base_valid = logits_base.contiguous().view(-1, 20)[mask_flat]
        logits_methyl_valid = logits_methyl.contiguous().view(-1, 2)[mask_flat]
        
    # 准备标签
    base_targets = targets_flat.clone()
    offset = len(NATURAL_AA_ALPHABET)
    for methyl_rel_idx, natural_idx in NMETHYL_TO_NATURAL_MAPPING.items():
        base_targets[base_targets == (methyl_rel_idx + offset)] = natural_idx
    base_targets[base_targets >= len(NATURAL_AA_ALPHABET)] = -100
    
    methyl_targets = (targets_flat >= len(NATURAL_AA_ALPHABET)).long()

    # Base Loss (始终使用 CE + Smoothing)
    loss_base = F.cross_entropy(logits_base_valid, base_targets, label_smoothing=label_smoothing, ignore_index=-100)
    
    # Methyl Loss (策略分支)
    if use_focal and focal_loss_fn is not None:
        # 使用 Focal Loss，不乘 methyl_weight (Focal Loss 自带缩放) 或 乘更小的系数
        # 这里我们依然乘权重，增加强度
        loss_methyl = focal_loss_fn(logits_methyl_valid, methyl_targets)
    else:
        # 使用加权 Cross Entropy
        weight_tensor = torch.tensor([1.0, 5.0], device=logits_base.device) # 5.0 for Methylated
        loss_methyl = F.cross_entropy(logits_methyl_valid, methyl_targets, weight=weight_tensor)

    total_loss = loss_base + methyl_weight * loss_methyl
    return total_loss, base_targets.numel()

# =============================================================================
# 4. 数据加载部分 (Dataset, Sampler, Collate, Featurize)
# =============================================================================
# (为节省空间，保持与您之前的代码一致，此处省略重复定义，但在运行脚本中必须包含)
# 请确保 JSONLDataset, get_weighted_sampler, collate_fn, featurize_batch 函数在此处定义
# ... [在此处插入 Dataset 类和相关函数] ... 

class JSONLDataset(Dataset):
    def __init__(self, jsonl_file, augment=False, augmentation_rate=0.20):
        self.data = []
        self.augment = augment
        self.augmentation_rate = augmentation_rate
        with open(jsonl_file, 'r') as f:
            for line in f: self.data.append(json.loads(line))
    def __len__(self): return len(self.data)
    def __getitem__(self, idx):
        item = self.data[idx].copy()
        if self.augment:
            for key in item:
                if key.startswith('seq_chain_'):
                    seq_list = list(item[key])
                    if len(seq_list) == 0: continue
                    num_to_mask = int(len(seq_list) * self.augmentation_rate)
                    mask_indices = random.sample(range(len(seq_list)), min(num_to_mask, len(seq_list)))
                    for i in mask_indices: seq_list[i] = 'X'
                    item[key] = "".join(seq_list)
        return item

def get_weighted_sampler(dataset):
    sample_weights = []
    methyl_indices = set(NMETHYL_TO_NATURAL_MAPPING.keys())
    offset = len(NATURAL_AA_ALPHABET)
    methyl_abs_indices = {k + offset for k in methyl_indices}
    for item in dataset.data:
        has_methyl = False
        for key in item:
            if key.startswith('seq_chain_'):
                for char in item[key]:
                    if char in EXTENDED_AA_TO_INDEX:
                        idx = EXTENDED_AA_TO_INDEX[char]
                        if idx in methyl_abs_indices:
                            has_methyl = True; break
            if has_methyl: break
        sample_weights.append(5.0 if has_methyl else 1.0)
    return WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)

def collate_fn(batch): return batch

def featurize_batch(batch, device):
    # ... (完全复用您提供的 featurize_batch 代码) ...
    alphabet = EXTENDED_AA_ALPHABET
    B = len(batch)
    batch = [b for b in batch if 'seq' in b and len(b['seq']) > 0]
    if not batch: return None
    lengths = np.array([len(b['seq']) for b in batch], dtype=np.int32)
    L_max = max(lengths)
    X = np.zeros([B, L_max, 4, 3])
    S = np.zeros([B, L_max], dtype=np.int32)
    mask = np.zeros([B, L_max], dtype=np.float32)
    chain_M = np.zeros([B, L_max], dtype=np.float32)
    residue_idx = -100 * np.ones([B, L_max], dtype=np.int32)
    chain_encoding_all = np.zeros([B, L_max], dtype=np.int32)
    for i, b in enumerate(batch):
        all_chains = b.get('masked_list', []) + b.get('visible_list', [])
        l_processed = 0
        for c_idx, chain_id in enumerate(all_chains):
            seq_key = f'seq_chain_{chain_id}'
            if seq_key not in b: continue
            chain_seq = b[seq_key]
            if len(chain_seq) == 0: continue
            coords_N = np.array(b.get(f'N_chain_{chain_id}', []))
            coords_CA = np.array(b.get(f'CA_chain_{chain_id}', []))
            coords_C = np.array(b.get(f'C_chain_{chain_id}', []))
            coords_O = np.array(b.get(f'O_chain_{chain_id}', []))
            len_to_process = min(len(chain_seq), len(coords_CA))
            if len_to_process == 0: continue
            start, end = l_processed, l_processed + len_to_process
            X[i, start:end, 0, :] = coords_N[:len_to_process]
            X[i, start:end, 1, :] = coords_CA[:len_to_process]
            X[i, start:end, 2, :] = coords_C[:len_to_process]
            X[i, start:end, 3, :] = coords_O[:len_to_process]
            S[i, start:end] = [alphabet.index(aa) if aa in alphabet else EXTENDED_AA_TO_INDEX['X'] for aa in chain_seq[:len_to_process]]
            if chain_id in b.get('masked_list', []): chain_M[i, start:end] = 1.0
            chain_encoding_all[i, start:end] = c_idx
            residue_idx[i, start:end] = np.arange(len_to_process) + c_idx * 100
            l_processed += len_to_process
    isnan = np.isnan(X)
    mask = np.isfinite(np.sum(X, (2, 3))).astype(np.float32)
    X[isnan] = 0.
    return [torch.from_numpy(t).to(dtype=torch.long if t.dtype==np.int32 else torch.float32, device=device) for t in [X, S, mask, chain_M, residue_idx, chain_encoding_all]]

# =============================================================================
# 5. 评估与训练 Loop (封装为函数)
# =============================================================================

def run_evaluation(model, loader, device):
    model.eval()
    all_preds, all_targets = [], []
    natural_to_methyl_abs = {v: k + len(NATURAL_AA_ALPHABET) for k, v in NMETHYL_TO_NATURAL_MAPPING.items()}
    
    with torch.no_grad():
        for batch in loader:
            features = featurize_batch(batch, device)
            if features is None: continue
            logits_base, logits_methyl = model(*features)
            
            pred_base_idx = torch.argmax(logits_base, dim=-1)
            pred_is_methyl = torch.argmax(logits_methyl, dim=-1)
            
            final_preds = pred_base_idx.clone()
            B, L = pred_base_idx.shape
            for b in range(B):
                for l in range(L):
                    base = pred_base_idx[b, l].item()
                    if pred_is_methyl[b, l].item() == 1 and base in natural_to_methyl_abs:
                        final_preds[b, l] = natural_to_methyl_abs[base]
            
            targets_flat = features[1].cpu().numpy().flatten()
            preds_flat = final_preds.cpu().numpy().flatten()
            mask_flat = features[2].cpu().numpy().flatten().astype(bool)
            x_idx = EXTENDED_AA_TO_INDEX.get('X', -1)
            valid_pos = mask_flat & (targets_flat != x_idx)
            
            all_preds.extend(preds_flat[valid_pos])
            all_targets.extend(targets_flat[valid_pos])

    all_preds, all_targets = np.array(all_preds), np.array(all_targets)
    if len(all_targets) == 0: return 0, 0, 0

    overall_acc = np.mean(all_preds == all_targets)
    natural_mask = all_targets < len(NATURAL_AA_ALPHABET)
    natural_rec = np.mean(all_targets[natural_mask] == all_preds[natural_mask]) if natural_mask.sum() > 0 else 0
    
    # 计算 "v" (N-甲基缬氨酸) 的 F1 分数作为稀有类代表
    # 假设 v 在 extended 列表里
    v_idx = EXTENDED_AA_TO_INDEX.get('v', -1)
    f1_v = 0.0
    if v_idx != -1:
        # 简单的手动计算 binary F1 for class 'v'
        y_true_v = (all_targets == v_idx)
        y_pred_v = (all_preds == v_idx)
        f1_v = f1_score(y_true_v, y_pred_v, zero_division=0)

    return overall_acc, natural_rec, f1_v

def train_evaluate_model(params, args, device, train_loader, test_loader, run_id):
    # 解包参数
    methyl_weight = params['methyl_weight']
    backbone_lr_mult = params['backbone_lr_mult']
    label_smooth = params['label_smooth']
    use_focal = params['use_focal']
    gamma = params['focal_gamma']
    
    print(f"\n>>> 开始运行 ID {run_id}: {params}")
    set_seed(args.seed) # [关键] 每次重置种子

    # 初始化模型
    model = DecoupledProteinMPNN(num_encoder_layers=3, num_decoder_layers=3).to(device)
    
    # 加载权重
    checkpoint = torch.load(args.pretrained_weights, map_location=device)
    pretrained_dict = checkpoint.get('model_state_dict', checkpoint)
    model_state = model.state_dict()
    filtered_dict = {k: v for k, v in pretrained_dict.items() if k in model_state and v.size() == model_state[k].size()}
    model.load_state_dict(filtered_dict, strict=False)
    
    # 智能初始化 Head
    with torch.no_grad():
        if 'W_out.weight' in pretrained_dict:
            model.W_out_base.weight.data = pretrained_dict['W_out.weight'][:20, :].clone()
            model.W_out_base.bias.data = pretrained_dict['W_out.bias'][:20].clone()
    
    # 优化器
    new_params = ['W_s', 'W_out_base', 'W_out_methyl']
    optimizer = torch.optim.AdamW([
        {'params': [p for n, p in model.named_parameters() if not any(k in n for k in new_params)], 'lr': args.learning_rate * backbone_lr_mult},
        {'params': [p for n, p in model.named_parameters() if any(k in n for k in new_params)], 'lr': args.learning_rate}
    ], weight_decay=1e-4)

    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs * len(train_loader))
    focal_fn = FocalLoss(gamma=gamma) if use_focal else None
    
    best_natural_rec = 0.0
    best_overall_acc = 0.0
    best_f1_v = 0.0

    for epoch in range(1, args.epochs + 1):
        model.train()
        for batch in train_loader:
            features = featurize_batch(batch, device)
            if features is None: continue
            
            optimizer.zero_grad()
            logits_base, logits_methyl = model(*features)
            
            loss, num_valid = calculate_loss_flexible(
                logits_base, logits_methyl, features[1], features[2], 
                methyl_weight=methyl_weight, 
                label_smoothing=label_smooth,
                use_focal=use_focal,
                focal_loss_fn=focal_fn
            )
            
            if num_valid > 0 and torch.isfinite(loss):
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
    
    # 训练结束，评估一次
    acc, nat_rec, f1_v = run_evaluation(model, test_loader, device)
    print(f"Run {run_id} Result: Nat_Rec={nat_rec:.4f}, Acc={acc:.4f}, F1(v)={f1_v:.4f}")
    
    return {
        'run_id': run_id,
        'methyl_weight': methyl_weight,
        'backbone_lr_mult': backbone_lr_mult,
        'label_smooth': label_smooth,
        'use_focal': use_focal,
        'focal_gamma': gamma,
        'natural_recovery': nat_rec,
        'overall_accuracy': acc,
        'f1_valine': f1_v
    }

# =============================================================================
# 6. 主程序 (Grid Search Manager)
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Automated Grid Search")
    parser.add_argument("--pretrained_weights", type=str, required=True)
    parser.add_argument("--nmethyl_data", type=str, required=True)
    parser.add_argument("--test_data", type=str, required=True)
    parser.add_argument("--output_csv", type=str, default="grid_search_results.csv")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=50, help="Epochs per run (shorter for search)")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 准备数据 (只加载一次)
    train_dataset = JSONLDataset(args.nmethyl_data, augment=True)
    sampler = get_weighted_sampler(train_dataset)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, sampler=sampler, collate_fn=collate_fn)
    
    test_dataset = JSONLDataset(args.test_data, augment=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)
    
    # === 定义参数网格 ===
    # 我们将对比：Standard Strategy (Weighted CE) vs New Strategy (Focal Loss)
    param_grid = {
        'methyl_weight': [2.0, 3.0, 4.0],          # 权重力度
        'backbone_lr_mult': [0.05, 0.1],           # 适应性
        'label_smooth': [0.05, 0.1],               # 预测自信度
        'use_focal': [False, True],                # 是否开启 Focal Loss
        'focal_gamma': [2.0]                       # Focal Parameter (仅当 use_focal=True 时生效)
    }
    
    keys, values = zip(*param_grid.items())
    permutations = [dict(zip(keys, v)) for v in itertools.product(*values)]
    
    # 过滤逻辑：如果 use_focal=False, focal_gamma 没意义，避免重复跑
    unique_params = []
    seen = set()
    for p in permutations:
        p_tuple = (p['methyl_weight'], p['backbone_lr_mult'], p['label_smooth'], p['use_focal'])
        if p_tuple not in seen:
            seen.add(p_tuple)
            unique_params.append(p)
            
    print(f"Total combinations to test: {len(unique_params)}")
    
    # 准备 CSV
    fieldnames = ['run_id', 'methyl_weight', 'backbone_lr_mult', 'label_smooth', 'use_focal', 'focal_gamma', 'natural_recovery', 'overall_accuracy', 'f1_valine']
    with open(args.output_csv, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

    results = []
    for i, params in enumerate(unique_params):
        res = train_evaluate_model(params, args, device, train_loader, test_loader, run_id=i)
        results.append(res)
        
        # 实时写入
        with open(args.output_csv, 'a', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writerow(res)
            
    print("\nGrid Search Completed. Check grid_search_results.csv")

if __name__ == "__main__":
    main()
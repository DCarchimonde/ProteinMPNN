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
from sklearn.metrics import classification_report, confusion_matrix
from torch.optim.lr_scheduler import CosineAnnealingLR

# --- [系统设置] 动态路径与种子锁定 ---
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

# --- [核心组件 1] Focal Loss ---
class FocalLoss(nn.Module):
    """
    针对极度不平衡数据的焦点损失函数
    Gamma > 0 时，减少易分类样本的权重，强制模型关注'难'样本(N-甲基化)
    """
    def __init__(self, alpha=1, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        if self.reduction == 'mean': return focal_loss.mean()
        elif self.reduction == 'sum': return focal_loss.sum()
        return focal_loss

# --- [核心组件 2] Decoupled Architecture ---
class DecoupledProteinMPNN(ProteinMPNN):
    def __init__(self, hidden_dim=128, **kwargs):
        super().__init__(num_letters=21, hidden_dim=hidden_dim, vocab=21, **kwargs)
        self.W_s = nn.Embedding(len(EXTENDED_AA_ALPHABET), hidden_dim)
        # Head 1: 预测基础氨基酸 (20类)
        self.W_out_base = nn.Linear(hidden_dim, len(NATURAL_AA_ALPHABET))
        # Head 2: 预测是否甲基化 (2类)
        self.W_out_methyl = nn.Linear(hidden_dim, 2)

    def forward(self, X, S, mask, chain_M, residue_idx, chain_encoding_all):
        # 复用原始MPNN的Encoder/Decoder逻辑
        E, E_idx = self.features(X, mask, residue_idx, chain_encoding_all)
        h_V = torch.zeros((E.shape[0], E.shape[1], E.shape[-1]), device=X.device)
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
            
        # 双头输出
        logits_base = self.W_out_base(h_V)
        logits_methyl = self.W_out_methyl(h_V)
        return logits_base, logits_methyl

# --- [核心组件 3] 混合损失函数 ---
def calculate_final_loss(logits_base, logits_methyl, targets, mask, 
                         methyl_weight=5.0, focal_loss_fn=None):
    mask_flat = mask.contiguous().view(-1).bool()
    if mask_flat.sum() == 0: return torch.tensor(0.0, device=logits_base.device), 0
    
    targets_flat = targets.contiguous().view(-1)[mask_flat]
    
    # 过滤 'X'
    x_idx = EXTENDED_AA_TO_INDEX.get('X', -1)
    if x_idx != -1:
        valid_mask = targets_flat != x_idx
        if valid_mask.sum() == 0: return torch.tensor(0.0, device=logits_base.device), 0
        targets_flat = targets_flat[valid_mask]
        logits_base = logits_base.contiguous().view(-1, 20)[mask_flat][valid_mask]
        logits_methyl = logits_methyl.contiguous().view(-1, 2)[mask_flat][valid_mask]
    else:
        logits_base = logits_base.contiguous().view(-1, 20)[mask_flat]
        logits_methyl = logits_methyl.contiguous().view(-1, 2)[mask_flat]

    # 1. Base Targets (映射回天然AA)
    base_targets = targets_flat.clone()
    offset = len(NATURAL_AA_ALPHABET)
    for methyl_rel, natural_idx in NMETHYL_TO_NATURAL_MAPPING.items():
        base_targets[base_targets == (methyl_rel + offset)] = natural_idx
    base_targets[base_targets >= len(NATURAL_AA_ALPHABET)] = -100 # Ignore invalid
    
    # 2. Methyl Targets (0 或 1)
    methyl_targets = (targets_flat >= len(NATURAL_AA_ALPHABET)).long()

    # 计算 Loss
    # Head 1: 始终使用 CrossEntropy + Label Smoothing (0.1)
    loss_base = F.cross_entropy(logits_base, base_targets, label_smoothing=0.1, ignore_index=-100)
    
    # Head 2: 使用 Focal Loss (因为这里极度不平衡)
    # 即使使用了 Focal Loss，我们依然乘以 methyl_weight (5.0) 以确保梯度足够大
    if focal_loss_fn:
        loss_methyl = focal_loss_fn(logits_methyl, methyl_targets)
    else:
        loss_methyl = F.cross_entropy(logits_methyl, methyl_targets, weight=torch.tensor([1.0, 5.0], device=logits_base.device))
        
    total_loss = loss_base + methyl_weight * loss_methyl
    return total_loss, base_targets.numel()

# --- 数据处理 ---
class JSONLDataset(Dataset):
    def __init__(self, jsonl_file, augment=False, aug_rate=0.2):
        self.data = []
        self.augment = augment
        self.aug_rate = aug_rate
        with open(jsonl_file, 'r') as f:
            for line in f: self.data.append(json.loads(line))
            
    def __len__(self): return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx].copy()
        if self.augment: # 简单的Mask增强
            for key in item:
                if key.startswith('seq_chain_'):
                    seq = list(item[key])
                    if not seq: continue
                    n_mask = int(len(seq) * self.aug_rate)
                    if n_mask > 0:
                        indices = random.sample(range(len(seq)), n_mask)
                        for i in indices: seq[i] = 'X'
                    item[key] = "".join(seq)
        return item

def get_weighted_sampler(dataset):
    # 5倍上采样包含N-甲基化的样本
    weights = []
    offset = len(NATURAL_AA_ALPHABET)
    methyl_indices = {k + offset for k in NMETHYL_TO_NATURAL_MAPPING.keys()}
    
    print("Scanning dataset for N-methylated residues...")
    count = 0
    for item in dataset.data:
        has_methyl = False
        for key in item:
            if key.startswith('seq_chain_'):
                for char in item[key]:
                    if EXTENDED_AA_TO_INDEX.get(char, -1) in methyl_indices:
                        has_methyl = True; break
            if has_methyl: break
        
        if has_methyl: 
            weights.append(5.0)
            count += 1
        else: 
            weights.append(1.0)
            
    print(f"Found {count} samples with N-methylation. Up-weighting them.")
    return WeightedRandomSampler(weights, len(weights), replacement=True)

def collate_fn(batch): return batch

def featurize_batch(batch, device):
    # 标准ProteinMPNN特征提取，简化版
    alphabet = EXTENDED_AA_ALPHABET
    B = len(batch)
    batch = [b for b in batch if 'seq' in b and len(b['seq']) > 0]
    if not batch: return None
    lengths = [len(b['seq']) for b in batch]
    L_max = max(lengths)
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
            if not seq: continue
            # 读取坐标 (简化)
            N = np.array(b.get(f'N_chain_{c_id}', []))
            CA = np.array(b.get(f'CA_chain_{c_id}', []))
            C = np.array(b.get(f'C_chain_{c_id}', []))
            O = np.array(b.get(f'O_chain_{c_id}', []))
            l = min(len(seq), len(CA))
            if l == 0: continue
            
            X[i, l_p:l_p+l, 0, :] = N[:l]; X[i, l_p:l_p+l, 1, :] = CA[:l]
            X[i, l_p:l_p+l, 2, :] = C[:l]; X[i, l_p:l_p+l, 3, :] = O[:l]
            S[i, l_p:l_p+l] = [alphabet.index(aa) if aa in alphabet else EXTENDED_AA_TO_INDEX['X'] for aa in seq[:l]]
            if c_id in b.get('masked_list', []): chain_M[i, l_p:l_p+l] = 1.0
            residue_idx[i, l_p:l_p+l] = np.arange(l) + c_i * 100
            chain_encoding_all[i, l_p:l_p+l] = c_i
            l_p += l
            
    isnan = np.isnan(X); mask = np.isfinite(np.sum(X, (2,3))).astype(np.float32); X[isnan] = 0.
    return [torch.from_numpy(t).to(dtype=torch.long if t.dtype==np.int32 else torch.float32, device=device) for t in [X, S, mask, chain_M, residue_idx, chain_encoding_all]]

# --- 主流程 ---
def final_evaluation(model, loader, device):
    model.eval()
    print("\n=== Final Evaluation ===")
    all_preds, all_targets = [], []
    nat_to_me_abs = {v: k + len(NATURAL_AA_ALPHABET) for k, v in NMETHYL_TO_NATURAL_MAPPING.items()}
    
    with torch.no_grad():
        for batch in loader:
            f = featurize_batch(batch, device)
            if f is None: continue
            lb, lm = model(*f)
            
            # 推理逻辑：组合两个头
            base = torch.argmax(lb, -1)
            is_me = torch.argmax(lm, -1)
            final = base.clone()
            
            # 向量化更新：如果预测甲基化且该AA支持甲基化，则替换
            for n_idx, m_idx in nat_to_me_abs.items():
                mask_update = (is_me == 1) & (base == n_idx)
                final[mask_update] = m_idx
                
            # 收集
            tgts = f[1].cpu().numpy().flatten()
            preds = final.cpu().numpy().flatten()
            mask = f[2].cpu().numpy().flatten().bool()
            valid = mask & (tgts != EXTENDED_AA_TO_INDEX.get('X', -1))
            
            all_preds.extend(preds[valid])
            all_targets.extend(tgts[valid])
            
    all_preds, all_targets = np.array(all_preds), np.array(all_targets)
    acc = np.mean(all_preds == all_targets)
    
    # 详细报告
    labels = sorted(np.unique(np.concatenate([all_targets, all_preds])).astype(int))
    names = [EXTENDED_AA_ALPHABET[i] for i in labels if i < len(EXTENDED_AA_ALPHABET)]
    print(classification_report(all_targets, all_preds, labels=labels, target_names=names, zero_division=0))
    print(f"Overall Accuracy: {acc:.4f}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrained_weights", type=str, required=True)
    parser.add_argument("--nmethyl_data", type=str, required=True)
    parser.add_argument("--test_data", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="./final_model_production")
    parser.add_argument("--epochs", type=int, default=250) # 生产级 Epochs
    args = parser.parse_args()
    
    set_seed(42)
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. 加载数据
    train_ds = JSONLDataset(args.nmethyl_data, augment=True)
    sampler = get_weighted_sampler(train_ds) # 必须使用Sampler
    train_loader = DataLoader(train_ds, batch_size=8, sampler=sampler, collate_fn=collate_fn)
    
    test_ds = JSONLDataset(args.test_data)
    test_loader = DataLoader(test_ds, batch_size=8, shuffle=False, collate_fn=collate_fn)
    
    # 2. 初始化模型
    model = DecoupledProteinMPNN().to(device)
    
    # 3. 加载预训练权重 (智能过滤)
    ckpt = torch.load(args.pretrained_weights, map_location=device)
    state = ckpt.get('model_state_dict', ckpt)
    model_dict = model.state_dict()
    state = {k: v for k, v in state.items() if k in model_dict and v.shape == model_dict[k].shape}
    model.load_state_dict(state, strict=False)
    
    # 4. 初始化 Heads
    with torch.no_grad():
        if 'W_out.weight' in ckpt:
            model.W_out_base.weight.data = ckpt['W_out.weight'][:20].clone()
            model.W_out_base.bias.data = ckpt['W_out.bias'][:20].clone()
            
    # 5. 优化器 (冠军配置: lr_mult=0.1)
    new_layers = ['W_s', 'W_out_base', 'W_out_methyl']
    optimizer = torch.optim.AdamW([
        {'params': [p for n, p in model.named_parameters() if not any(k in n for k in new_layers)], 'lr': 1e-4 * 0.1}, # 0.1 Multiplier
        {'params': [p for n, p in model.named_parameters() if any(k in n for k in new_layers)], 'lr': 1e-4}
    ], weight_decay=1e-4)
    
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs * len(train_loader))
    focal_loss = FocalLoss(gamma=2.0) # 冠军配置: Gamma=2.0
    
    # 6. 训练循环
    best_loss = float('inf')
    print(f"Starting Production Training for {args.epochs} epochs...")
    
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss, n_steps = 0, 0
        
        for batch in train_loader:
            f = featurize_batch(batch, device)
            if f is None: continue
            
            optimizer.zero_grad()
            lb, lm = model(*f)
            
            # 冠军配置: Methyl Weight = 5.0
            loss, valid = calculate_final_loss(lb, lm, f[1], f[2], methyl_weight=5.0, focal_loss_fn=focal_loss)
            
            if valid > 0 and torch.isfinite(loss):
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                total_loss += loss.item()
                n_steps += 1
                
        avg_loss = total_loss / n_steps if n_steps > 0 else 0
        print(f"Epoch {epoch}: Loss = {avg_loss:.4f}")
        
        # 保存 Checkpoint
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save({'model_state_dict': model.state_dict()}, os.path.join(args.output_dir, "best_model.pt"))
            
    # 7. 结束
    print("Training Complete.")
    model.load_state_dict(torch.load(os.path.join(args.output_dir, "best_model.pt"))['model_state_dict'])
    final_evaluation(model, test_loader, device)

if __name__ == "__main__":
    main()
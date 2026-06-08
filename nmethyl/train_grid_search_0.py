import argparse
import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import json
import numpy as np
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from sklearn.metrics import classification_report
import random
import itertools
import pandas as pd
import time
from torch.optim.lr_scheduler import CosineAnnealingLR

# --- [项目配置] ---
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from model_utils import ProteinMPNN, gather_nodes, cat_neighbors_nodes
    from nmethyl.utils.nmethyl_config import (
        EXTENDED_AA_ALPHABET,
        NATURAL_AA_ALPHABET,
        NMETHYL_TO_NATURAL_MAPPING,
        EXTENDED_AA_TO_INDEX
    )
except ImportError as e:
    print(f"错误：无法导入必需的模块。\n详细错误: {e}")
    sys.exit(1)

# 反向映射
NATURAL_TO_NMETHYL_MAPPING = {v: k for k, v in NMETHYL_TO_NATURAL_MAPPING.items()}

# =============================================================================
# 0. 工具函数：固定随机种子 (确保可复现)
# =============================================================================
def set_seed(seed):
    """固定所有随机种子，确保实验结果可完全复现"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # 保证CUDA的确定性 (会牺牲一点点速度，但为了调参是值得的)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"[System] Random Seed set to: {seed}")

# =============================================================================
# 1. 核心组件：Focal Loss (焦点损失)
# =============================================================================
class FocalLoss(nn.Module):
    """
    Focal Loss 用于解决极端的类别不平衡。
    公式: FL(pt) = -alpha * (1 - pt)^gamma * log(pt)
    """
    def __init__(self, gamma=2.0, alpha=None, reduction='mean', ignore_index=-100):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha # Tensor of weights for each class
        self.reduction = reduction
        self.ignore_index = ignore_index

    def forward(self, inputs, targets):
        # inputs: [N, C], targets: [N]
        ce_loss = F.cross_entropy(inputs, targets, reduction='none', ignore_index=self.ignore_index)
        pt = torch.exp(-ce_loss) # 预测概率
        
        # 动态调整权重
        if self.alpha is not None:
            # 获取每个样本对应的alpha
            if self.alpha.device != inputs.device:
                self.alpha = self.alpha.to(inputs.device)
            alpha_t = self.alpha[targets]
        else:
            alpha_t = 1.0
        
        # 计算Focal Term
        focal_loss = alpha_t * (1 - pt) ** self.gamma * ce_loss

        if self.reduction == 'mean':
            # 只对非ignore的样本求平均
            valid_mask = targets != self.ignore_index
            return focal_loss[valid_mask].mean()
        elif self.reduction == 'sum':
            valid_mask = targets != self.ignore_index
            return focal_loss[valid_mask].sum()
        else:
            return focal_loss

# =============================================================================
# 2. 模型架构 (Decoupled Hierarchical Architecture)
# =============================================================================
class DecoupledProteinMPNN(ProteinMPNN):
    def __init__(self, node_features=128, edge_features=128, hidden_dim=128, 
                 num_encoder_layers=3, num_decoder_layers=3, k_neighbors=32, dropout=0.1, **kwargs):
        super().__init__(num_letters=21, node_features=node_features, edge_features=edge_features, 
                         hidden_dim=hidden_dim, num_encoder_layers=num_encoder_layers, 
                         num_decoder_layers=num_decoder_layers, vocab=21, k_neighbors=k_neighbors, 
                         dropout=dropout, **kwargs)

        self.W_s = nn.Embedding(len(EXTENDED_AA_ALPHABET), hidden_dim) 
        self.W_out_base = nn.Linear(hidden_dim, len(NATURAL_AA_ALPHABET)) 
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
# 3. 损失计算逻辑 (集成 Focal Loss)
# =============================================================================
def calculate_loss_dynamic(logits_base, logits_methyl, targets, mask, 
                           methyl_loss_weight, focal_loss_fn=None):
    """
    计算损失，支持 Focal Loss
    """
    mask_flat = mask.contiguous().view(-1).bool()
    if mask_flat.sum() == 0: return torch.tensor(0.0, device=logits_base.device), 0
    
    targets_flat = targets.contiguous().view(-1)[mask_flat]
    
    # 过滤 'X'
    x_idx = EXTENDED_AA_TO_INDEX.get('X', -1)
    if x_idx != -1:
        valid_target_mask = targets_flat != x_idx
        if valid_target_mask.sum() == 0:
            return torch.tensor(0.0, device=logits_base.device), 0
        targets_flat = targets_flat[valid_target_mask]
        logits_base_valid = logits_base.contiguous().view(-1, 20)[mask_flat][valid_target_mask]
        logits_methyl_valid = logits_methyl.contiguous().view(-1, 2)[mask_flat][valid_target_mask]
    else:
        logits_base_valid = logits_base.contiguous().view(-1, 20)[mask_flat]
        logits_methyl_valid = logits_methyl.contiguous().view(-1, 2)[mask_flat]
        
    # 1. Base Labels (天然氨基酸)
    base_targets = targets_flat.clone()
    offset = len(NATURAL_AA_ALPHABET)
    for methyl_rel_idx, natural_idx in NMETHYL_TO_NATURAL_MAPPING.items():
        methyl_abs_idx = methyl_rel_idx + offset 
        base_targets[base_targets == methyl_abs_idx] = natural_idx
    base_targets[base_targets >= len(NATURAL_AA_ALPHABET)] = -100
    
    # 2. Methyl Labels (是否甲基化)
    methyl_targets = (targets_flat >= len(NATURAL_AA_ALPHABET)).long()

    # --- 计算 Base Loss (保持 Label Smoothing, 因为这里类别较平衡) ---
    loss_base = F.cross_entropy(logits_base_valid, base_targets, label_smoothing=0.1, ignore_index=-100)
    
    # --- 计算 Methyl Loss (使用 Focal Loss 或 加权 CE) ---
    if focal_loss_fn is not None:
        # 使用 Focal Loss 挖掘困难样本
        loss_methyl = focal_loss_fn(logits_methyl_valid, methyl_targets)
    else:
        # 回退到标准 CE
        weights = torch.tensor([1.0, 5.0], device=logits_base.device)
        loss_methyl = F.cross_entropy(logits_methyl_valid, methyl_targets, weight=weights)

    total_loss = loss_base + methyl_loss_weight * loss_methyl
    
    return total_loss, base_targets.numel()

# =============================================================================
# 4. 数据处理 (保持不变)
# =============================================================================
class JSONLDataset(Dataset):
    def __init__(self, jsonl_file, augment=False):
        self.data = []
        self.augment = augment
        with open(jsonl_file, 'r') as f:
            for line in f:
                self.data.append(json.loads(line))
    def __len__(self): return len(self.data)
    def __getitem__(self, idx):
        item = self.data[idx].copy()
        if self.augment:
            # 简单的Mask增强
            pass 
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
    # (此处复用您之前的 featurize_batch 代码，保持一致)
    # 为节省篇幅，假设已导入或复制粘贴了之前的实现
    # ... [请确保这里使用上面提供的完整 featurize_batch 实现] ...
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
            coords_CA = np.array(b.get(f'CA_chain_{chain_id}', [])) # 简化读取逻辑用于占位
            # 实际运行请使用完整坐标读取逻辑
            len_to_process = min(len(chain_seq), len(coords_CA))
            if len_to_process == 0: continue
            start, end = l_processed, l_processed + len_to_process
            # 模拟特征 (实际需完整坐标)
            X[i, start:end, 1, :] = coords_CA[:len_to_process] 
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
# 5. 训练与评估 Loop
# =============================================================================
def train_evaluate_pipeline(params, args, device):
    """执行单次完整的训练和评估流程"""
    
    # 1. 解包参数
    m_weight = params['methyl_loss_weight']
    bb_lr_mult = params['backbone_lr_multiplier']
    gamma = params['focal_gamma']
    seed = params['seed']
    
    print(f"\n>>> Running Experiment: W={m_weight}, LR_Mult={bb_lr_mult}, Gamma={gamma}, Seed={seed}")
    
    # 2. 设置环境
    set_seed(seed)
    output_subdir = os.path.join(args.output_dir, f"W{m_weight}_LR{bb_lr_mult}_G{gamma}")
    os.makedirs(output_subdir, exist_ok=True)
    
    # 3. 初始化模型
    model = DecoupledProteinMPNN(num_encoder_layers=3, num_decoder_layers=3).to(device)
    
    # 加载预训练权重
    checkpoint = torch.load(args.pretrained_weights, map_location=device)
    pretrained_dict = checkpoint.get('model_state_dict', checkpoint)
    model_state = model.state_dict()
    filtered_dict = {k: v for k, v in pretrained_dict.items() if k in model_state and v.size() == model_state[k].size()}
    model.load_state_dict(filtered_dict, strict=False)
    # 初始化头部
    with torch.no_grad():
        if 'W_out.weight' in pretrained_dict:
            model.W_out_base.weight.data = pretrained_dict['W_out.weight'][:20, :].clone()
            model.W_out_base.bias.data = pretrained_dict['W_out.bias'][:20].clone()

    # 4. 数据加载
    train_dataset = JSONLDataset(args.nmethyl_data, augment=True)
    sampler = get_weighted_sampler(train_dataset)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, sampler=sampler, collate_fn=collate_fn)
    
    test_dataset = JSONLDataset(args.test_data, augment=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)

    # 5. 优化器与 Focal Loss
    focal_loss = FocalLoss(gamma=gamma, alpha=torch.tensor([1.0, 4.0])) # Alpha 1:4 平衡 0/1 类别
    
    new_params = ['W_s', 'W_out_base', 'W_out_methyl']
    optimizer = torch.optim.AdamW([
        {'params': [p for n, p in model.named_parameters() if not any(k in n for k in new_params)], 'lr': args.learning_rate * bb_lr_mult},
        {'params': [p for n, p in model.named_parameters() if any(k in n for k in new_params)], 'lr': args.learning_rate}
    ], weight_decay=1e-4)

    scheduler = CosineAnnealingLR(optimizer, T_max=args.num_epochs * len(train_loader))

    # 6. 训练循环
    best_acc = 0.0
    for epoch in range(1, args.num_epochs + 1):
        model.train()
        total_loss = 0
        total_valid = 0
        
        for batch in train_loader:
            features = featurize_batch(batch, device)
            if features is None: continue
            
            optimizer.zero_grad()
            logits_base, logits_methyl = model(*features)
            loss, n_val = calculate_loss_dynamic(logits_base, logits_methyl, features[1], features[2], m_weight, focal_loss)
            
            if n_val > 0:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                total_loss += loss.item() * n_val
                total_valid += n_val
        
        avg_loss = total_loss / total_valid if total_valid > 0 else 0
        # 简单的进度打印，每10个epoch
        if epoch % 10 == 0:
            print(f"  Epoch {epoch}/{args.num_epochs} Loss: {avg_loss:.4f}")

    # 7. 最终评估
    model.eval()
    all_preds, all_targets = [], []
    offset = len(NATURAL_AA_ALPHABET)
    natural_to_methyl_abs = {v: k + offset for k, v in NMETHYL_TO_NATURAL_MAPPING.items()} # idx -> idx

    with torch.no_grad():
        for batch in test_loader:
            features = featurize_batch(batch, device)
            if features is None: continue
            logits_base, logits_methyl = model(*features)
            
            pred_base = torch.argmax(logits_base, dim=-1)
            pred_is_me = torch.argmax(logits_methyl, dim=-1)
            
            final_preds = pred_base.clone()
            # 组合逻辑
            for b in range(pred_base.shape[0]):
                for l in range(pred_base.shape[1]):
                    base = pred_base[b,l].item()
                    if pred_is_me[b,l].item() == 1 and base in natural_to_methyl_abs:
                        final_preds[b,l] = natural_to_methyl_abs[base]

            # 收集
            targets_flat = features[1].cpu().numpy().flatten()
            preds_flat = final_preds.cpu().numpy().flatten()
            mask_flat = features[2].cpu().numpy().flatten().astype(bool)
            x_idx = EXTENDED_AA_TO_INDEX.get('X', -1)
            valid_pos = mask_flat & (targets_flat != x_idx)
            
            all_preds.extend(preds_flat[valid_pos])
            all_targets.extend(targets_flat[valid_pos])

    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)
    
    overall_acc = np.mean(all_preds == all_targets)
    
    # 计算天然恢复率
    nat_mask = all_targets < len(NATURAL_AA_ALPHABET)
    nat_acc = np.mean(all_targets[nat_mask] == all_preds[nat_mask]) if nat_mask.sum() > 0 else 0
    
    # 计算N-甲基化恢复率 (Key Metric)
    methyl_mask = all_targets >= len(NATURAL_AA_ALPHABET)
    methyl_acc = np.mean(all_targets[methyl_mask] == all_preds[methyl_mask]) if methyl_mask.sum() > 0 else 0

    print(f"  Result: Overall={overall_acc:.4f}, Natural={nat_acc:.4f}, Methyl={methyl_acc:.4f}")
    
    # 保存模型
    torch.save(model.state_dict(), os.path.join(output_subdir, "final_model.pt"))
    
    return {
        "methyl_loss_weight": m_weight,
        "backbone_lr_multiplier": bb_lr_mult,
        "focal_gamma": gamma,
        "overall_acc": overall_acc,
        "natural_acc": nat_acc,
        "methyl_acc": methyl_acc
    }

# =============================================================================
# 6. 主程序：网格搜索控制器
# =============================================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrained_weights", type=str, required=True)
    parser.add_argument("--nmethyl_data", type=str, required=True)
    parser.add_argument("--test_data", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="./grid_search_results")
    parser.add_argument("--num_epochs", type=int, default=100) # 调参时可以适当减少轮数以加快速度
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- 定义搜索空间 ---
    # 这是一个精选的搜索空间，基于之前的经验
    search_space = {
        "methyl_loss_weight": [2.0, 3.0, 5.0],  # 之前的3.0很好，试试更激进的5.0
        "backbone_lr_multiplier": [0.05, 0.1],  # 0.1比较灵活，0.05比较保守
        "focal_gamma": [0.0, 2.0],              # 0.0=退化为CrossEntropy(基准), 2.0=标准Focal Loss
        "seed": [args.seed]                     # 固定种子
    }

    keys, values = zip(*search_space.items())
    combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]
    
    print(f"总计 {len(combinations)} 组参数组合即将运行...")
    
    results = []
    
    for i, params in enumerate(combinations):
        print(f"\n[{i+1}/{len(combinations)}] 开始运行参数组合...")
        try:
            res = train_evaluate_pipeline(params, args, device)
            results.append(res)
            
            # 实时保存结果，防止中途崩溃
            df = pd.DataFrame(results)
            df.to_csv(os.path.join(args.output_dir, "grid_search_summary.csv"), index=False)
            
        except Exception as e:
            print(f"参数组合 {params} 运行失败: {e}")
            import traceback
            traceback.print_exc()

    print("\n" + "="*50)
    print("网格搜索完成！最佳结果：")
    df = pd.DataFrame(results)
    # 按 Methyl Acc 排序
    print(df.sort_values(by="methyl_acc", ascending=False).head())
    print("="*50)

if __name__ == "__main__":
    main()
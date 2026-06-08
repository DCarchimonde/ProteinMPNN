import argparse
import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import json
import numpy as np
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import random
from collections import Counter
from torch.optim.lr_scheduler import CosineAnnealingLR

# --- [关键] 动态添加项目根目录到Python路径 ---
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# --- [关键] 从您的文件中导入必需的模块 ---
try:
    from model_utils import ProteinMPNN, gather_nodes, cat_neighbors_nodes
    from nmethyl.utils.nmethyl_config import (
        EXTENDED_AA_ALPHABET,
        NATURAL_AA_ALPHABET,
        NMETHYL_TO_NATURAL_MAPPING,
        EXTENDED_AA_TO_INDEX
    )
except ImportError as e:
    print(f"错误：无法导入必需的模块。请确保您的目录结构正确，且文件都存在。\n详细错误: {e}")
    sys.exit(1)


# =============================================================================
# 1. 损失函数定义 (Loss Functions)
# =============================================================================

class FocalLoss(nn.Module):
    """
    焦点损失 (Focal Loss)
    用于解决极端的类别不平衡问题。它降低了简单样本（模型已有把握的）的权重，
    专注于困难样本（N-甲基化氨基酸）。
    公式: FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
    """
    def __init__(self, alpha=None, gamma=2.0, ignore_index=-100, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha # 类别权重张量
        self.ignore_index = ignore_index
        self.reduction = reduction

    def forward(self, input, target):
        # input: (N, C) log_probs (因为我们模型输出的是 log_softmax)
        log_pt = input
        pt = torch.exp(log_pt)
        
        # 计算 focal term: (1 - pt)^gamma
        # gather: 选择与 target 对应的概率值
        pt_gather = pt.gather(1, target.unsqueeze(1)).squeeze(1)
        focal_term = (1 - pt_gather).pow(self.gamma)
        
        # 计算标准 NLL Loss (不带 reduction，以便乘以 focal term)
        loss = F.nll_loss(log_pt, target, weight=self.alpha, reduction='none', ignore_index=self.ignore_index)
        
        # 结合 focal term
        loss = focal_term * loss

        if self.reduction == 'mean':
            valid_mask = target != self.ignore_index
            if valid_mask.sum() > 0:
                return loss[valid_mask].mean()
            else:
                return torch.tensor(0.0, device=input.device)
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


class LabelSmoothingNLLLoss(nn.Module):
    """带有标签平滑的负对数似然损失 (用于辅助任务或不使用Focal Loss时)"""
    def __init__(self, smoothing=0.1):
        super(LabelSmoothingNLLLoss, self).__init__()
        self.smoothing = smoothing

    def forward(self, x, target):
        confidence = 1. - self.smoothing
        log_probs = x
        nll_loss = -log_probs.gather(dim=-1, index=target.unsqueeze(1)).squeeze(1)
        smooth_loss = -log_probs.mean(dim=-1)
        loss = (confidence * nll_loss + self.smoothing * smooth_loss)
        return loss.mean()


def calculate_combined_loss(log_probs_main, log_probs_aux, targets, mask, main_loss_fn, aux_loss_fn, aux_weight):
    """计算主任务和辅助任务的加权损失"""
    # 1. 主任务损失 (35类)
    mask_flat = mask.contiguous().view(-1).bool()
    if mask_flat.sum() == 0:
        return torch.tensor(0.0, device=log_probs_main.device), 0
    
    log_probs_main_flat = log_probs_main.contiguous().view(-1, log_probs_main.size(-1))
    targets_flat = targets.contiguous().view(-1)
    
    valid_log_probs_main = log_probs_main_flat[mask_flat]
    valid_targets = targets_flat[mask_flat]
    
    loss_main = main_loss_fn(valid_log_probs_main, valid_targets)
    num_valid = valid_targets.numel()

    # 2. 辅助任务损失 (仅天然氨基酸)
    loss_aux = torch.tensor(0.0, device=log_probs_main.device)
    if aux_loss_fn is not None and aux_weight > 0:
        natural_aa_mask = (targets < len(NATURAL_AA_ALPHABET)) & mask.bool()
        natural_mask_flat = natural_aa_mask.contiguous().view(-1).bool()
        
        if natural_mask_flat.sum() > 0:
            log_probs_aux_flat = log_probs_aux.contiguous().view(-1, log_probs_aux.size(-1))
            valid_log_probs_aux = log_probs_aux_flat[natural_mask_flat]
            valid_targets_aux = targets_flat[natural_mask_flat]
            loss_aux = aux_loss_fn(valid_log_probs_aux, valid_targets_aux)

    combined_loss = loss_main + aux_weight * loss_aux
    
    if not torch.isfinite(combined_loss):
        return loss_main, num_valid
    
    return combined_loss, num_valid


def calculate_class_weights(jsonl_file, device):
    """计算类别权重以处理数据不平衡"""
    print("正在计算类别权重以处理数据不平衡 (用于 Focal Loss)...")
    counts = Counter()
    
    with open(jsonl_file, 'r') as f:
        for line in f:
            item = json.loads(line)
            for key in item:
                if key.startswith('seq_chain_'):
                    seq = item[key]
                    for char in seq:
                        if char in EXTENDED_AA_TO_INDEX:
                            counts[EXTENDED_AA_TO_INDEX[char]] += 1

    num_classes = len(EXTENDED_AA_ALPHABET)
    weights = torch.ones(num_classes, device=device)
    total_count = sum(counts.values())
    
    for idx in range(num_classes):
        count = counts[idx]
        if count > 0:
            # 经典的平衡策略: total / (num_classes * count)
            weights[idx] = total_count / (count * num_classes)
        else:
            weights[idx] = 1.0 

    # 归一化权重
    weights = weights / weights.mean()
    print("  类别权重计算完成。")
    return weights


# =============================================================================
# 2. 模型架构 (Model Architecture)
# =============================================================================

class ExtendedProteinMPNN(ProteinMPNN):
    def __init__(self, num_letters=len(EXTENDED_AA_ALPHABET), k_neighbors=32, **kwargs):
        super().__init__(num_letters=21, vocab=21, k_neighbors=k_neighbors, **kwargs)

        # 覆盖主输出层以支持扩展字母表
        self.W_s = nn.Embedding(num_letters, self.hidden_dim) 
        self.W_out = nn.Linear(self.hidden_dim, num_letters) 

        # 辅助任务头 (20类天然氨基酸)
        self.W_out_natural = nn.Linear(self.hidden_dim, len(NATURAL_AA_ALPHABET))
        print(f"初始化模型: 主头 ({num_letters} 类), 辅助头 ({len(NATURAL_AA_ALPHABET)} 类)。")

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
        
        logits_main = self.W_out(h_V)
        log_probs_main = F.log_softmax(logits_main, dim=-1)

        logits_aux = self.W_out_natural(h_V)
        log_probs_aux = F.log_softmax(logits_aux, dim=-1)

        return log_probs_main, log_probs_aux


# =============================================================================
# 3. 数据处理与特征化
# =============================================================================

class JSONLDataset(Dataset):
    def __init__(self, jsonl_file, augment=False, augmentation_rate=0.20):
        self.data = []
        self.augment = augment
        self.augmentation_rate = augmentation_rate
        self.mask_token_idx = EXTENDED_AA_TO_INDEX.get('X', -1)
        with open(jsonl_file, 'r') as f:
            for line in f:
                self.data.append(json.loads(line))
    
    def __len__(self):
        return len(self.data)
    
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

# [关键] 全局函数，无缩进
def collate_fn(batch):
    return batch

def featurize_batch(batch, device):
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
# 4. 训练流程
# =============================================================================

def train_one_epoch(epoch, model, loader, optimizer, loss_fns, aux_weight, device, scheduler, warmup_steps, current_step):
    model.train()
    total_loss, total_valid = 0, 0
    main_loss_fn, aux_loss_fn = loss_fns

    for batch_idx, batch in enumerate(loader):
        if current_step < warmup_steps:
            lr_scale = (current_step + 1) / warmup_steps
            for group in optimizer.param_groups:
                group['lr'] = group['initial_lr'] * lr_scale
        current_step += 1
        
        try:
            features = featurize_batch(batch, device)
            if features is None: continue
            
            optimizer.zero_grad()
            log_probs_main, log_probs_aux = model(*features)
            
            loss, num_valid = calculate_combined_loss(log_probs_main, log_probs_aux, features[1], features[2], main_loss_fn, aux_loss_fn, aux_weight)
            
            if num_valid > 0 and torch.isfinite(loss):
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                total_loss += loss.item() * num_valid
                total_valid += num_valid
            
            if batch_idx % 20 == 0 and num_valid > 0:
                print(f"Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item():.4f}, LR: {optimizer.param_groups[0]['lr']:.6f}")
        except Exception as e:
            print(f"Error in training batch {batch_idx}: {e}")
            continue
            
    if scheduler is not None and current_step >= warmup_steps:
        scheduler.step()
        
    return total_loss / total_valid if total_valid > 0 else float('inf'), current_step

def final_evaluation(model, test_loader, device, output_dir):
    model.eval()
    print("Starting final model evaluation...")
    all_preds, all_targets = [], []
    with torch.no_grad():
        for batch in test_loader:
            try:
                features = featurize_batch(batch, device)
                if features is None: continue
                log_probs_main, _ = model(*features)
                preds = torch.argmax(log_probs_main, dim=-1)
                mask_flat = features[2].cpu().numpy().flatten().astype(bool)
                all_preds.extend(preds.cpu().numpy().flatten()[mask_flat])
                all_targets.extend(features[1].cpu().numpy().flatten()[mask_flat])
            except Exception as e:
                print(f"Error in test batch: {e}")

    all_preds, all_targets = np.array(all_preds), np.array(all_targets)
    overall_acc = np.mean(all_preds == all_targets) if len(all_targets) > 0 else 0
    
    natural_mask = all_targets < len(NATURAL_AA_ALPHABET)
    natural_acc = np.mean(all_targets[natural_mask] == all_preds[natural_mask]) if natural_mask.sum() > 0 else 0

    print("\n" + "="*50 + "\nTEST RESULTS\n" + "="*50)
    print(f"Overall Accuracy ({len(EXTENDED_AA_ALPHABET)} Classes): {overall_acc:.4f} ({overall_acc*100:.2f}%)")
    print("\n" + "-"*50)
    print(f"--- KEY METRIC vs BENCHMARK ---")
    print(f"Sequence Recovery on Natural AAs: {natural_acc:.4f} ({natural_acc*100:.2f}%)")
    print(f"Benchmark (ProteinMPNN):        52.40%")
    print(f"-"*50 + "\n")

    unique_labels = np.unique(np.concatenate((all_targets, all_preds))).astype(int)
    class_names = [EXTENDED_AA_ALPHABET[i] for i in unique_labels if i < len(EXTENDED_AA_ALPHABET)]
    report = classification_report(all_targets, all_preds, labels=unique_labels, target_names=class_names, zero_division=0)
    
    print(f"Detailed Classification Report:\n{report}")


# =============================================================================
# 5. 主执行函数
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Ultimate Training Script for N-methylated ProteinMPNN")
    parser.add_argument("--pretrained_weights", type=str, required=True)
    parser.add_argument("--nmethyl_data", type=str, required=True)
    parser.add_argument("--test_data", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="./final_model")
    parser.add_argument("--num_epochs", type=int, default=150)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--backbone_lr_multiplier", type=float, default=0.08)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--early_stopping_patience", type=int, default=25)
    parser.add_argument("--use_augmentation", action="store_true")
    parser.add_argument("--num_frozen_epochs", type=int, default=5)
    parser.add_argument("--label_smoothing", type=float, default=0.1, help="标签平滑因子")
    parser.add_argument("--use_warmup_scheduler", action="store_true")
    parser.add_argument("--warmup_epochs", type=int, default=3)
    parser.add_argument("--use_aux_loss", action="store_true")
    parser.add_argument("--aux_loss_weight", type=float, default=0.6)
    
    # --- [关键] Focal Loss 参数 ---
    parser.add_argument("--use_focal_loss", action="store_true", help="是否使用Focal Loss")
    parser.add_argument("--focal_loss_gamma", type=float, default=2.0, help="Focal Loss的gamma参数")
    
    # 仅用于测试模式
    parser.add_argument("--test_only", action="store_true")
    parser.add_argument("--test_model_path", type=str, default=None)
    
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    model = ExtendedProteinMPNN(num_letters=len(EXTENDED_AA_ALPHABET), node_features=128, edge_features=128, hidden_dim=128, num_encoder_layers=3, num_decoder_layers=3).to(device)

    if args.test_only:
        if not args.test_model_path: raise ValueError("--test_only requires --test_model_path")
        checkpoint = torch.load(args.test_model_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        test_dataset = JSONLDataset(args.test_data, augment=False)
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)
        final_evaluation(model, test_loader, device, args.output_dir)
        return

    train_dataset = JSONLDataset(args.nmethyl_data, augment=args.use_augmentation)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    
    # --- 损失函数选择逻辑 ---
    if args.use_focal_loss:
        print(f"正在使用 Focal Loss (gamma={args.focal_loss_gamma})")
        class_weights = calculate_class_weights(args.nmethyl_data, device)
        main_loss_fn = FocalLoss(alpha=class_weights, gamma=args.focal_loss_gamma).to(device)
        aux_loss_fn = nn.NLLLoss().to(device) if args.use_aux_loss else None
    else:
        print(f"正在使用标准 Label Smoothing (smoothing={args.label_smoothing})")
        main_loss_fn = LabelSmoothingNLLLoss(smoothing=args.label_smoothing).to(device)
        aux_loss_fn = LabelSmoothingNLLLoss(smoothing=args.label_smoothing).to(device) if args.use_aux_loss else None
    
    print(f"Loading pretrained weights from: {args.pretrained_weights}")
    checkpoint = torch.load(args.pretrained_weights, map_location=device)
    pretrained_dict = checkpoint.get('model_state_dict', checkpoint)
    
    # --- [关键修复] 过滤不匹配的权重 ---
    # 这个逻辑防止了 RuntimeError: size mismatch
    model_state = model.state_dict()
    filtered_dict = {k: v for k, v in pretrained_dict.items() if k in model_state and v.size() == model_state[k].size()}
    model.load_state_dict(filtered_dict, strict=False)
    
    # 权重迁移初始化
    with torch.no_grad():
        natural_emb = pretrained_dict['W_s.weight'][:20, :].clone()
        natural_out_w = pretrained_dict['W_out.weight'][:20, :].clone()
        natural_out_b = pretrained_dict['W_out.bias'][:20].clone()
        
        model.W_s.weight.data[:20, :] = natural_emb
        model.W_out.weight.data[:20, :] = natural_out_w
        model.W_out.bias.data[:20] = natural_out_b
        
        for nmethyl_idx, natural_idx in NMETHYL_TO_NATURAL_MAPPING.items():
            model.W_s.weight.data[20 + nmethyl_idx] = natural_emb[natural_idx] + torch.randn_like(natural_emb[0]) * 0.01
            model.W_out.weight.data[20 + nmethyl_idx] = natural_out_w[natural_idx]
            model.W_out.bias.data[20 + nmethyl_idx] = natural_out_b[natural_idx]
        
        model.W_out_natural.weight.data = natural_out_w
        model.W_out_natural.bias.data = natural_out_b

    best_loss = float('inf')
    
    # STAGE 1
    if args.num_frozen_epochs > 0:
        print("\n=== STAGE 1: Training HEADS ===")
        for n, p in model.named_parameters():
            if not any(k in n for k in ['W_s', 'W_out', 'W_out_natural']): p.requires_grad = False
        
        optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.learning_rate, weight_decay=args.weight_decay)
        for epoch in range(1, args.num_frozen_epochs + 1):
            loss, _ = train_one_epoch(epoch, model, train_loader, optimizer, (main_loss_fn, aux_loss_fn), args.aux_loss_weight, device, None, 0, 0)
            print(f"Stage 1 - Epoch {epoch}, Loss: {loss:.4f}")

    # STAGE 2
    print("\n=== STAGE 2: Fine-tuning ALL ===")
    for p in model.parameters(): p.requires_grad = True
    
    new_params = ['W_s', 'W_out', 'W_out_natural']
    optimizer = torch.optim.AdamW([
        {'params': [p for n, p in model.named_parameters() if not any(k in n for k in new_params)], 'lr': args.learning_rate * args.backbone_lr_multiplier},
        {'params': [p for n, p in model.named_parameters() if any(k in n for k in new_params)], 'lr': args.learning_rate}
    ], weight_decay=args.weight_decay)

    scheduler = None
    if args.use_warmup_scheduler:
        scheduler = CosineAnnealingLR(optimizer, T_max=(args.num_epochs - args.num_frozen_epochs) * len(train_loader))
    
    patience, current_step = 0, 0
    for epoch in range(args.num_frozen_epochs + 1, args.num_epochs + 1):
        avg_loss, current_step = train_one_epoch(epoch, model, train_loader, optimizer, (main_loss_fn, aux_loss_fn), args.aux_loss_weight, device, scheduler, args.warmup_epochs * len(train_loader), current_step)
        print(f"Stage 2 - Epoch {epoch}, Loss: {avg_loss:.4f}")
        
        if avg_loss < best_loss:
            best_loss, patience = avg_loss, 0
            torch.save({'model_state_dict': model.state_dict()}, os.path.join(args.output_dir, "best_model.pt"))
            print("New best model saved.")
        else:
            patience += 1
            if patience >= args.early_stopping_patience:
                print("Early stopping triggered.")
                break

    print("Training finished.")
    if args.test_data:
        model.load_state_dict(torch.load(os.path.join(args.output_dir, "best_model.pt"))['model_state_dict'])
        final_evaluation(model, DataLoader(JSONLDataset(args.test_data), batch_size=args.batch_size, collate_fn=collate_fn), device, args.output_dir)

if __name__ == "__main__":
    main()
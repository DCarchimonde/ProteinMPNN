import argparse

import os

import sys

import torch

import torch.nn as nn

import torch.nn.functional as F

import json

import numpy as np

from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

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



# 构建反向映射：天然AA索引 -> N-甲基化AA索引 (用于推理组合)

NATURAL_TO_NMETHYL_MAPPING = {v: k for k, v in NMETHYL_TO_NATURAL_MAPPING.items()}





# =============================================================================

# 1. 解耦分层模型架构 (Decoupled Hierarchical Architecture)

# =============================================================================



class DecoupledProteinMPNN(ProteinMPNN):

    """

    解耦模型：

    Head 1: 预测基础氨基酸类型 (20类)，利用结构相似性。

    Head 2: 预测是否N-甲基化 (2类)，专注细微差异。

    """

    def __init__(self, node_features=128, edge_features=128, hidden_dim=128, 

                 num_encoder_layers=3, num_decoder_layers=3, k_neighbors=32, dropout=0.1, **kwargs):

        # 初始化父类 (vocab=21)

        super().__init__(num_letters=21, node_features=node_features, edge_features=edge_features, 

                         hidden_dim=hidden_dim, num_encoder_layers=num_encoder_layers, 

                         num_decoder_layers=num_decoder_layers, vocab=21, k_neighbors=k_neighbors, 

                         dropout=dropout, **kwargs)



        # 覆盖嵌入层以支持35种输入字符 (包含N-甲基化)

        self.W_s = nn.Embedding(len(EXTENDED_AA_ALPHABET), hidden_dim) 

        

        # --- 核心改变：双头输出 ---

        

        # Head 1: 基础类型预测 (Base Type Identity) - 20类

        # 任务：判断它是 A, R, N ... 还是 V (不区分是否甲基化)

        self.W_out_base = nn.Linear(hidden_dim, len(NATURAL_AA_ALPHABET)) 



        # Head 2: 甲基化状态预测 (Methylation Status) - 2类 (0: No, 1: Yes)

        # 任务：判断它是否被N-甲基化

        self.W_out_methyl = nn.Linear(hidden_dim, 2)

        

        print(f"初始化解耦模型: Base Head (20类) + Methylation Head (2类)")



    def forward(self, X, S, mask, chain_M, residue_idx, chain_encoding_all):

        device = X.device

        # ... 特征提取与编码器 (与原始MPNN一致) ...

        E, E_idx = self.features(X, mask, residue_idx, chain_encoding_all)

        h_V = torch.zeros((E.shape[0], E.shape[1], E.shape[-1]), device=E.device)

        h_E = self.W_e(E)



        mask_attend = gather_nodes(mask.unsqueeze(-1), E_idx).squeeze(-1)

        mask_attend = mask.unsqueeze(-1) * mask_attend

        for layer in self.encoder_layers:

            h_V, h_E = torch.utils.checkpoint.checkpoint(layer, h_V, h_E, E_idx, mask, mask_attend)



        # ... 解码器 (与原始MPNN一致) ...

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

        

        # --- 双头输出 ---

        logits_base = self.W_out_base(h_V)   # [B, L, 20]

        logits_methyl = self.W_out_methyl(h_V) # [B, L, 2]



        return logits_base, logits_methyl





# =============================================================================

# 2. 损失计算逻辑 (Decoupled Loss) - [已修复索引错误]

# =============================================================================



def calculate_decoupled_loss(logits_base, logits_methyl, targets, mask, methyl_weight=2.0):

    """

    计算解耦损失：

    1. Base Loss: 强迫模型学会识别基础氨基酸 (A vs G vs ...)。

    2. Methyl Loss: 强迫模型学会识别是否甲基化。

    """

    mask_flat = mask.contiguous().view(-1).bool()

    if mask_flat.sum() == 0: return torch.tensor(0.0, device=logits_base.device), 0

    

    targets_flat = targets.contiguous().view(-1)[mask_flat] # [N_valid]

    

    # --- 准备标签 ---

    

    # 0. 处理未知字符 'X'

    # 获取 'X' 的索引 (通常是34)

    x_idx = EXTENDED_AA_TO_INDEX.get('X', -1)

    

    # 过滤掉 'X' 标签，避免干扰训练

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

        

    

    # 1. Base Labels: 将所有N-甲基化标签转换回对应的天然标签

    # [Fix] 使用偏移量来正确匹配配置中的相对索引

    base_targets = targets_flat.clone()

    offset = len(NATURAL_AA_ALPHABET) # 20

    

    for methyl_rel_idx, natural_idx in NMETHYL_TO_NATURAL_MAPPING.items():

        # 配置文件中的key是0,1,2... 但输入target是 20,21,22...

        methyl_abs_idx = methyl_rel_idx + offset 

        base_targets[base_targets == methyl_abs_idx] = natural_idx

        

    # 最后的安全检查：任何仍超出20范围的标签设为ignore (防止CUDA错误)

    base_targets[base_targets >= len(NATURAL_AA_ALPHABET)] = -100

    

    # 2. Methyl Labels: 0 (Natural) or 1 (Methylated)

    # 假设前20个是天然 (0-19)，后面是N-甲基化 (>=20)

    methyl_targets = (targets_flat >= len(NATURAL_AA_ALPHABET)).long()



    # --- 计算损失 ---



    # Base Loss: 标准分类损失 (Label Smoothing 可选)

    loss_base = F.cross_entropy(logits_base_valid, base_targets, label_smoothing=0.1, ignore_index=-100)

    

    # Methyl Loss: 给予更高权重

    loss_methyl = F.cross_entropy(logits_methyl_valid, methyl_targets, weight=torch.tensor([1.0, 5.0], device=logits_base.device))



    # 总损失

    total_loss = loss_base + methyl_weight * loss_methyl

    

    return total_loss, base_targets.numel()





# =============================================================================

# 3. 数据加载与加权采样 (Weighted Sampler)

# =============================================================================



class JSONLDataset(Dataset):

    def __init__(self, jsonl_file, augment=False, augmentation_rate=0.20):

        self.data = []

        self.augment = augment

        self.augmentation_rate = augmentation_rate

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



def get_weighted_sampler(dataset):

    """

    创建一个加权采样器，让包含N-甲基化氨基酸的样本在训练中出现得更频繁。

    """

    print("正在构建加权采样器 (Weighted Sampler)...")

    sample_weights = []

    methyl_indices = set(NMETHYL_TO_NATURAL_MAPPING.keys()) # 这里是相对索引

    # 转换为绝对索引

    offset = len(NATURAL_AA_ALPHABET)

    methyl_abs_indices = {k + offset for k in methyl_indices}

    

    n_methyl_samples = 0

    

    for item in dataset.data:

        has_methyl = False

        for key in item:

            if key.startswith('seq_chain_'):

                for char in item[key]:

                    if char in EXTENDED_AA_TO_INDEX:

                        idx = EXTENDED_AA_TO_INDEX[char]

                        if idx in methyl_abs_indices:

                            has_methyl = True

                            break

            if has_methyl: break

        

        if has_methyl:

            sample_weights.append(5.0) # N-甲基化样本权重高

            n_methyl_samples += 1

        else:

            sample_weights.append(1.0) # 普通样本权重低

            

    print(f"  发现 {n_methyl_samples} 个包含N-甲基化AA的样本，已增加其采样权重。")

    

    sampler = WeightedRandomSampler(

        weights=sample_weights,

        num_samples=len(sample_weights),

        replacement=True

    )

    return sampler



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

# 4. 训练与推理流程

# =============================================================================



def train_one_epoch(epoch, model, loader, optimizer, methyl_loss_weight, device, scheduler, warmup_steps, current_step):

    model.train()

    total_loss, total_valid = 0, 0



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

            logits_base, logits_methyl = model(*features) # 获取双头输出

            

            loss, num_valid = calculate_decoupled_loss(logits_base, logits_methyl, features[1], features[2], methyl_loss_weight)

            

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

    print("Starting final model evaluation (Decoupled Inference)...")

    all_preds, all_targets = [], []

    

    # 准备反向映射：天然AA索引 -> N-甲基化AA索引

    offset = len(NATURAL_AA_ALPHABET)

    # NATURAL_TO_NMETHYL_MAPPING 格式: {Natural_Idx: Methyl_Rel_Idx}

    # 我们需要: {Natural_Idx: Methyl_Abs_Idx}

    natural_to_methyl_abs = {}

    for methyl_rel, natural_idx in NMETHYL_TO_NATURAL_MAPPING.items():

        natural_to_methyl_abs[natural_idx] = methyl_rel + offset

        

    with torch.no_grad():

        for batch in test_loader:

            try:

                features = featurize_batch(batch, device)

                if features is None: continue

                logits_base, logits_methyl = model(*features)

                

                # --- 解耦推理逻辑 ---

                # 1. 获取基础氨基酸预测

                pred_base_idx = torch.argmax(logits_base, dim=-1)

                

                # 2. 获取甲基化状态预测 (0: No, 1: Yes)

                pred_is_methyl = torch.argmax(logits_methyl, dim=-1)

                

                # 3. 组合结果

                final_preds = pred_base_idx.clone()

                mask_flat = features[2].cpu().numpy().flatten().astype(bool)

                

                # 遍历每个预测位置 (组合逻辑)

                B, L = pred_base_idx.shape

                for b in range(B):

                    for l in range(L):

                        base = pred_base_idx[b, l].item()

                        is_me = pred_is_methyl[b, l].item()

                        

                        # 如果预测为甲基化，且该基础氨基酸有对应的甲基化版本，则替换为甲基化版本

                        if is_me == 1:

                            if base in natural_to_methyl_abs:

                                final_preds[b, l] = natural_to_methyl_abs[base]

                

                # 收集结果

                # 获取 targets 并过滤掉 'X'

                targets_flat = features[1].cpu().numpy().flatten()

                preds_flat = final_preds.cpu().numpy().flatten()

                

                # 双重过滤：既要mask有效，又不能是X

                x_idx = EXTENDED_AA_TO_INDEX.get('X', -1)

                valid_pos = mask_flat & (targets_flat != x_idx)

                

                all_preds.extend(preds_flat[valid_pos])

                all_targets.extend(targets_flat[valid_pos])

            except Exception as e:

                print(f"Error in test batch: {e}")



    all_preds, all_targets = np.array(all_preds), np.array(all_targets)

    if len(all_targets) == 0:

        print("No valid targets found for evaluation.")

        return



    overall_acc = np.mean(all_preds == all_targets)

    

    natural_mask = all_targets < len(NATURAL_AA_ALPHABET)

    natural_acc = np.mean(all_targets[natural_mask] == all_preds[natural_mask]) if natural_mask.sum() > 0 else 0



    print("\n" + "="*50 + "\nTEST RESULTS (DECOUPLED MODEL)\n" + "="*50)

    print(f"Overall Accuracy: {overall_acc:.4f} ({overall_acc*100:.2f}%)")

    print(f"Sequence Recovery on Natural AAs: {natural_acc:.4f} ({natural_acc*100:.2f}%)")

    print(f"-"*50 + "\n")



    unique_labels = np.unique(np.concatenate((all_targets, all_preds))).astype(int)

    class_names = [EXTENDED_AA_ALPHABET[i] for i in unique_labels if i < len(EXTENDED_AA_ALPHABET)]

    report = classification_report(all_targets, all_preds, labels=unique_labels, target_names=class_names, zero_division=0)

    print(report)



def main():

    parser = argparse.ArgumentParser(description="Decoupled Training Script")

    parser.add_argument("--pretrained_weights", type=str, required=True)

    parser.add_argument("--nmethyl_data", type=str, required=True)

    parser.add_argument("--test_data", type=str, required=True)

    parser.add_argument("--output_dir", type=str, default="./final_decoupled")

    parser.add_argument("--num_epochs", type=int, default=150)

    parser.add_argument("--batch_size", type=int, default=8)

    parser.add_argument("--learning_rate", type=float, default=1e-4)

    parser.add_argument("--backbone_lr_multiplier", type=float, default=0.08)

    parser.add_argument("--weight_decay", type=float, default=1e-4)

    parser.add_argument("--early_stopping_patience", type=int, default=25)

    parser.add_argument("--use_augmentation", action="store_true")

    parser.add_argument("--num_frozen_epochs", type=int, default=5)

    parser.add_argument("--use_warmup_scheduler", action="store_true")

    parser.add_argument("--warmup_epochs", type=int, default=3)

    parser.add_argument("--methyl_loss_weight", type=float, default=2.0, help="Weight for methylation status loss")

    

    parser.add_argument("--test_only", action="store_true")

    parser.add_argument("--test_model_path", type=str, default=None)

    

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    

    model = DecoupledProteinMPNN(num_encoder_layers=3, num_decoder_layers=3).to(device)



    if args.test_only:

        checkpoint = torch.load(args.test_model_path, map_location=device)

        model.load_state_dict(checkpoint['model_state_dict'])

        test_dataset = JSONLDataset(args.test_data, augment=False)

        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)

        final_evaluation(model, test_loader, device, args.output_dir)

        return



    train_dataset = JSONLDataset(args.nmethyl_data, augment=args.use_augmentation)

    # --- [关键] 使用加权采样器解决不平衡 ---

    sampler = get_weighted_sampler(train_dataset)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, sampler=sampler, collate_fn=collate_fn)

    

    print(f"Loading pretrained weights from: {args.pretrained_weights}")

    checkpoint = torch.load(args.pretrained_weights, map_location=device)

    pretrained_dict = checkpoint.get('model_state_dict', checkpoint)

    

    # 过滤不匹配的权重

    model_state = model.state_dict()

    filtered_dict = {k: v for k, v in pretrained_dict.items() if k in model_state and v.size() == model_state[k].size()}

    model.load_state_dict(filtered_dict, strict=False)

    

    # 智能初始化: Base Head 使用预训练权重, Methyl Head 随机

    with torch.no_grad():

        if 'W_out.weight' in pretrained_dict:

            natural_out_w = pretrained_dict['W_out.weight'][:20, :].clone()

            natural_out_b = pretrained_dict['W_out.bias'][:20].clone()

            model.W_out_base.weight.data = natural_out_w

            model.W_out_base.bias.data = natural_out_b



    best_loss = float('inf')

    

    # STAGE 1: 冻结主干

    if args.num_frozen_epochs > 0:

        print("\n=== STAGE 1: Training HEADS ===")

        for n, p in model.named_parameters():

            if not any(k in n for k in ['W_s', 'W_out_base', 'W_out_methyl']): p.requires_grad = False

        

        optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.learning_rate, weight_decay=args.weight_decay)

        for epoch in range(1, args.num_frozen_epochs + 1):

            loss, _ = train_one_epoch(epoch, model, train_loader, optimizer, args.methyl_loss_weight, device, None, 0, 0)

            print(f"Stage 1 - Epoch {epoch}, Loss: {loss:.4f}")



    # STAGE 2: 全局微调

    print("\n=== STAGE 2: Fine-tuning ALL ===")

    for p in model.parameters(): p.requires_grad = True

    

    new_params = ['W_s', 'W_out_base', 'W_out_methyl']

    optimizer = torch.optim.AdamW([

        {'params': [p for n, p in model.named_parameters() if not any(k in n for k in new_params)], 'lr': args.learning_rate * args.backbone_lr_multiplier},

        {'params': [p for n, p in model.named_parameters() if any(k in n for k in new_params)], 'lr': args.learning_rate}

    ], weight_decay=args.weight_decay)



    scheduler = None

    if args.use_warmup_scheduler:

        scheduler = CosineAnnealingLR(optimizer, T_max=(args.num_epochs - args.num_frozen_epochs) * len(train_loader))

    

    patience, current_step = 0, 0

    for epoch in range(args.num_frozen_epochs + 1, args.num_epochs + 1):

        avg_loss, current_step = train_one_epoch(epoch, model, train_loader, optimizer, args.methyl_loss_weight, device, scheduler, args.warmup_epochs * len(train_loader), current_step)

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

    # 测试逻辑... (同上)

    if args.test_data:

         model.load_state_dict(torch.load(os.path.join(args.output_dir, "best_model.pt"))['model_state_dict'])

         test_dataset = JSONLDataset(args.test_data, augment=False)

         test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)

         final_evaluation(model, test_loader, device, args.output_dir)



if __name__ == "__main__":

    main()
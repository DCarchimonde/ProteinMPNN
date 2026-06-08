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
        if self.reduction == 'mean': return focal_loss.mean()
        elif self.reduction == 'sum': return focal_loss.sum()
        else: return focal_loss

# =============================================================================
# 2. 模型定义 (Decoupled: 双头设计)
# =============================================================================
class DecoupledProteinMPNN(ProteinMPNN):
    def __init__(self, hidden_dim=128, augment_eps=0.2, **kwargs):
        super().__init__(
            num_letters=21, 
            hidden_dim=hidden_dim, 
            vocab=21, 
            k_neighbors=48, 
            augment_eps=augment_eps, 
            **kwargs
        )
        # 覆盖父类的 W_s，扩展到 40+ 维度
        self.W_s = nn.Embedding(len(EXTENDED_AA_ALPHABET), hidden_dim)
        # 头1：预测基础氨基酸 (20类)
        self.W_out_base = nn.Linear(hidden_dim, len(NATURAL_AA_ALPHABET))
        # 头2：预测是否甲基化 (2类)
        self.W_out_methyl = nn.Linear(hidden_dim, 2)

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
    
    # 1. 映射到基础氨基酸标签 (0-19)
    base_targets = targets_flat.clone()
    offset = len(NATURAL_AA_ALPHABET)
    for methyl_rel, natural_idx in NMETHYL_TO_NATURAL_MAPPING.items():
        base_targets[base_targets == (methyl_rel + offset)] = natural_idx
    base_targets[base_targets >= len(NATURAL_AA_ALPHABET)] = -100
    
    # 2. 甲基化标签 (0/1)
    methyl_targets = (targets_flat >= len(NATURAL_AA_ALPHABET)).long()
    
    # Loss 1: 基础分类 (CrossEntropy)
    loss_base = F.cross_entropy(logits_base, base_targets, label_smoothing=0.1, ignore_index=-100)
    
    # Loss 2: 甲基化分类 (Focal Loss)
    ce_loss_methyl = F.cross_entropy(logits_methyl, methyl_targets, reduction='none', weight=torch.tensor([1.0, 5.0], device=logits_base.device))
    pt = torch.exp(-ce_loss_methyl)
    loss_methyl_focal = ((1 - pt) ** 2.0) * ce_loss_methyl
    loss_methyl = loss_methyl_focal.mean()
    
    return loss_base + methyl_weight * loss_methyl, base_targets.numel()

# =============================================================================
# 4. 数据处理与特征提取
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

def collate_fn(batch): return batch

def featurize_batch(batch, device):
    B = len(batch)
    batch = [b for b in batch if 'seq_chain_A' in b and len(b['seq_chain_A']) > 0]
    if not batch: return None
    
    lengths = []
    for b in batch:
        l = 0
        for k in b.keys():
            if k.startswith('seq_chain_'): l += len(b[k])
        lengths.append(l)
    L_max = max(lengths)
    
    X = np.zeros([B, L_max, 4, 3])
    S = np.zeros([B, L_max], dtype=np.int32)
    mask = np.zeros([B, L_max], dtype=np.float32)
    chain_M = np.zeros([B, L_max], dtype=np.float32)
    residue_idx = -100 * np.ones([B, L_max], dtype=np.int32)
    chain_encoding_all = np.zeros([B, L_max], dtype=np.int32)
    
    for i, b in enumerate(batch):
        all_chains = []
        for k in b.keys():
            if k.startswith('seq_chain_'):
                all_chains.append(k.split('_')[-1])
        all_chains = sorted(list(set(all_chains)))
        
        l_p = 0
        for c_i, c_id in enumerate(all_chains):
            seq = b.get(f'seq_chain_{c_id}', '')
            N = np.array(b.get(f'N_chain_{c_id}', []))
            CA = np.array(b.get(f'CA_chain_{c_id}', []))
            C = np.array(b.get(f'C_chain_{c_id}', []))
            O = np.array(b.get(f'O_chain_{c_id}', []))
            l = min(len(seq), len(CA))
            if l == 0: continue
            
            X[i, l_p:l_p+l, 0, :] = N[:l]
            X[i, l_p:l_p+l, 1, :] = CA[:l]
            X[i, l_p:l_p+l, 2, :] = C[:l]
            X[i, l_p:l_p+l, 3, :] = O[:l]
            
            indices = []
            for aa in seq[:l]:
                idx = EXTENDED_AA_TO_INDEX.get(aa, EXTENDED_AA_TO_INDEX['X'])
                indices.append(idx)
            S[i, l_p:l_p+l] = indices
            
            chain_M[i, l_p:l_p+l] = 1.0 
            residue_idx[i, l_p:l_p+l] = np.arange(l) + c_i * 100
            chain_encoding_all[i, l_p:l_p+l] = c_i
            l_p += l
            
    isnan = np.isnan(X); mask = np.isfinite(np.sum(X, (2,3))).astype(np.float32); X[isnan] = 0.
    return [torch.from_numpy(t).to(dtype=torch.long if t.dtype==np.int32 else torch.float32, device=device) for t in [X, S, mask, chain_M, residue_idx, chain_encoding_all]]

# =============================================================================
# 5. 最终评估函数 (含 Base AA 测算)
# =============================================================================
def final_evaluation(model, loader, device):
    model.eval()
    print("\n" + "="*50)
    print("📊 FINAL EVALUATION REPORT (DECOUPLED LOGIC)")
    print("="*50)
    
    all_preds_base, all_targets_base = [], []
    all_preds_methyl, all_targets_methyl = [], []
    all_preds_combined, all_targets_combined = [], []
    
    nat_to_me_abs = {v: k + len(NATURAL_AA_ALPHABET) for k, v in NMETHYL_TO_NATURAL_MAPPING.items()}
    offset = len(NATURAL_AA_ALPHABET)

    with torch.no_grad():
        for batch in loader:
            f = featurize_batch(batch, device)
            if f is None: continue
            X, S, mask, chain_M, residue_idx, chain_encoding_all = f
            
            lb, lm = model(X, S, mask, chain_M, residue_idx, chain_encoding_all)
            
            # --- 预测解析 ---
            pred_base_idx = torch.argmax(lb, -1) 
            probs_methyl = F.softmax(lm, dim=-1)[:, :, 1]
            pred_is_methyl = (probs_methyl > 0.3).long() # 阈值 0.3
            
            # 组合最终结果 (端到端 41类)
            final_pred = pred_base_idx.clone()
            for n_idx, m_idx in nat_to_me_abs.items():
                mask_update = (pred_is_methyl == 1) & (pred_base_idx == n_idx)
                final_pred[mask_update] = m_idx
            
            # --- 标签解析 ---
            targets = S.cpu().numpy().flatten()
            preds_b = pred_base_idx.cpu().numpy().flatten()
            preds_m = pred_is_methyl.cpu().numpy().flatten()
            preds_final = final_pred.cpu().numpy().flatten()
            mask_flat = mask.cpu().numpy().flatten().astype(bool)
            
            valid = mask_flat & (targets != EXTENDED_AA_TO_INDEX.get('X', -1))
            
            batch_targets = targets[valid]
            
            # 基础目标 (Base Targets)
            batch_targets_base = batch_targets.copy()
            for m_rel, n_idx in NMETHYL_TO_NATURAL_MAPPING.items():
                batch_targets_base[batch_targets_base == (m_rel + offset)] = n_idx
            
            # 甲基化目标 (Methyl Targets)
            batch_targets_methyl = (batch_targets >= offset).astype(int)
            
            all_targets_base.extend(batch_targets_base)
            all_preds_base.extend(preds_b[valid])
            
            all_targets_methyl.extend(batch_targets_methyl)
            all_preds_methyl.extend(preds_m[valid])
            
            all_targets_combined.extend(batch_targets)
            all_preds_combined.extend(preds_final[valid])

    # --- 计算指标 ---
    all_targets_base = np.array(all_targets_base)
    all_preds_base = np.array(all_preds_base)
    all_targets_methyl = np.array(all_targets_methyl)
    all_preds_methyl = np.array(all_preds_methyl)
    all_targets_combined = np.array(all_targets_combined)
    all_preds_combined = np.array(all_preds_combined)
    
    base_acc = np.mean(all_targets_base == all_preds_base)
    methyl_acc = np.mean(all_targets_methyl == all_preds_methyl)
    total_acc = np.mean(all_targets_combined == all_preds_combined)
    
    print(f"\n✅ Base AA Accuracy (Ignoring Methylation): {base_acc*100:.2f}%")
    print(f"   (不考虑甲基化，只看氨基酸种类是否正确)")
    print(f"\n✅ Methylation Detection Accuracy: {methyl_acc*100:.2f}%")
    print(f"   (甲基化状态判别准确率)")
    print(f"\n🔥 Total End-to-End Accuracy: {total_acc*100:.2f}%")
    print(f"   (严格全匹配准确率)")
    print("="*50)

# =============================================================================
# 6. 运行入口 (含权重修复逻辑)
# =============================================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrained_weights", type=str, required=True)
    parser.add_argument("--nmethyl_data", type=str, required=True)
    parser.add_argument("--test_data", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="./run_decoupled_final")
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--methyl_loss_weight", type=float, default=10.0)
    parser.add_argument("--lr", type=float, default=2e-4)
    args = parser.parse_args()
    
    set_seed(42)
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = DecoupledProteinMPNN(augment_eps=0.2).to(device)
    
    print(f"Loading weights from: {args.pretrained_weights}")
    ckpt = torch.load(args.pretrained_weights, map_location=device)
    state_dict = ckpt.get('model_state_dict', ckpt)
    
    # -----------------------------------------------------------
    # 🔧 关键修复：W_s 维度不匹配处理 (21 vs 40)
    # -----------------------------------------------------------
    if 'W_s.weight' in state_dict:
        pt_W_s = state_dict['W_s.weight'] # [21, 128]
        # 只复制前21行 (天然氨基酸)，剩下的19行 (甲基化) 保持随机初始化
        model.W_s.weight.data[:21, :] = pt_W_s[:21, :]
        # 从字典中删除 W_s，避免 load_state_dict 报错
        del state_dict['W_s.weight']
        print(" [Info] Solved W_s mismatch: Copied first 21 embeddings, kept rest random.")
    
    model.load_state_dict(state_dict, strict=False) 

    train_ds = JSONLDataset(args.nmethyl_data)
    sampler = get_weighted_sampler(train_ds, oversample_weight=50.0)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, sampler=sampler, collate_fn=collate_fn)
    test_ds = JSONLDataset(args.test_data)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10, verbose=True)

    print(">>> Starting Decoupled Training...")
    best_loss = float('inf')
    
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss, n_steps = 0, 0
        for batch in train_loader:
            f = featurize_batch(batch, device)
            if f is None: continue
            optimizer.zero_grad()
            X, S, mask, chain_M, residue_idx, chain_encoding_all = f
            
            lb, lm = model(X, S, mask, chain_M, residue_idx, chain_encoding_all)
            loss, valid = calculate_aggressive_loss(lb, lm, S, mask, methyl_weight=args.methyl_loss_weight)
            
            if valid > 0 and torch.isfinite(loss):
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                total_loss += loss.item()
                n_steps += 1
        
        avg_loss = total_loss / n_steps if n_steps > 0 else 0
        scheduler.step(avg_loss)
        
        if epoch % 10 == 0:
            print(f"Epoch {epoch}: Loss = {avg_loss:.4f}")

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save({'model_state_dict': model.state_dict()}, os.path.join(args.output_dir, "best_model_decoupled.pt"))

    print(">>> Loading best model for final evaluation...")
    model.load_state_dict(torch.load(os.path.join(args.output_dir, "best_model_decoupled.pt"))['model_state_dict'])
    final_evaluation(model, test_loader, device)

if __name__ == "__main__":
    main()
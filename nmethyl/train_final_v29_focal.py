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

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def set_strict_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)

try:
    from model_utils import ProteinMPNN, gather_nodes, cat_neighbors_nodes
    from nmethyl.utils.nmethyl_config import (
        EXTENDED_AA_ALPHABET, NATURAL_AA_ALPHABET,
        NMETHYL_TO_NATURAL_MAPPING, EXTENDED_AA_TO_INDEX, METHYL_AA_ALPHABET
    )
except ImportError as e:
    print(f"Error: Missing project files. {e}")
    sys.exit(1)

# =============================================================================
# 🔥 核心大杀器：Focal Loss
# =============================================================================
class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1-pt)**self.gamma * BCE_loss
        if self.reduction == 'mean':
            return torch.mean(F_loss)
        elif self.reduction == 'sum':
            return torch.sum(F_loss)
        else:
            return F_loss

# =============================================================================
# V29 Model: Hybrid (MLP Base + 20 Experts)
# =============================================================================
class HybridProteinMPNN(ProteinMPNN):
    def __init__(self, hidden_dim=128, augment_eps=0.1, **kwargs):
        super().__init__(
            num_letters=21, 
            hidden_dim=hidden_dim, 
            vocab=21, 
            k_neighbors=48, 
            augment_eps=augment_eps, 
            **kwargs
        )
        self.W_s = nn.Embedding(len(EXTENDED_AA_ALPHABET), hidden_dim)
        
        # 1. 强力 Base Head (来自 V28)
        self.W_out_base = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, len(NATURAL_AA_ALPHABET))
        )
        
        # 2. 20 Experts (来自 V27)
        self.W_out_experts = nn.Linear(hidden_dim, len(NATURAL_AA_ALPHABET))

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
        decoding_order = torch.argsort(chain_M + 0.0001)
        if self.training:
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
        
        # Outputs
        logits_base = self.W_out_base(h_V)
        logits_experts = self.W_out_experts(h_V)
        
        return logits_base, logits_experts

def smart_load_weights(model, pretrained_path, device):
    print(f"\n>>> [V29 Init] Loading {pretrained_path}...")
    ckpt = torch.load(pretrained_path, map_location=device)
    state_dict = ckpt.get('model_state_dict', ckpt)
    model_state = model.state_dict()
    load_dict = {}
    for k, v in state_dict.items():
        clean_k = k.replace("module.", "")
        
        # 适配 V28 的 MLP 结构 (因为我们这次用 V20 做底子，所以需要特殊处理)
        # 如果预训练权重是 Linear，我们把它放到 MLP 的最后一层
        if (clean_k == "W_out_base.weight" or clean_k == "W_out.weight"):
             target_k = "W_out_base.3.weight" # Sequential last layer
             if target_k in model_state: load_dict[target_k] = v[:20, :]
             continue
        if (clean_k == "W_out_base.bias" or clean_k == "W_out.bias"):
             target_k = "W_out_base.3.bias"
             if target_k in model_state: load_dict[target_k] = v[:20]
             continue
             
        if clean_k in model_state and v.shape == model_state[clean_k].shape:
            load_dict[clean_k] = v
            
    model.load_state_dict(load_dict, strict=False)
    
    # Embedding trick
    ws_file = state_dict.get('W_s.weight', state_dict.get('module.W_s.weight'))
    if ws_file is not None:
        with torch.no_grad():
            for i, char in enumerate(NATURAL_AA_ALPHABET):
                if char in EXTENDED_AA_TO_INDEX:
                    model.W_s.weight.data[EXTENDED_AA_TO_INDEX[char]].copy_(ws_file[i])
            for nme_idx, nat_idx in NMETHYL_TO_NATURAL_MAPPING.items():
                if nme_idx < len(METHYL_AA_ALPHABET):
                    nme_char = METHYL_AA_ALPHABET[nme_idx]
                    nat_char = NATURAL_AA_ALPHABET[nat_idx]
                    model.W_s.weight.data[EXTENDED_AA_TO_INDEX[nme_char]].copy_(
                        model.W_s.weight.data[EXTENDED_AA_TO_INDEX[nat_char]]
                    )
    print("✅ V29 Hybrid+Focal System Ready.")

# =============================================================================
# Loss Calculation (Using Focal Loss)
# =============================================================================
focal_criterion = None # Will init in main

def calculate_loss_v29(logits_base, logits_experts, targets, mask, device):
    global focal_criterion
    if focal_criterion is None: focal_criterion = FocalLoss(alpha=0.75, gamma=2.0).to(device)

    mask_flat = mask.contiguous().view(-1).bool()
    if mask_flat.sum() == 0: return torch.tensor(0.0, device=device), 0
    targets_flat = targets.contiguous().view(-1)[mask_flat]
    valid_pos = targets_flat != EXTENDED_AA_TO_INDEX.get('X', -1)
    if valid_pos.sum() == 0: return torch.tensor(0.0, device=device), 0
    targets_flat = targets_flat[valid_pos]
    
    logits_base = logits_base.contiguous().view(-1, 20)[mask_flat][valid_pos]
    logits_experts = logits_experts.contiguous().view(-1, 20)[mask_flat][valid_pos]
    
    # Labels
    base_targets = targets_flat.clone()
    offset = len(NATURAL_AA_ALPHABET)
    for methyl_rel, natural_idx in NMETHYL_TO_NATURAL_MAPPING.items():
        base_targets[base_targets == (methyl_rel + offset)] = natural_idx
    is_methylated = (targets_flat >= len(NATURAL_AA_ALPHABET)).float()
    
    # 1. Base Loss (Identity) - 依然用 CrossEntropy 但加权
    loss_base = F.cross_entropy(logits_base, base_targets, label_smoothing=0.1) * 2.0
    
    # 2. Expert Loss (Methylation) - 使用 Focal Loss!
    expert_logit = torch.gather(logits_experts, 1, base_targets.unsqueeze(1)).squeeze(1)
    
    # Focal Loss 会自动放大那些 "难分类样本" (即甲基化样本) 的梯度
    loss_expert = focal_criterion(expert_logit, is_methylated) * 5.0

    return loss_base + loss_expert, base_targets.numel()

# Standard Data Utils
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
        weights.append(10.0 if has_methyl else 1.0)
    return WeightedRandomSampler(weights, len(weights), replacement=True)

def collate_fn(batch): return batch

def featurize_batch(batch, device):
    B = len(batch)
    batch = [b for b in batch if 'seq_chain_A' in b and len(b['seq_chain_A']) > 0]
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
            for atom_idx, atom_name in enumerate(['N', 'CA', 'C', 'O']):
                 coords = np.array(b.get(f'{atom_name}_chain_{c_id}', []))
                 l = min(len(seq), len(coords))
                 if l > 0: X[i, l_p:l_p+l, atom_idx, :] = coords[:l]
            indices = [EXTENDED_AA_TO_INDEX.get(aa, EXTENDED_AA_TO_INDEX['X']) for aa in seq]
            l = len(indices)
            S[i, l_p:l_p+l] = indices
            if c_id in b.get('masked_list', []): chain_M[i, l_p:l_p+l] = 1.0
            residue_idx[i, l_p:l_p+l] = np.arange(l) + c_i * 100
            chain_encoding_all[i, l_p:l_p+l] = c_i
            l_p += l
    isnan = np.isnan(X); mask = np.isfinite(np.sum(X, (2,3))).astype(np.float32); X[isnan] = 0.
    return [torch.from_numpy(t).to(dtype=torch.long if t.dtype==np.int32 else torch.float32, device=device) for t in [X, S, mask, chain_M, residue_idx, chain_encoding_all]]

# =============================================================================
# Validation (With Full Report)
# =============================================================================
best_score = -1.0
def validate_and_report(model, loader, device, epoch_num, output_dir):
    global best_score
    model.eval()
    print(f"\n🔍 [Epoch {epoch_num}] Fusion Council (V29) Diagnosing...")
    
    all_targets, all_preds = [], []
    all_base_correct, all_base_total = 0, 0
    methyl_idx_to_nat_idx = {m + len(NATURAL_AA_ALPHABET): n for m, n in NMETHYL_TO_NATURAL_MAPPING.items()}
    methyl_stats = {m: {"count": 0, "correct": 0, "miss_nat": 0, "miss_other": 0} for m in METHYL_AA_ALPHABET}

    with torch.no_grad():
        for batch in loader:
            f = featurize_batch(batch, device)
            if f is None: continue
            X, S, mask, chain_M, residue_idx, chain_encoding_all = f
            l_base, l_experts = model(X, S, mask, chain_M, residue_idx, chain_encoding_all)
            
            # 1. Base Prediction
            pred_base_idx = torch.argmax(l_base, -1)
            
            # 2. Expert Consultation
            expert_logit = torch.gather(l_experts, -1, pred_base_idx.unsqueeze(-1)).squeeze(-1)
            pred_is_methyl = (torch.sigmoid(expert_logit) > 0.5).long()
            
            # 3. Combine
            final_pred = pred_base_idx.clone()
            for m_abs_idx, n_idx in methyl_idx_to_nat_idx.items():
                mask_update = (pred_is_methyl == 1) & (pred_base_idx == n_idx)
                final_pred[mask_update] = m_abs_idx

            tgts = S.cpu().numpy().flatten()
            preds = final_pred.cpu().numpy().flatten()
            mask_flat = mask.cpu().numpy().flatten().astype(bool)
            valid = mask_flat & (tgts != EXTENDED_AA_TO_INDEX.get('X', -1))
            
            # Base Acc Logic
            base_targets = tgts[valid].copy()
            base_preds = preds[valid].copy()
            for m_rel, n_idx in NMETHYL_TO_NATURAL_MAPPING.items():
                m_idx = m_rel + len(NATURAL_AA_ALPHABET)
                base_targets[base_targets == m_idx] = n_idx
                base_preds[base_preds == m_idx] = n_idx
            all_base_total += len(base_targets)
            all_base_correct += np.sum(base_targets == base_preds)

            # Detailed Stats
            for t, p in zip(tgts[valid], preds[valid]):
                if t >= len(NATURAL_AA_ALPHABET):
                     t_char = EXTENDED_AA_ALPHABET[t]
                     if t_char not in methyl_stats: continue
                     methyl_stats[t_char]["count"] += 1
                     if p == t: methyl_stats[t_char]["correct"] += 1
                     else:
                         n_idx = methyl_idx_to_nat_idx.get(t, -1)
                         if p == n_idx: methyl_stats[t_char]["miss_nat"] += 1
                         else: methyl_stats[t_char]["miss_other"] += 1
                         
            all_targets.extend(tgts[valid])
            all_preds.extend(preds[valid])

    all_targets = np.array(all_targets)
    all_preds = np.array(all_preds)
    
    methyl_mask = all_targets >= len(NATURAL_AA_ALPHABET)
    natural_mask = all_targets < len(NATURAL_AA_ALPHABET)
    
    methyl_recall = np.sum(all_preds[methyl_mask] == all_targets[methyl_mask]) / np.sum(methyl_mask) if np.sum(methyl_mask) > 0 else 0
    base_acc = all_base_correct / all_base_total if all_base_total > 0 else 0
    
    print(f"\n📊 Report Card (V29 Fusion+Focal):")
    print(f"✅ Methyl Recall: {methyl_recall*100:.2f}%")
    print("-" * 65)
    print(f"{'Methyl AA':<10} {'Count':<8} {'Recall':<10} {'Miss->Nat':<10} {'Miss->Other':<10}")
    print("-" * 65)
    for aa in sorted(methyl_stats.keys()):
        s = methyl_stats[aa]
        rec = (s['correct']/s['count']*100) if s['count'] > 0 else 0
        print(f"{aa:<10} {s['count']:<8} {rec:5.1f}%     {s['miss_nat']:<10} {s['miss_other']:<10}")
    print("-" * 65)
    
    # 综合评分：更看重 BaseAcc 了，因为 Miss->Other 是瓶颈
    score = (0.4 * base_acc) + (0.6 * methyl_recall)
    print(f"🏆 Score: {score:.4f} (BaseAcc: {base_acc:.4f}, MeRec: {methyl_recall:.4f})")
    
    if score > best_score:
        best_score = score
        torch.save(model.state_dict(), os.path.join(output_dir, f"best_model_v29_{score:.4f}.pt"))
        print("🌟 New Best Model Saved!")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrained_weights", type=str, required=True)
    parser.add_argument("--nmethyl_data", type=str, required=True)
    parser.add_argument("--test_data", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="./run_v29_focal")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=8)
    args = parser.parse_args()
    
    set_strict_seed(42)
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = HybridProteinMPNN(augment_eps=0.1).to(device)
    smart_load_weights(model, args.pretrained_weights, device)

    train_ds = JSONLDataset(args.nmethyl_data)
    sampler = get_weighted_sampler(train_ds)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, sampler=sampler, collate_fn=collate_fn)
    test_ds = JSONLDataset(args.test_data)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    
    print("🚀 Starting V29 Fusion Run (MLP Base + 20 Experts + Focal Loss)...")
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0
        steps = 0
        for batch in train_loader:
            f = featurize_batch(batch, device)
            if f is None: continue
            optimizer.zero_grad()
            X, S, mask, chain_M, residue_idx, chain_encoding_all = f
            l_base, l_experts = model(X, S, mask, chain_M, residue_idx, chain_encoding_all)
            loss, _ = calculate_loss_v29(l_base, l_experts, S, mask, device)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()
            steps += 1
        
        print(f"Epoch {epoch}: Loss = {total_loss/steps:.4f}", end="\r")
        if epoch % 5 == 0:
            validate_and_report(model, test_loader, device, epoch, args.output_dir)

if __name__ == "__main__":
    main()
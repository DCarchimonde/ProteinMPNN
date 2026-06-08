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
import copy
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from sklearn.metrics import classification_report
from torch.optim.lr_scheduler import CosineAnnealingLR

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
# 1. 解耦模型 (Student)
# =============================================================================
class DecoupledProteinMPNN(ProteinMPNN):
    def __init__(self, hidden_dim=128, **kwargs):
        super().__init__(num_letters=21, hidden_dim=hidden_dim, vocab=21, k_neighbors=48, **kwargs)
        self.W_s = nn.Embedding(len(EXTENDED_AA_ALPHABET), hidden_dim)
        self.W_out_base = nn.Linear(hidden_dim, len(NATURAL_AA_ALPHABET))
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
# 2. 权重加载 (Student & Teacher)
# =============================================================================
def load_weights_for_student(model, pretrained_path, device):
    print(f"  [Student] Loading weights from {pretrained_path}...")
    ckpt = torch.load(pretrained_path, map_location=device)
    state_dict = ckpt.get('model_state_dict', ckpt)
    model_state = model.state_dict()
    
    # 1. Backbone
    load_dict = {}
    for k, v in state_dict.items():
        clean_k = k.replace("module.", "")
        if clean_k in model_state and v.shape == model_state[clean_k].shape:
            load_dict[clean_k] = v
    model.load_state_dict(load_dict, strict=False)
    print(f"  ✅ Student Backbone loaded ({len(load_dict)} keys).")

    # 2. Embedding (Parent Init)
    ws_file = state_dict.get('W_s.weight', state_dict.get('module.W_s.weight'))
    if ws_file is not None:
        STD = 'ACDEFGHIKLMNPQRSTVWYX'
        with torch.no_grad():
            for i, char in enumerate(STD):
                if char in EXTENDED_AA_TO_INDEX:
                    model.W_s.weight.data[EXTENDED_AA_TO_INDEX[char]].copy_(ws_file[i])
            # N-Methyl Parents
            from nmethyl.utils.nmethyl_config import METHYL_AA_ALPHABET
            for nme_idx, nat_idx in NMETHYL_TO_NATURAL_MAPPING.items():
                if nme_idx < len(METHYL_AA_ALPHABET):
                    nme_char = METHYL_AA_ALPHABET[nme_idx]
                    nat_char = NATURAL_AA_ALPHABET[nat_idx]
                    if nme_char in EXTENDED_AA_TO_INDEX and nat_char in EXTENDED_AA_TO_INDEX:
                        parent_emb = model.W_s.weight.data[EXTENDED_AA_TO_INDEX[nat_char]]
                        model.W_s.weight.data[EXTENDED_AA_TO_INDEX[nme_char]].copy_(parent_emb)
        print("  ✅ Student Embeddings initialized (with Parent Init).")

    # 3. Head
    w_out = state_dict.get('W_out.weight', state_dict.get('module.W_out.weight'))
    b_out = state_dict.get('W_out.bias', state_dict.get('module.W_out.bias'))
    if w_out is not None:
        with torch.no_grad():
            model.W_out_base.weight.data.copy_(w_out[:20])
            model.W_out_base.bias.data.copy_(b_out[:20])
        print("  ✅ Student Base Head initialized.")

def load_weights_for_teacher(model, pretrained_path, device):
    print(f"  [Teacher] Loading weights from {pretrained_path}...")
    ckpt = torch.load(pretrained_path, map_location=device)
    state_dict = ckpt.get('model_state_dict', ckpt)
    clean_state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    model.load_state_dict(clean_state_dict, strict=True) 
    print("  ✅ Teacher loaded successfully.")

# =============================================================================
# 3. 蒸馏损失函数
# =============================================================================
def distillation_loss(student_logits, teacher_logits, targets, mask, temp=2.0, alpha=0.5):
    mask_flat = mask.view(-1).bool()
    if mask_flat.sum() == 0: return torch.tensor(0.0, device=student_logits.device)

    # Teacher (Soft Target)
    teacher_probs = F.softmax(teacher_logits[:, :, :20] / temp, dim=-1)
    teacher_probs_flat = teacher_probs.contiguous().view(-1, 20)[mask_flat]

    # Student (Log Softmax)
    student_logits_flat = student_logits.contiguous().view(-1, 20)[mask_flat]
    student_log_probs = F.log_softmax(student_logits_flat / temp, dim=-1)

    # KL Divergence
    loss_distill = F.kl_div(student_log_probs, teacher_probs_flat, reduction='batchmean') * (temp ** 2)

    # Ground Truth Loss
    targets_flat = targets.contiguous().view(-1)[mask_flat]
    x_idx = EXTENDED_AA_TO_INDEX['X']
    
    base_targets = targets_flat.clone()
    offset = len(NATURAL_AA_ALPHABET)
    for m_rel, n_idx in NMETHYL_TO_NATURAL_MAPPING.items():
        base_targets[base_targets == (m_rel + offset)] = n_idx
    
    valid_mask = targets_flat != x_idx
    if valid_mask.sum() > 0:
        loss_base_ce = F.cross_entropy(student_logits_flat[valid_mask], base_targets[valid_mask], label_smoothing=0.1)
    else:
        loss_base_ce = torch.tensor(0.0, device=student_logits.device)

    return alpha * loss_distill + (1 - alpha) * loss_base_ce

# =============================================================================
# 4. 数据与训练 Loop
# =============================================================================
class JSONLDataset(Dataset):
    def __init__(self, jsonl_file):
        self.data = []
        with open(jsonl_file, 'r') as f:
            for line in f: self.data.append(json.loads(line))
    def __len__(self): return len(self.data)
    def __getitem__(self, idx): return self.data[idx]

def get_weighted_sampler(dataset, oversample_weight=20.0):
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
    # 返回明确的6个张量
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
            N = np.array(b.get(f'N_chain_{c_id}', []))
            CA = np.array(b.get(f'CA_chain_{c_id}', []))
            C = np.array(b.get(f'C_chain_{c_id}', []))
            O = np.array(b.get(f'O_chain_{c_id}', []))
            l = min(len(seq), len(CA))
            if l == 0: continue
            X[i, l_p:l_p+l, 0, :] = N[:l]; X[i, l_p:l_p+l, 1, :] = CA[:l]
            X[i, l_p:l_p+l, 2, :] = C[:l]; X[i, l_p:l_p+l, 3, :] = O[:l]
            indices = []
            for aa in seq[:l]:
                idx = EXTENDED_AA_TO_INDEX.get(aa, EXTENDED_AA_TO_INDEX['X'])
                indices.append(idx)
            S[i, l_p:l_p+l] = indices
            if c_id in b.get('masked_list', []): chain_M[i, l_p:l_p+l] = 1.0
            residue_idx[i, l_p:l_p+l] = np.arange(l) + c_i * 100
            chain_encoding_all[i, l_p:l_p+l] = c_i
            l_p += l
            
    isnan = np.isnan(X); mask = np.isfinite(np.sum(X, (2,3))).astype(np.float32); X[isnan] = 0.
    # 1. X, 2. S, 3. mask, 4. chain_M, 5. residue_idx, 6. chain_encoding_all
    return [torch.from_numpy(t).to(dtype=torch.long if t.dtype==np.int32 else torch.float32, device=device) for t in [X, S, mask, chain_M, residue_idx, chain_encoding_all]]

def final_evaluation(model, loader, device):
    model.eval()
    print("\n=== Final Evaluation (Distilled Model) ===")
    all_preds, all_targets = [], []
    nat_to_me_abs = {v: k + len(NATURAL_AA_ALPHABET) for k, v in NMETHYL_TO_NATURAL_MAPPING.items()}
    with torch.no_grad():
        for batch in loader:
            f = featurize_batch(batch, device)
            if f is None: continue
            # 安全解包 (6个)
            X, S, mask, chain_M, residue_idx, chain_encoding_all = f[0], f[1], f[2], f[3], f[4], f[5]
            
            lb, lm = model(X, S, mask, chain_M, residue_idx, chain_encoding_all)
            base = torch.argmax(lb, -1)
            is_me = torch.argmax(lm, -1)
            final = base.clone()
            for n_idx, m_idx in nat_to_me_abs.items():
                mask_update = (is_me == 1) & (base == n_idx)
                final[mask_update] = m_idx
            tgts = S.cpu().numpy().flatten()
            preds = final.cpu().numpy().flatten()
            mask_flat = mask.cpu().numpy().flatten().astype(bool)
            valid = mask_flat & (tgts != EXTENDED_AA_TO_INDEX.get('X', -1))
            all_preds.extend(preds[valid])
            all_targets.extend(tgts[valid])
    all_preds, all_targets = np.array(all_preds), np.array(all_targets)
    if len(all_targets) == 0: return
    acc = np.mean(all_preds == all_targets)
    natural_mask = all_targets < len(NATURAL_AA_ALPHABET)
    natural_acc = np.mean(all_targets[natural_mask] == all_preds[natural_mask]) if natural_mask.sum() > 0 else 0
    labels = sorted(np.unique(np.concatenate([all_targets, all_preds])).astype(int))
    names = [EXTENDED_AA_ALPHABET[i] for i in labels if i < len(EXTENDED_AA_ALPHABET)]
    print(classification_report(all_targets, all_preds, labels=labels, target_names=names, zero_division=0))
    print(f"Overall Accuracy: {acc:.4f}")
    print(f"Natural AA Recovery: {natural_acc:.4f}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrained_weights", type=str, required=True)
    parser.add_argument("--nmethyl_data", type=str, required=True)
    parser.add_argument("--test_data", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="./final_run_v11_distill")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--methyl_weight", type=float, default=4.0)
    parser.add_argument("--distill_alpha", type=float, default=0.8)
    parser.add_argument("--distill_temp", type=float, default=2.0)
    args = parser.parse_args()
    
    set_seed(42)
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. Teacher
    print("\n>>> Initializing Teacher Model...")
    # Teacher 是标准的 MPNN, 只有21个字母
    teacher_model = ProteinMPNN(num_letters=21, vocab=21, k_neighbors=48).to(device)
    load_weights_for_teacher(teacher_model, args.pretrained_weights, device)
    teacher_model.eval() 
    for param in teacher_model.parameters(): param.requires_grad = False

    # 2. Student
    print("\n>>> Initializing Student Model...")
    student_model = DecoupledProteinMPNN().to(device)
    load_weights_for_student(student_model, args.pretrained_weights, device)

    train_ds = JSONLDataset(args.nmethyl_data)
    sampler = get_weighted_sampler(train_ds)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, sampler=sampler, collate_fn=collate_fn)
    test_ds = JSONLDataset(args.test_data)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)

    optimizer = torch.optim.AdamW([
        {'params': [p for n, p in student_model.named_parameters() if 'W_out' not in n], 'lr': 1e-4 * 0.1},
        {'params': [p for n, p in student_model.named_parameters() if 'W_out' in n], 'lr': 1e-4}
    ], weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs * len(train_loader))

    print(f"\nStarting Distillation Training (Alpha={args.distill_alpha}, Temp={args.distill_temp})...")
    best_loss = float('inf')
    for epoch in range(1, args.epochs + 1):
        student_model.train()
        total_loss, n_steps = 0, 0
        
        for batch in train_loader:
            f_student = featurize_batch(batch, device)
            if f_student is None: continue
            
            # [修复] 安全解包：明确是 6 个变量
            X, S, mask, chain_M, residue_idx, chain_encoding_all = f_student[0], f_student[1], f_student[2], f_student[3], f_student[4], f_student[5]

            # Teacher Forward Preparation
            f_teacher_S = S.clone()
            offset = len(NATURAL_AA_ALPHABET)
            for m_rel, n_idx in NMETHYL_TO_NATURAL_MAPPING.items():
                f_teacher_S[f_teacher_S == (m_rel + offset)] = n_idx
            f_teacher_S[f_teacher_S >= 21] = 20 
            
            # Teacher Forward
            with torch.no_grad():
                teacher_logits = teacher_model(X, f_teacher_S, mask, chain_M, residue_idx, chain_encoding_all)
            
            # Student Forward
            optimizer.zero_grad()
            logits_base, logits_methyl = student_model(X, S, mask, chain_M, residue_idx, chain_encoding_all)
            
            # Loss
            targets = S
            loss_base = distillation_loss(logits_base, teacher_logits, targets, mask, 
                                          temp=args.distill_temp, alpha=args.distill_alpha)
            
            mask_flat = mask.view(-1).bool()
            logits_methyl_valid = logits_methyl.contiguous().view(-1, 2)[mask_flat]
            methyl_targets = (targets >= len(NATURAL_AA_ALPHABET)).long()
            methyl_targets_valid = methyl_targets.contiguous().view(-1)[mask_flat]
            x_idx = EXTENDED_AA_TO_INDEX['X']
            targets_flat = targets.contiguous().view(-1)[mask_flat]
            valid_pos = targets_flat != x_idx
            
            if valid_pos.sum() > 0:
                loss_methyl_val = F.cross_entropy(logits_methyl_valid[valid_pos], methyl_targets_valid[valid_pos], 
                                                  weight=torch.tensor([1.0, 5.0], device=device))
            else:
                loss_methyl_val = torch.tensor(0.0, device=device)

            loss = loss_base + args.methyl_weight * loss_methyl_val
            
            if torch.isfinite(loss):
                loss.backward()
                torch.nn.utils.clip_grad_norm_(student_model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                total_loss += loss.item()
                n_steps += 1
                
        avg_loss = total_loss / n_steps if n_steps > 0 else 0
        if epoch % 10 == 0: print(f"Epoch {epoch}: Loss = {avg_loss:.4f}")
        
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save({'model_state_dict': student_model.state_dict()}, os.path.join(args.output_dir, "best_model.pt"))

    student_model.load_state_dict(torch.load(os.path.join(args.output_dir, "best_model.pt"))['model_state_dict'])
    final_evaluation(student_model, test_loader, device)

if __name__ == "__main__":
    main()
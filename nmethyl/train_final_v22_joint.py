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

# --- 系统路径 ---
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# =============================================================================
# 0. 随机种子
# =============================================================================
def set_strict_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)
    print(f"[System] 🔒 Random Seed STRICTLY locked to: {seed}")

# 物理铁律：氢键距离阈值
HBOND_MAX_DIST = 3.5 

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
# 1. 物理辅助函数
# =============================================================================
def compute_hbond_mask(X, mask):
    # 计算 N...O 距离，判断是否不可能发生甲基化
    B, L, _, _ = X.shape
    N_atoms = X[:, :, 0, :] 
    O_atoms = X[:, :, 3, :]
    dists = torch.cdist(N_atoms, O_atoms)
    diag_mask = torch.eye(L, device=X.device).unsqueeze(0).expand(B, -1, -1)
    dists = dists.masked_fill(diag_mask > 0.5, 999.9)
    min_dist_per_N, _ = torch.min(dists, dim=-1)
    has_hbond = min_dist_per_N < HBOND_MAX_DIST
    return has_hbond & (mask > 0.5)

# =============================================================================
# 2. 模型定义 (V22: Joint Probability Experts)
# =============================================================================
class JointExpertProteinMPNN(ProteinMPNN):
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
        
        # HEAD 1: Base Identity (你是谁？) -> 输出 20 个 Logits
        self.W_out_base = nn.Linear(hidden_dim, len(NATURAL_AA_ALPHABET))
        
        # HEAD 2: Experts (如果你是你，你甲基化了吗？) 
        # 我们用一个单独的层输出 20 个值，每个值代表对应氨基酸的甲基化倾向 Logit
        self.W_out_experts = nn.Linear(hidden_dim, len(NATURAL_AA_ALPHABET))

    def forward(self, X, S, mask, chain_M, residue_idx, chain_encoding_all):
        # --- Standard Encoder / Decoder ---
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
        
        # --- V22 Outputs ---
        # 1. Base Identity Logits [B, L, 20]
        logits_base = self.W_out_base(h_V) 
        
        # 2. Expert Methylation Logits [B, L, 20]
        # (Meaning: Logit(Methylated) given it is AA_i)
        logits_methyl = self.W_out_experts(h_V)
        
        return logits_base, logits_methyl

# =============================================================================
# 3. 智能初始化
# =============================================================================
def smart_load_weights(model, pretrained_path, device):
    print(f"\n>>> [V22 Init] Loading {pretrained_path}...")
    ckpt = torch.load(pretrained_path, map_location=device)
    state_dict = ckpt.get('model_state_dict', ckpt)
    model_state = model.state_dict()
    load_dict = {}
    for k, v in state_dict.items():
        clean_k = k.replace("module.", "")
        if clean_k == "W_out_base.weight" and "W_out.weight" in state_dict:
             load_dict[clean_k] = state_dict["W_out.weight"][:20, :]
             continue
        if clean_k == "W_out_base.bias" and "W_out.bias" in state_dict:
             load_dict[clean_k] = state_dict["W_out.bias"][:20]
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
                    nat_emb_idx = EXTENDED_AA_TO_INDEX[nat_char]
                    nme_emb_idx = EXTENDED_AA_TO_INDEX[nme_char]
                    parent_emb = model.W_s.weight.data[nat_emb_idx]
                    model.W_s.weight.data[nme_emb_idx].copy_(parent_emb + torch.randn_like(parent_emb)*0.01)
    print("✅ V22 Joint Probability System Ready.")

# =============================================================================
# 4. 训练 Loss (Joint Training)
# =============================================================================
def calculate_loss_v22(logits_base, logits_methyl, targets, mask):
    mask_flat = mask.contiguous().view(-1).bool()
    if mask_flat.sum() == 0: return torch.tensor(0.0, device=logits_base.device), 0
    targets_flat = targets.contiguous().view(-1)[mask_flat]
    valid_pos = targets_flat != EXTENDED_AA_TO_INDEX.get('X', -1)
    if valid_pos.sum() == 0: return torch.tensor(0.0, device=logits_base.device), 0
    targets_flat = targets_flat[valid_pos]
    
    # [N, 20]
    logits_base = logits_base.contiguous().view(-1, 20)[mask_flat][valid_pos]
    logits_methyl = logits_methyl.contiguous().view(-1, 20)[mask_flat][valid_pos]
    
    # --- Prepare Labels ---
    # 1. Base Identity Label (0-19)
    base_targets = targets_flat.clone()
    offset = len(NATURAL_AA_ALPHABET)
    for methyl_rel, natural_idx in NMETHYL_TO_NATURAL_MAPPING.items():
        base_targets[base_targets == (methyl_rel + offset)] = natural_idx
    
    # 2. Methylation Status (0 or 1)
    is_methylated = (targets_flat >= len(NATURAL_AA_ALPHABET)).float()
    
    # --- Loss 1: Identity (Cross Entropy) ---
    base_targets_clamped = base_targets.clone()
    base_targets_clamped[base_targets_clamped >= 20] = -100
    loss_base = F.cross_entropy(logits_base, base_targets_clamped, ignore_index=-100)
    
    # --- Loss 2: Expert Methylation Prediction ---
    # 我们只训练 Truth Identity 对应的那个专家！
    # 如果 Truth 是 A (无论是否甲基化)，我们就只惩罚 Expert_A
    valid_expert_mask = base_targets_clamped != -100
    
    if valid_expert_mask.sum() > 0:
        # 取出 Truth 对应的 Expert 的 Logit
        target_expert_logits = torch.gather(
            logits_methyl[valid_expert_mask], 
            1, 
            base_targets_clamped[valid_expert_mask].unsqueeze(1)
        ).squeeze(1)
        
        target_expert_labels = is_methylated[valid_expert_mask]
        
        # 使用 BCE Loss (甲基化很难，加大权重)
        loss_expert = F.binary_cross_entropy_with_logits(
            target_expert_logits,
            target_expert_labels,
            pos_weight=torch.tensor([5.0], device=logits_base.device)
        )
    else:
        loss_expert = torch.tensor(0.0, device=logits_base.device)

    # 平衡 Loss: 
    # Base 很容易学 (Loss低)，Expert 很难学。
    # 我们希望 Expert 能够在这里学到精髓。
    return loss_base + 3.0 * loss_expert, base_targets.numel()

# =============================================================================
# 5. 数据处理
# =============================================================================
class JSONLDataset(Dataset):
    def __init__(self, jsonl_file):
        self.data = []
        with open(jsonl_file, 'r') as f:
            for line in f: self.data.append(json.loads(line))
    def __len__(self): return len(self.data)
    def __getitem__(self, idx): return self.data[idx]

def get_weighted_sampler(dataset, oversample_weight=10.0):
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
    return [torch.from_numpy(t).to(dtype=torch.long if t.dtype==np.int32 else torch.float32, device=device) for t in [X, S, mask, chain_M, residue_idx, chain_encoding_all]]

# =============================================================================
# 6. 验证函数 (V22: 联合概率推理)
# =============================================================================
def validate_and_report(model, loader, device, epoch_num):
    model.eval()
    print(f"\n🔍 [Epoch {epoch_num}] Joint-Council Diagnosing...")
    
    all_targets, all_preds = [], []
    methyl_idx_to_nat_idx = {m + len(NATURAL_AA_ALPHABET): n for m, n in NMETHYL_TO_NATURAL_MAPPING.items()}
    nat_idx_to_methyl_idx = {n: m + len(NATURAL_AA_ALPHABET) for m, n in NMETHYL_TO_NATURAL_MAPPING.items()}

    with torch.no_grad():
        for batch in loader:
            f = featurize_batch(batch, device)
            if f is None: continue
            X, S, mask, chain_M, residue_idx, chain_encoding_all = f
            
            # Get Logits
            l_base, l_methyl = model(X, S, mask, chain_M, residue_idx, chain_encoding_all)
            
            # --- V22 Logic: Construct Full Probability Space ---
            # l_base: [B, L, 20] (Identity Logits)
            # l_methyl: [B, L, 20] (Methylation 'Yes' Logits for each AA)
            
            # 1. 物理过滤 (Hydrogen Bond)
            # 如果有氢键，force methyl_logit to -inf
            has_h_bond = compute_hbond_mask(X, mask) # [B, L]
            # 扩展到 [B, L, 20]
            h_bond_penalty = torch.zeros_like(l_methyl)
            h_bond_penalty[has_h_bond] = -1e9
            
            l_methyl_safe = l_methyl + h_bond_penalty
            
            # 2. 计算联合分数 (Simply adding logits works like multiplying probabilities)
            # Score(Nat_X) = Base_Logit(X) + Logit_Prob(Not_Methyl) 
            #              ≈ Base_Logit(X) - l_methyl (Simplified heuristic)
            # Score(Met_X) = Base_Logit(X) + l_methyl
            
            # 为了严谨，我们计算 LogSoftmax
            log_prob_id = F.log_softmax(l_base, dim=-1) # [B, L, 20]
            prob_methyl = torch.sigmoid(l_methyl_safe)  # [B, L, 20]
            log_p_yes = torch.log(prob_methyl + 1e-9)
            log_p_no = torch.log(1 - prob_methyl + 1e-9)
            
            # 构建最终 40 类分数
            # Natural [0-19]
            score_nat = log_prob_id + log_p_no
            # Methyl [20-39] (Only mapped ones exist, others are -inf)
            score_met = torch.full_like(score_nat, -1e9)
            
            for nat_idx, met_abs_idx in nat_idx_to_methyl_idx.items():
                # Methyl Score = Identity Score + Methyl_Yes Score
                score_met[:, :, nat_idx] = log_prob_id[:, :, nat_idx] + log_p_yes[:, :, nat_idx]
            
            # 比较 Natural vs Methyl
            # 我们需要把 score_met 映射回 absolute indices 才能做 argmax
            # 这里做一个简化的做法：直接比较 score_nat[i] 和 score_met[i]
            
            final_pred = torch.zeros_like(S)
            
            for i in range(20):
                s_n = score_nat[:, :, i]
                s_m = score_met[:, :, i] # This is the score for Methyl-version of AA i
                
                # 如果这个氨基酸没有甲基化版本（比如 Gly可能没有），s_m 应该是极小值
                if i not in nat_idx_to_methyl_idx:
                    s_m = -1e9
                
                # 这是一个局部比较，但我们需要全局比较。
                # 让我们构建一个 [B, L, 40] 的大表
                pass 
            
            # 更简单的实现：
            # 先拿到 Top 1 Identity
            # 然后看该 Identity 的 Methyl 分数是否足以翻盘？
            # 不，这又回到了 V21。我们必须构建 [B, L, ~35] 的 Tensor。
            
            full_scores = torch.full((S.shape[0], S.shape[1], 45), -1e9, device=S.device)
            
            # Fill Natural Scores (0-19)
            full_scores[:, :, 0:20] = score_nat
            
            # Fill Methyl Scores (at their absolute indices)
            for nat_idx, met_abs_idx in nat_idx_to_methyl_idx.items():
                 full_scores[:, :, met_abs_idx] = score_met[:, :, nat_idx]
            
            # Global Argmax
            final_pred = torch.argmax(full_scores, dim=-1)
            
            # Collect
            tgts = S.cpu().numpy().flatten()
            preds = final_pred.cpu().numpy().flatten()
            mask_flat = mask.cpu().numpy().flatten().astype(bool)
            valid = mask_flat & (tgts != EXTENDED_AA_TO_INDEX.get('X', -1))
            all_targets.extend(tgts[valid])
            all_preds.extend(preds[valid])

    all_targets = np.array(all_targets)
    all_preds = np.array(all_preds)

    methyl_mask = all_targets >= len(NATURAL_AA_ALPHABET)
    natural_mask = all_targets < len(NATURAL_AA_ALPHABET)
    
    methyl_correct = np.sum(all_preds[methyl_mask] == all_targets[methyl_mask])
    methyl_total = np.sum(methyl_mask)
    methyl_recall = methyl_correct / methyl_total if methyl_total > 0 else 0.0
    
    nat_preds_as_methyl = np.sum(all_preds[natural_mask] >= len(NATURAL_AA_ALPHABET))
    nat_total = np.sum(natural_mask)
    false_positive_rate = nat_preds_as_methyl / nat_total if nat_total > 0 else 0.0

    print(f"\n📊 --- REPORT CARD (Epoch {epoch_num}) ---")
    print(f"✅ Methyl Recall:    {methyl_recall*100:.2f}%  ({methyl_correct}/{methyl_total})")
    print(f"⚠️ False Alarm Rate: {false_positive_rate*100:.2f}%")
    
    print("\n🧐 Detailed Joint-Probability Report:")
    print(f"{'Methyl AA':<15} {'Count':<8} {'Recall':<10} {'Miss->Nat':<10} {'Miss->Other'}")
    print("-" * 60)
    for m_char in METHYL_AA_ALPHABET:
        m_idx = EXTENDED_AA_TO_INDEX[m_char]
        if m_idx not in methyl_idx_to_nat_idx: continue
        n_idx = methyl_idx_to_nat_idx[m_idx]
        indices = np.where(all_targets == m_idx)[0]
        count = len(indices)
        if count == 0: continue
        correct = np.sum(all_preds[indices] == m_idx)
        mis_as_nat = np.sum(all_preds[indices] == n_idx)
        mis_other = count - correct - mis_as_nat
        rec = correct / count * 100
        print(f"{m_char:<15} {count:<8} {rec:5.1f}%     {mis_as_nat:<10} {mis_other}")
    print("-" * 60)
    
    base_acc = np.mean(all_preds[natural_mask] == all_targets[natural_mask]) if nat_total > 0 else 0
    composite_score = (0.3 * base_acc) + (0.7 * methyl_recall)
    print(f"🏆 Joint Score: {composite_score:.4f} (BaseAcc: {base_acc:.4f}, MeRec: {methyl_recall:.4f})")
    return composite_score

# =============================================================================
# 7. 主循环
# =============================================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrained_weights", type=str, required=True)
    parser.add_argument("--nmethyl_data", type=str, required=True)
    parser.add_argument("--test_data", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="./run_v22_joint")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=8)
    args = parser.parse_args()
    
    set_strict_seed(42)
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = JointExpertProteinMPNN(augment_eps=0.1).to(device)
    smart_load_weights(model, args.pretrained_weights, device)

    train_ds = JSONLDataset(args.nmethyl_data)
    sampler = get_weighted_sampler(train_ds, oversample_weight=10.0)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, sampler=sampler, collate_fn=collate_fn)
    test_ds = JSONLDataset(args.test_data)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=5e-4, epochs=args.epochs, steps_per_epoch=len(train_loader), pct_start=0.2)

    print("🚀 Starting V22: Joint Probability Expert System...")
    best_score = 0.0
    
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0
        steps = 0
        for batch in train_loader:
            f = featurize_batch(batch, device)
            if f is None: continue
            optimizer.zero_grad()
            X, S, mask, chain_M, residue_idx, chain_encoding_all = f
            l_base, l_methyl = model(X, S, mask, chain_M, residue_idx, chain_encoding_all)
            
            loss, valid = calculate_loss_v22(l_base, l_methyl, S, mask)
            
            if valid > 0:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                total_loss += loss.item()
                steps += 1
        
        avg_loss = total_loss / steps if steps > 0 else 0
        print(f"Epoch {epoch}: Loss = {avg_loss:.4f}", end="\r")

        if epoch % 5 == 0:
            current_score = validate_and_report(model, test_loader, device, epoch)
            if current_score > best_score:
                best_score = current_score
                save_path = os.path.join(args.output_dir, "best_model_v22.pt")
                torch.save({'model_state_dict': model.state_dict()}, save_path)
                print(f"🌟 New Best Model Saved! (Score: {best_score:.4f})")

if __name__ == "__main__":
    main()
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
# 0. 随机种子锁定
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
# 1. 模型定义 (V19: Soft Mixture of Experts)
# =============================================================================
class SoftMoEProteinMPNN(ProteinMPNN):
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
        
        # 1. 门控网络 (Gating Network) / 身份识别
        # 它输出的是每个专家的权重 (即它是某种氨基酸的概率)
        self.W_out_gate = nn.Linear(hidden_dim, len(NATURAL_AA_ALPHABET))
        
        # 2. 20个独立专家 (The Council)
        # 每个专家只输出 1 个值 (Logit for Methylation)
        # 以前是输出2个做分类，现在为了方便加权，我们输出1个值，通过 Sigmoid 变概率
        self.methyl_experts = nn.ModuleList([
            nn.Linear(hidden_dim, 1) for _ in range(len(NATURAL_AA_ALPHABET))
        ])

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
        if self.training:
            decoding_order = torch.argsort((chain_M + 0.0001) * (torch.abs(torch.randn(chain_M.shape, device=X.device))))
        else:
            decoding_order = torch.argsort(chain_M + 0.0001)

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
        
        # --- V19 核心逻辑 ---
        
        # 1. 门控 logits (判断氨基酸类型) [B, L, 20]
        logits_gate = self.W_out_gate(h_V) 
        
        # 2. 专家 logits [B, L, 20]
        # 这种写法稍微有点慢，但在 PyTorch 里是最清晰的
        # 也可以用 1x1 卷积或者 grouped linear 优化，但这里为了逻辑清晰先用 list
        expert_outputs_list = [expert(h_V) for expert in self.methyl_experts]
        logits_experts = torch.cat(expert_outputs_list, dim=-1) # [B, L, 20]
        
        return logits_gate, logits_experts

# =============================================================================
# 2. 智能权重初始化
# =============================================================================
def smart_load_weights(model, pretrained_path, device):
    print(f"\n>>> [V19 Init] Loading {pretrained_path}...")
    ckpt = torch.load(pretrained_path, map_location=device)
    state_dict = ckpt.get('model_state_dict', ckpt)
    
    model_state = model.state_dict()
    load_dict = {}
    for k, v in state_dict.items():
        clean_k = k.replace("module.", "")
        # 复用 output layer 给 gate
        if clean_k == "W_out_gate.weight" and "W_out.weight" in state_dict:
             load_dict[clean_k] = state_dict["W_out.weight"][:20, :]
             continue
        if clean_k == "W_out_gate.bias" and "W_out.bias" in state_dict:
             load_dict[clean_k] = state_dict["W_out.bias"][:20]
             continue
        if clean_k in model_state and v.shape == model_state[clean_k].shape:
            load_dict[clean_k] = v
            
    model.load_state_dict(load_dict, strict=False)
    
    # Init Embedding
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
                    model.W_s.weight.data[nme_emb_idx].copy_(parent_emb + torch.randn_like(parent_emb)*0.001)

    print("✅ V19 Soft-MoE System Assembled.")

# =============================================================================
# 3. 专家训练 Loss (Training Logic)
# =============================================================================
def calculate_loss_v19(logits_gate, logits_experts, targets, mask):
    """
    训练策略：
    1. Gate Loss: 必须学会认人 (CrossEntropy)
    2. Expert Loss: 只有【真正的那个氨基酸】对应的专家，会被惩罚/奖励。
       别的专家瞎猜，我们不怪它，也不教它（因为这本来就不是它的专业）。
    """
    mask_flat = mask.contiguous().view(-1).bool()
    if mask_flat.sum() == 0: return torch.tensor(0.0, device=logits_gate.device), 0
    
    targets_flat = targets.contiguous().view(-1)[mask_flat]
    valid_pos = targets_flat != EXTENDED_AA_TO_INDEX.get('X', -1)
    if valid_pos.sum() == 0: return torch.tensor(0.0, device=logits_gate.device), 0
    
    targets_flat = targets_flat[valid_pos]
    
    # [N, 20]
    logits_gate = logits_gate.contiguous().view(-1, 20)[mask_flat][valid_pos]
    # [N, 20] - 每个专家对每个样本的一个打分
    logits_experts = logits_experts.contiguous().view(-1, 20)[mask_flat][valid_pos]
    
    # --- 准备标签 ---
    # 1. 真实身份 (0-19)
    identity_targets = targets_flat.clone()
    offset = len(NATURAL_AA_ALPHABET)
    for methyl_rel, natural_idx in NMETHYL_TO_NATURAL_MAPPING.items():
        identity_targets[identity_targets == (methyl_rel + offset)] = natural_idx
    
    # 2. 是否甲基化 (0/1)
    is_methylated = (targets_flat >= len(NATURAL_AA_ALPHABET)).float()
    
    # --- Loss 1: Gate (认人) ---
    # 忽略那些完全不认识的氨基酸(>=20)
    identity_targets_clamped = identity_targets.clone()
    identity_targets_clamped[identity_targets_clamped >= 20] = -100
    loss_gate = F.cross_entropy(logits_gate, identity_targets_clamped, ignore_index=-100)
    
    # --- Loss 2: Expert (专精) ---
    # 我们只挑选 ground truth 对应的那个 expert 的输出
    # 如果样本是 A (或者 Methyl-A)，我们只看 expert_A 的输出
    
    # 只有标签有效的才算
    valid_expert_mask = identity_targets_clamped != -100
    if valid_expert_mask.sum() > 0:
        # [M, 20] -> 选出对应的 [M, 1]
        selected_experts_logits = torch.gather(
            logits_experts[valid_expert_mask], 
            1, 
            identity_targets_clamped[valid_expert_mask].unsqueeze(1)
        ).squeeze(1)
        
        selected_methyl_targets = is_methylated[valid_expert_mask]
        
        # BCEWithLogitsLoss
        loss_expert = F.binary_cross_entropy_with_logits(
            selected_experts_logits,
            selected_methyl_targets,
            pos_weight=torch.tensor([6.0], device=logits_gate.device) # 保持高敏感度
        )
    else:
        loss_expert = torch.tensor(0.0, device=logits_gate.device)

    return loss_gate + 2.0 * loss_expert, identity_targets.numel()

# =============================================================================
# 4. 数据处理 (标准)
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
# 5. 验证函数 (Soft Voting Inference)
# =============================================================================
def validate_and_report(model, loader, device, epoch_num):
    model.eval()
    print(f"\n🔍 [Epoch {epoch_num}] Soft-MoE Council Session...")
    
    all_targets, all_preds = [], []
    methyl_idx_to_nat_idx = {m + len(NATURAL_AA_ALPHABET): n for m, n in NMETHYL_TO_NATURAL_MAPPING.items()}

    with torch.no_grad():
        for batch in loader:
            f = featurize_batch(batch, device)
            if f is None: continue
            X, S, mask, chain_M, residue_idx, chain_encoding_all = f
            
            # Forward
            l_gate, l_experts = model(X, S, mask, chain_M, residue_idx, chain_encoding_all)
            
            # --- Inference: Soft Voting Logic ---
            
            # 1. 计算每个专家的话语权 (Probability of being that AA)
            # [B, L, 20]
            probs_gate = F.softmax(l_gate, dim=-1)
            
            # 2. 计算每个专家判断“是甲基化”的概率
            # [B, L, 20] -> Sigmoid
            probs_expert_says_yes = torch.sigmoid(l_experts)
            
            # 3. 加权求和！(The Magic Step)
            # 最终甲基化概率 = sum(我是这个氨基酸的概率 * 这个氨基酸甲基化的概率)
            # [B, L]
            final_methyl_prob = torch.sum(probs_gate * probs_expert_says_yes, dim=-1)
            
            # 4. 决定 Base (还是选概率最大的那个作为基底)
            pred_base_idx = torch.argmax(probs_gate, -1)
            
            # 5. 阈值判定
            pred_is_methyl = (final_methyl_prob > 0.45).long() # Soft Voting 比较准，阈值可以高一点
            
            # Combine
            final_pred = pred_base_idx.clone()
            for m_abs_idx, n_idx in methyl_idx_to_nat_idx.items():
                # 只有当 (加权判定是甲基化) AND (基底判定正确) 时，才改写
                mask_update = (pred_is_methyl == 1) & (pred_base_idx == n_idx)
                final_pred[mask_update] = m_abs_idx
            
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
    
    print("\n🧐 Detailed Council Report:")
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
    print(f"🏆 Soft-MoE Score: {composite_score:.4f} (BaseAcc: {base_acc:.4f}, MeRec: {methyl_recall:.4f})")
    
    return composite_score

# =============================================================================
# 6. 主循环
# =============================================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrained_weights", type=str, required=True)
    parser.add_argument("--nmethyl_data", type=str, required=True)
    parser.add_argument("--test_data", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="./run_v19_soft_moe")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=8)
    args = parser.parse_args()
    
    set_strict_seed(42)
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = SoftMoEProteinMPNN(augment_eps=0.1).to(device)
    smart_load_weights(model, args.pretrained_weights, device)

    train_ds = JSONLDataset(args.nmethyl_data)
    sampler = get_weighted_sampler(train_ds, oversample_weight=10.0)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, sampler=sampler, collate_fn=collate_fn)
    test_ds = JSONLDataset(args.test_data)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=5e-4, epochs=args.epochs, steps_per_epoch=len(train_loader), pct_start=0.2)

    print("🚀 Starting V19: Soft-MoE Council...")
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
            l_gate, l_experts = model(X, S, mask, chain_M, residue_idx, chain_encoding_all)
            
            loss, valid = calculate_loss_v19(l_gate, l_experts, S, mask)
            
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
                save_path = os.path.join(args.output_dir, "best_model_v19.pt")
                torch.save({'model_state_dict': model.state_dict()}, save_path)
                print(f"🌟 New Best Model Saved! (Score: {best_score:.4f})")

if __name__ == "__main__":
    main()
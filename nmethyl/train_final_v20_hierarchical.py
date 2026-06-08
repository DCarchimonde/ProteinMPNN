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
# 1. 模型定义 (V20: Hierarchical Experts)
# =============================================================================
class HierarchicalProteinMPNN(ProteinMPNN):
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
        
        # HEAD 1: Base Identity Classifier (你是谁？)
        # 任务：必须把 Methyl-A 和 A 都分类为 A (Index 0-19)
        self.W_out_base = nn.Linear(hidden_dim, len(NATURAL_AA_ALPHABET))
        
        # HEAD 2: The 20 Independent Experts (你有病吗？)
        # 这是一个包含 20 个独立线性层的列表
        # Expert[0] 负责判断 A 是否甲基化
        # Expert[1] 负责判断 C 是否甲基化...
        self.experts = nn.ModuleList([
            nn.Linear(hidden_dim, 1) for _ in range(len(NATURAL_AA_ALPHABET))
        ])

    def forward(self, X, S, mask, chain_M, residue_idx, chain_encoding_all):
        # --- Encoder / Decoder (Standard ProteinMPNN) ---
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
        
        # --- V20 Logic ---
        
        # 1. Base Logic (Identify the Amino Acid)
        logits_base = self.W_out_base(h_V) # [B, L, 20]
        
        # 2. Experts Logic (Methylation Check)
        # 我们一次性计算所有专家的输出，形成 [B, L, 20]
        # 每一列代表假设它是该氨基酸时的甲基化 Logit
        expert_outputs = [expert(h_V) for expert in self.experts]
        logits_experts = torch.cat(expert_outputs, dim=-1) # [B, L, 20]
        
        return logits_base, logits_experts

# =============================================================================
# 2. 智能权重初始化 (迁移学习)
# =============================================================================
def smart_load_weights(model, pretrained_path, device):
    print(f"\n>>> [V20 Init] Loading {pretrained_path}...")
    ckpt = torch.load(pretrained_path, map_location=device)
    state_dict = ckpt.get('model_state_dict', ckpt)
    
    model_state = model.state_dict()
    load_dict = {}
    for k, v in state_dict.items():
        clean_k = k.replace("module.", "")
        
        # 将原版模型的输出层 (W_out) 赋给我们的 Base Classifier (W_out_base)
        # 这样模型一开始就拥有了很强的“认人”能力！
        if clean_k == "W_out_base.weight" and "W_out.weight" in state_dict:
             load_dict[clean_k] = state_dict["W_out.weight"][:20, :]
             continue
        if clean_k == "W_out_base.bias" and "W_out.bias" in state_dict:
             load_dict[clean_k] = state_dict["W_out.bias"][:20]
             continue
             
        if clean_k in model_state and v.shape == model_state[clean_k].shape:
            load_dict[clean_k] = v
            
    model.load_state_dict(load_dict, strict=False)
    
    # Init Embedding (Extended)
    ws_file = state_dict.get('W_s.weight', state_dict.get('module.W_s.weight'))
    if ws_file is not None:
        with torch.no_grad():
            for i, char in enumerate(NATURAL_AA_ALPHABET):
                if char in EXTENDED_AA_TO_INDEX:
                    model.W_s.weight.data[EXTENDED_AA_TO_INDEX[char]].copy_(ws_file[i])
            for nme_idx, nat_idx in NMETHYL_TO_NATURAL_MAPPING.items():
                if nme_idx < len(METHYL_AA_ALPHABET):
                    # 甲基化氨基酸 embedding 初始化为 (天然氨基酸 + 微小扰动)
                    nme_char = METHYL_AA_ALPHABET[nme_idx]
                    nat_char = NATURAL_AA_ALPHABET[nat_idx]
                    nat_emb_idx = EXTENDED_AA_TO_INDEX[nat_char]
                    nme_emb_idx = EXTENDED_AA_TO_INDEX[nme_char]
                    parent_emb = model.W_s.weight.data[nat_emb_idx]
                    model.W_s.weight.data[nme_emb_idx].copy_(parent_emb + torch.randn_like(parent_emb)*0.01)

    print("✅ V20 Hierarchical System Ready.")

# =============================================================================
# 3. 专精 Loss (分而治之)
# =============================================================================
def calculate_loss_v20(logits_base, logits_experts, targets, mask):
    """
    V20 训练核心：
    1. Base Loss: 所有样本（无论是否甲基化）都必须被正确分类为 20 种天然氨基酸之一。
       (这将大幅降低 Miss -> Other)
    2. Expert Loss: 只有对应身份的专家才会被训练。
       (比如标签是 A，那么只有 Expert_A 会产生 Loss，其他 19 个专家的输出被忽略)
    """
    mask_flat = mask.contiguous().view(-1).bool()
    if mask_flat.sum() == 0: return torch.tensor(0.0, device=logits_base.device), 0
    
    targets_flat = targets.contiguous().view(-1)[mask_flat]
    valid_pos = targets_flat != EXTENDED_AA_TO_INDEX.get('X', -1)
    if valid_pos.sum() == 0: return torch.tensor(0.0, device=logits_base.device), 0
    
    targets_flat = targets_flat[valid_pos]
    
    # [N, 20]
    logits_base = logits_base.contiguous().view(-1, 20)[mask_flat][valid_pos]
    logits_experts = logits_experts.contiguous().view(-1, 20)[mask_flat][valid_pos]
    
    # --- 1. 准备 Base Labels (认祖归宗) ---
    # 所有 Methyl-X (index > 20) 都要映射回 Natural-X (index 0-19)
    base_targets = targets_flat.clone()
    offset = len(NATURAL_AA_ALPHABET)
    for methyl_rel, natural_idx in NMETHYL_TO_NATURAL_MAPPING.items():
        base_targets[base_targets == (methyl_rel + offset)] = natural_idx
    
    # --- 2. 准备 Expert Labels (是否甲基化) ---
    # 1 if Methylated, 0 if Natural
    is_methylated = (targets_flat >= len(NATURAL_AA_ALPHABET)).float()
    
    # --- Loss 1: Base Classification (解决 Miss->Other 的关键) ---
    base_targets_clamped = base_targets.clone()
    base_targets_clamped[base_targets_clamped >= 20] = -100 # Safety
    loss_base = F.cross_entropy(logits_base, base_targets_clamped, ignore_index=-100)
    
    # --- Loss 2: Targeted Expert Training (解决 Recall 的关键) ---
    valid_expert_mask = base_targets_clamped != -100
    if valid_expert_mask.sum() > 0:
        # 关键操作：Gather
        # 我们只从 20 个专家的输出中，挑出 Truth 对应的那个专家
        # 比如 Truth 是 A，我们只取 Expert_A 的预测值
        target_expert_logits = torch.gather(
            logits_experts[valid_expert_mask], 
            1, 
            base_targets_clamped[valid_expert_mask].unsqueeze(1)
        ).squeeze(1)
        
        target_expert_labels = is_methylated[valid_expert_mask]
        
        # 二分类 Loss
        loss_expert = F.binary_cross_entropy_with_logits(
            target_expert_logits,
            target_expert_labels,
            pos_weight=torch.tensor([6.0], device=logits_base.device)
        )
    else:
        loss_expert = torch.tensor(0.0, device=logits_base.device)

    # 1.0 vs 2.0 的权重：保证基础认知准确的前提下，猛攻专家能力
    return loss_base + 2.0 * loss_expert, base_targets.numel()

# =============================================================================
# 4. 数据处理
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
# 5. 验证函数 (Hard Hierarchical Inference)
# =============================================================================
def validate_and_report(model, loader, device, epoch_num):
    model.eval()
    print(f"\n🔍 [Epoch {epoch_num}] Expert Council Diagnosing...")
    
    all_targets, all_preds = [], []
    methyl_idx_to_nat_idx = {m + len(NATURAL_AA_ALPHABET): n for m, n in NMETHYL_TO_NATURAL_MAPPING.items()}

    with torch.no_grad():
        for batch in loader:
            f = featurize_batch(batch, device)
            if f is None: continue
            X, S, mask, chain_M, residue_idx, chain_encoding_all = f
            
            l_base, l_experts = model(X, S, mask, chain_M, residue_idx, chain_encoding_all)
            
            # 1. 第一层：Base Identity (必须先认准人！)
            pred_base_idx = torch.argmax(l_base, -1) # [B, L]
            
            # 2. 第二层：Ask the Expert
            # 根据第一层认出是谁，去问对应的专家
            selected_expert_logit = torch.gather(
                l_experts, 
                -1, 
                pred_base_idx.unsqueeze(-1)
            ).squeeze(-1) # [B, L]
            
            # 3. 专家裁决
            probs_methyl = torch.sigmoid(selected_expert_logit)
            pred_is_methyl = (probs_methyl > 0.5).long()
            
            # 4. 组合结果
            final_pred = pred_base_idx.clone()
            
            for m_abs_idx, n_idx in methyl_idx_to_nat_idx.items():
                # 逻辑：Identity 是 A, 且 A专家说是 Methyl -> 输出 Methyl-A
                mask_update = (pred_is_methyl == 1) & (pred_base_idx == n_idx)
                final_pred[mask_update] = m_abs_idx
            
            # 收集
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
    
    # 重点看 Miss->Other 是否减少！
    print("\n🧐 Detailed Specialist Report:")
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
    print(f"🏆 Specialist Score: {composite_score:.4f} (BaseAcc: {base_acc:.4f}, MeRec: {methyl_recall:.4f})")
    
    return composite_score

# =============================================================================
# 6. 主循环
# =============================================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrained_weights", type=str, required=True)
    parser.add_argument("--nmethyl_data", type=str, required=True)
    parser.add_argument("--test_data", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="./run_v20_experts")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=8)
    args = parser.parse_args()
    
    set_strict_seed(42)
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = HierarchicalProteinMPNN(augment_eps=0.1).to(device)
    smart_load_weights(model, args.pretrained_weights, device)

    train_ds = JSONLDataset(args.nmethyl_data)
    sampler = get_weighted_sampler(train_ds, oversample_weight=10.0)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, sampler=sampler, collate_fn=collate_fn)
    test_ds = JSONLDataset(args.test_data)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=5e-4, epochs=args.epochs, steps_per_epoch=len(train_loader), pct_start=0.2)

    print("🚀 Starting V20: Hierarchical Expert System...")
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
            l_base, l_experts = model(X, S, mask, chain_M, residue_idx, chain_encoding_all)
            
            loss, valid = calculate_loss_v20(l_base, l_experts, S, mask)
            
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
                save_path = os.path.join(args.output_dir, "best_model_v20.pt")
                torch.save({'model_state_dict': model.state_dict()}, save_path)
                print(f"🌟 New Best Model Saved! (Score: {best_score:.4f})")

if __name__ == "__main__":
    main()
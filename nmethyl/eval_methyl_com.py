import os
import sys
os.environ["OMP_NUM_THREADS"] = "1"
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import json
import numpy as np
from torch.utils.data import Dataset, DataLoader

# =============================================================================
# 1. 基础配置
# =============================================================================
NATURAL_AA_ALPHABET = "ACDEFGHIKLMNPQRSTVWY"
EXTENDED_AA_ALPHABET = "ACDEFGHIKLMNPQRSTVWYacdefghiklmnqrstvwvyX"
EXTENDED_AA_TO_INDEX = {aa: i for i, aa in enumerate(EXTENDED_AA_ALPHABET)}
NATURAL_AA_TO_INDEX = {aa: i for i, aa in enumerate(NATURAL_AA_ALPHABET)}

# 小写/甲基化 AA -> 对应天然 AA 的 index。
# 注意：这个脚本评估天然 AA 恢复率时，会把小写甲基化残基全部还原成大写天然 AA。
NMETHYL_TO_NATURAL_MAPPING = {}
for i, aa in enumerate(NATURAL_AA_ALPHABET):
    lower_aa = aa.lower()
    if lower_aa in EXTENDED_AA_TO_INDEX:
        nmethyl_idx = EXTENDED_AA_TO_INDEX[lower_aa]
        natural_idx = EXTENDED_AA_TO_INDEX[aa]
        NMETHYL_TO_NATURAL_MAPPING[nmethyl_idx - 20] = natural_idx

# 直接版本：extended index -> natural 0~19 index；X/未知返回 -1。
EXTENDED_TO_NATURAL_INDEX = np.full(len(EXTENDED_AA_ALPHABET), -1, dtype=np.int64)
for aa in NATURAL_AA_ALPHABET:
    EXTENDED_TO_NATURAL_INDEX[EXTENDED_AA_TO_INDEX[aa]] = NATURAL_AA_TO_INDEX[aa]
    lower_aa = aa.lower()
    if lower_aa in EXTENDED_AA_TO_INDEX:
        EXTENDED_TO_NATURAL_INDEX[EXTENDED_AA_TO_INDEX[lower_aa]] = NATURAL_AA_TO_INDEX[aa]

# =============================================================================
# 2. 模型结构 (RobustHierarchicalProteinMPNN)
# =============================================================================
def gather_nodes(nodes, neighbor_idx):
    neighbors_flat = neighbor_idx.view((neighbor_idx.shape[0], -1))
    neighbors_flat = neighbors_flat.unsqueeze(-1).expand(-1, -1, nodes.size(2))
    return torch.gather(nodes, 1, neighbors_flat).view(list(neighbor_idx.shape)[:3] + [-1])

def cat_neighbors_nodes(h_nodes, h_neighbors, E_idx):
    return torch.cat([h_neighbors, gather_nodes(h_nodes, E_idx)], -1)

class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.W_in = nn.Linear(d_model, d_ff)
        self.W_out = nn.Linear(d_ff, d_model)
        self.act = nn.GELU()
    def forward(self, x):
        return self.W_out(self.act(self.W_in(x)))

class EncLayer(nn.Module):
    def __init__(self, num_hidden, num_in, dropout=0.1, scale=30):
        super().__init__()
        self.scale = scale
        self.dropout1, self.dropout2 = nn.Dropout(dropout), nn.Dropout(dropout)
        self.norm1, self.norm2 = nn.LayerNorm(num_hidden), nn.LayerNorm(num_hidden)
        self.W1 = nn.Linear(num_hidden + num_in, num_hidden)
        self.W2 = nn.Linear(num_hidden, num_hidden)
        self.W3 = nn.Linear(num_hidden, num_hidden)
        self.act = nn.GELU()
        self.dense = PositionwiseFeedForward(num_hidden, num_hidden * 4, dropout)
    def forward(self, h_V, h_E, E_idx, mask_V=None, mask_attend=None):
        h_EV = torch.cat([h_V.unsqueeze(-2).expand(-1, -1, h_E.size(-2), -1), gather_nodes(h_V, E_idx), h_E], -1)
        h_message = self.W3(self.act(self.W2(self.act(self.W1(h_EV)))))
        if mask_attend is not None:
            h_message = mask_attend.unsqueeze(-1) * h_message
        h_V = self.norm1(h_V + self.dropout1(torch.sum(h_message, -2) / self.scale))
        return self.norm2(h_V + self.dropout2(self.dense(h_V))), h_E

class DecLayer(nn.Module):
    def __init__(self, num_hidden, num_in, dropout=0.1, scale=30):
        super().__init__()
        self.scale = scale
        self.dropout1, self.dropout2 = nn.Dropout(dropout), nn.Dropout(dropout)
        self.norm1, self.norm2 = nn.LayerNorm(num_hidden), nn.LayerNorm(num_hidden)
        self.W1 = nn.Linear(num_hidden + num_in, num_hidden)
        self.W2 = nn.Linear(num_hidden, num_hidden)
        self.W3 = nn.Linear(num_hidden, num_hidden)
        self.act = nn.GELU()
        self.dense = PositionwiseFeedForward(num_hidden, num_hidden * 4, dropout)
    def forward(self, h_V, h_E, mask_V=None, mask_attend=None):
        h_cat = torch.cat([h_V.unsqueeze(-2).expand(-1, -1, h_E.size(-2), -1), h_E], -1)
        h_message = self.W3(self.act(self.W2(self.act(self.W1(h_cat)))))
        if mask_attend is not None:
            h_message = mask_attend.unsqueeze(-1) * h_message
        h_V = self.norm1(h_V + self.dropout1(torch.sum(h_message, -2) / self.scale))
        return self.norm2(h_V + self.dropout2(self.dense(h_V)))

class RobustHierarchicalProteinMPNN(nn.Module):
    def __init__(self, hidden_dim=128, k_neighbors=48, augment_eps=0.0, dropout=0.1):
        super().__init__()
        self.hidden_dim, self.k_neighbors, self.augment_eps = hidden_dim, k_neighbors, augment_eps
        self.features = nn.ModuleDict({
            'embeddings': nn.ModuleDict({'linear': nn.Linear(66, 16)}),
            'edge_embedding': nn.Linear(416, 128, bias=False),
            'norm_edges': nn.LayerNorm(128)
        })
        self.W_e = nn.Linear(128, hidden_dim, bias=True)
        self.W_s = nn.Embedding(len(EXTENDED_AA_ALPHABET), hidden_dim)
        self.encoder_layers = nn.ModuleList([EncLayer(hidden_dim, hidden_dim * 2, dropout=dropout) for _ in range(3)])
        self.decoder_layers = nn.ModuleList([DecLayer(hidden_dim, hidden_dim * 3, dropout=dropout) for _ in range(3)])
        self.W_out_base = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, len(NATURAL_AA_ALPHABET))
        )
        self.experts = nn.ModuleList([nn.Linear(hidden_dim, 1) for _ in range(len(NATURAL_AA_ALPHABET))])

    def forward(self, X, S, mask, chain_M, residue_idx, chain_encoding_all):
        b, c = X[:, :, 1, :] - X[:, :, 0, :], X[:, :, 2, :] - X[:, :, 1, :]
        a = torch.cross(b, c, dim=-1)
        Cb = -0.58273431 * a + 0.56802827 * b - 0.54067466 * c + X[:, :, 1, :]
        dist = torch.norm(X[:, :, 1, :].unsqueeze(1) - X[:, :, 1, :].unsqueeze(2), dim=-1) + (1.0 - (mask.unsqueeze(1) * mask.unsqueeze(2))) * 1e8
        E_idx = torch.topk(dist, min(self.k_neighbors, dist.shape[-1]), dim=-1, largest=False)[1]
        offset = torch.gather(residue_idx.unsqueeze(1) - residue_idx.unsqueeze(2), 2, E_idx)
        pos_emb = self.features['embeddings']['linear'](F.one_hot(torch.clip(offset + 32, 0, 64), 66).float())
        RBF_all = [self._rbf(torch.gather(dist, 2, E_idx))]
        for i, a1 in enumerate([X[:, :, 0, :], X[:, :, 2, :], X[:, :, 3, :], Cb, X[:, :, 1, :]]):
            for j, a2 in enumerate([X[:, :, 0, :], X[:, :, 2, :], X[:, :, 3, :], Cb, X[:, :, 1, :]]):
                if i != 4 or j != 4:
                    RBF_all.append(self._rbf(torch.gather(torch.norm(a1.unsqueeze(1) - a2.unsqueeze(2), dim=-1), 2, E_idx)))
        E = self.features['norm_edges'](self.features['edge_embedding'](torch.cat((pos_emb, torch.cat(RBF_all, dim=-1)), -1)))
        h_V, h_E = torch.zeros((E.shape[0], E.shape[1], E.shape[-1]), device=E.device), self.W_e(E)
        mask_attend = mask.unsqueeze(-1) * gather_nodes(mask.unsqueeze(-1), E_idx).squeeze(-1)
        for layer in self.encoder_layers:
            h_V, h_E = layer(h_V, h_E, E_idx, mask, mask_attend)
        h_ES = cat_neighbors_nodes(self.W_s(S), h_E, E_idx)
        h_EXV_encoder = cat_neighbors_nodes(h_V, cat_neighbors_nodes(torch.zeros_like(self.W_s(S)), h_E, E_idx), E_idx)
        permutation_matrix_reverse = torch.nn.functional.one_hot(torch.argsort(chain_M * mask + 0.0001), num_classes=E_idx.shape[1]).float()
        order_mask_backward = torch.einsum(
            'ij, biq, bjp->bqp',
            (1 - torch.triu(torch.ones(E_idx.shape[1], E_idx.shape[1], device=X.device))),
            permutation_matrix_reverse,
            permutation_matrix_reverse
        )
        mask_attend = torch.gather(order_mask_backward, 2, E_idx).unsqueeze(-1)
        mask_bw = mask.view([mask.size(0), mask.size(1), 1, 1]) * mask_attend
        mask_fw = mask.view([mask.size(0), mask.size(1), 1, 1]) * (1. - mask_attend)
        for layer in self.decoder_layers:
            h_V = layer(h_V, mask_bw * cat_neighbors_nodes(h_V, h_ES, E_idx) + mask_fw * h_EXV_encoder, mask)

        logits_base = self.W_out_base(h_V)
        logits_experts = torch.cat([expert(h_V) for expert in self.experts], dim=-1)
        return logits_base, logits_experts

    def _rbf(self, D):
        return torch.exp(-((D.unsqueeze(-1) - torch.linspace(2., 22., 16, device=D.device)) / ((22. - 2.) / 16)) ** 2)

# =============================================================================
# 3. 防弹加载与复合物数据处理
# =============================================================================
def bulletproof_load_weights(model, checkpoint_path, device):
    ckpt = torch.load(checkpoint_path, map_location=device)
    state_dict = ckpt.get('model_state_dict', ckpt)
    if 'W_out.weight' in state_dict:
        state_dict['W_out_base.3.weight'] = state_dict.pop('W_out.weight')
        state_dict['W_out_base.3.bias'] = state_dict.pop('W_out.bias')
    elif 'module.W_out.weight' in state_dict:
        state_dict['W_out_base.3.weight'] = state_dict.pop('module.W_out.weight')
        state_dict['W_out_base.3.bias'] = state_dict.pop('module.W_out.bias')

    model_state = model.state_dict()
    new_state_dict = {}
    for k, v in state_dict.items():
        clean_k = k.replace('module.', '')
        if clean_k in model_state:
            if v.shape != model_state[clean_k].shape:
                new_v = model_state[clean_k].clone()
                slices = tuple(slice(0, min(dim_v, dim_m)) for dim_v, dim_m in zip(v.shape, new_v.shape))
                new_v[slices] = v[slices]
                new_state_dict[clean_k] = new_v
            else:
                new_state_dict[clean_k] = v
    model.load_state_dict(new_state_dict, strict=False)
    print("✅ 权重防弹加载成功！")

def get_chain_order(entry):
    """兼容 ProteinMPNN jsonl 的 masked_list/visible_list；没有时自动从 seq_chain_* 推断多链。"""
    chains = []
    for c in entry.get('masked_list', []) + entry.get('visible_list', []):
        if c not in chains:
            chains.append(c)
    if chains:
        return chains

    inferred = []
    for k in entry.keys():
        if k.startswith('seq_chain_'):
            inferred.append(k.replace('seq_chain_', ''))
    if inferred:
        return sorted(inferred)
    return ['A']

def chain_length(entry, chain_id):
    seq = entry.get(f'seq_chain_{chain_id}', '')
    N = entry.get(f'N_chain_{chain_id}', [])
    CA = entry.get(f'CA_chain_{chain_id}', [])
    C = entry.get(f'C_chain_{chain_id}', [])
    O = entry.get(f'O_chain_{chain_id}', [])
    return min(len(seq), len(N), len(CA), len(C), len(O))

def featurize_batch(batch, device):
    """
    复合物版本 featurize：
    - 不再强制要求 seq_chain_A；支持 A/B/C/... 任意多链。
    - L_max 按所有链长度总和计算，而不是只看 A 链。
    - padding 坐标用 NaN 初始化，mask 不会把 padding 当成真实残基。
    - chain_M: masked_list 中的链为 1，visible_list 中的链为 0；如果没有 masked_list/visible_list，就默认全评估。
    """
    prepared = []
    for b in batch:
        chains = get_chain_order(b)
        lengths = [chain_length(b, c) for c in chains]
        total_len = int(sum(lengths))
        if total_len > 0:
            prepared.append((b, chains, lengths, total_len))

    if not prepared:
        return None

    B = len(prepared)
    L_max = max(x[3] for x in prepared)
    X = np.full([B, L_max, 4, 3], np.nan, dtype=np.float32)
    S = np.full([B, L_max], EXTENDED_AA_TO_INDEX.get('X', len(EXTENDED_AA_ALPHABET) - 1), dtype=np.int32)
    residue_idx = -100 * np.ones([B, L_max], dtype=np.int32)
    chain_M = np.zeros([B, L_max], dtype=np.float32)
    chain_encoding_all = np.zeros([B, L_max], dtype=np.int32)

    for i, (b, all_chains, lengths, total_len) in enumerate(prepared):
        masked_set = set(b.get('masked_list', []))
        # 如果 jsonl 里没有 masked_list/visible_list，默认所有链都作为评估/设计链。
        default_all_masked = len(b.get('masked_list', [])) == 0 and len(b.get('visible_list', [])) == 0
        l_p = 0
        for c_i, c_id in enumerate(all_chains):
            l = lengths[c_i]
            if l == 0:
                continue

            seq = b.get(f'seq_chain_{c_id}', '')[:l]
            N = np.asarray(b.get(f'N_chain_{c_id}', []), dtype=np.float32)[:l]
            CA = np.asarray(b.get(f'CA_chain_{c_id}', []), dtype=np.float32)[:l]
            C = np.asarray(b.get(f'C_chain_{c_id}', []), dtype=np.float32)[:l]
            O = np.asarray(b.get(f'O_chain_{c_id}', []), dtype=np.float32)[:l]

            # 保留你原来的 CA/O 防呆逻辑。
            if len(N) > 0 and len(CA) > 0 and len(O) > 0:
                if np.linalg.norm(N[:1] - O[:1]) < np.linalg.norm(N[:1] - CA[:1]) and np.linalg.norm(N[:1] - O[:1]) < 1.6:
                    CA, O = O, CA

            X[i, l_p:l_p + l, 0, :] = N
            X[i, l_p:l_p + l, 1, :] = CA
            X[i, l_p:l_p + l, 2, :] = C
            X[i, l_p:l_p + l, 3, :] = O
            S[i, l_p:l_p + l] = [EXTENDED_AA_TO_INDEX.get(aa, EXTENDED_AA_TO_INDEX.get('X', 40)) for aa in seq]

            if default_all_masked or c_id in masked_set:
                chain_M[i, l_p:l_p + l] = 1.0
            else:
                chain_M[i, l_p:l_p + l] = 0.0

            # 不同链之间拉开 residue_idx，避免链间位置编码混在一起。
            residue_idx[i, l_p:l_p + l] = np.arange(l, dtype=np.int32) + c_i * 100
            chain_encoding_all[i, l_p:l_p + l] = c_i
            l_p += l

    mask = np.isfinite(np.sum(X, axis=(2, 3))).astype(np.float32)
    X[np.isnan(X)] = 0.0

    tensors = [X, S, mask, chain_M, residue_idx, chain_encoding_all]
    return [
        torch.from_numpy(t).to(dtype=torch.long if t.dtype == np.int32 else torch.float32, device=device)
        for t in tensors
    ]

class JSONLDataset(Dataset):
    def __init__(self, jsonl_file):
        self.data = [json.loads(line) for line in open(jsonl_file, 'r') if line.strip()]
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        return self.data[idx]

# =============================================================================
# 4A. 原功能保留：纯净解耦的二分类评估 (Decoupled Binary Evaluation)
# =============================================================================
def eval_binary_classifier(model, loader, device, thresholds):
    model.eval()
    print("🧪 正在评估甲基化探测器 (Decoupled Binary Evaluation)...")

    true_labels = []
    pred_probs = []

    offset = len(NATURAL_AA_ALPHABET)  # 20
    x_idx = EXTENDED_AA_TO_INDEX.get('X', 40)

    with torch.no_grad():
        for batch in loader:
            f = featurize_batch(batch, device)
            if f is None:
                continue
            X, S, mask, chain_M, residue_idx, chain_encoding_all = f

            l_base, l_experts = model(X, S, mask, chain_M, residue_idx, chain_encoding_all)

            # --- 核心：解耦评估 ---
            # 1. 提取真实的天然氨基酸序列 (Ground Truth Base)
            true_base_idx = S.clone()
            for m_rel, n_idx in NMETHYL_TO_NATURAL_MAPPING.items():
                true_base_idx[true_base_idx == (m_rel + offset)] = n_idx
            true_base_idx[true_base_idx >= 20] = 0  # 越界保护

            # 2. 直接拿真实的 Base AA 去问专家：你要不要加甲基？
            expert_logit = torch.gather(l_experts, -1, true_base_idx.unsqueeze(-1)).squeeze(-1)
            prob_methyl = torch.sigmoid(expert_logit)

            tgts = S.cpu().numpy().flatten()
            p_methyl = prob_methyl.cpu().numpy().flatten()
            mask_flat = mask.cpu().numpy().flatten().astype(bool)
            valid = mask_flat & (tgts != x_idx)

            true_labels.extend((tgts[valid] >= offset).astype(int))
            pred_probs.extend(p_methyl[valid])

    true_labels = np.array(true_labels)
    pred_probs = np.array(pred_probs)

    print("\n" + "=" * 95)
    print(f"{'Threshold':<10} | {'Binary Accuracy':<18} | {'Precision (精准率)':<18} | {'Recall (召回率)':<18} | {'F1-Score':<10}")
    print("-" * 95)

    for thresh in thresholds:
        pred_labels = (pred_probs > thresh).astype(int)

        TP = np.sum((true_labels == 1) & (pred_labels == 1))
        TN = np.sum((true_labels == 0) & (pred_labels == 0))
        FP = np.sum((true_labels == 0) & (pred_labels == 1))
        FN = np.sum((true_labels == 1) & (pred_labels == 0))

        acc = (TP + TN) / (TP + TN + FP + FN) * 100 if (TP + TN + FP + FN) > 0 else 0
        precision = TP / (TP + FP) * 100 if (TP + FP) > 0 else 0
        recall = TP / (TP + FN) * 100 if (TP + FN) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        marker = "🔥" if f1 > 35 else ""
        print(f"Thr = {thresh:<4.2f} | {acc:>14.2f}%   | {precision:>14.2f}%   | {recall:>14.2f}%   | {f1:>6.2f} {marker}")

    print("=" * 95)
    print("💡 论文说明（发给师兄看）：")
    print("  1. 【解耦评估】这里摒弃了基础氨基酸预测错误带来的级联误差，专门测了专家预测头的硬实力。")
    print("  2. 【Binary Accuracy】就是你记忆中的 80-90% 准确率，它包含了大量的 True Negatives（正确预测不加甲基）。")
    print("  3. 【F1-Score】这是目前学术界公认的最客观的二分类指标，建议把这几列数据画成折线图。")

# =============================================================================
# 4B. 新增：复合物天然 AA 恢复率 / 多分类 F1 评估
# =============================================================================
def _safe_div(num, den):
    return float(num) / float(den) if den > 0 else 0.0

def compute_multiclass_metrics(y_true, y_pred, num_classes=20):
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        if 0 <= t < num_classes and 0 <= p < num_classes:
            cm[t, p] += 1

    support = cm.sum(axis=1)
    tp = np.diag(cm)
    pred_count = cm.sum(axis=0)

    precision = np.array([_safe_div(tp[i], pred_count[i]) for i in range(num_classes)])
    recall = np.array([_safe_div(tp[i], support[i]) for i in range(num_classes)])
    f1 = np.array([
        _safe_div(2 * precision[i] * recall[i], precision[i] + recall[i])
        for i in range(num_classes)
    ])

    present = support > 0
    total = support.sum()
    accuracy = _safe_div(tp.sum(), total)
    macro_precision = precision[present].mean() if np.any(present) else 0.0
    macro_recall = recall[present].mean() if np.any(present) else 0.0
    macro_f1 = f1[present].mean() if np.any(present) else 0.0
    weighted_precision = _safe_div((precision * support).sum(), total)
    weighted_recall = _safe_div((recall * support).sum(), total)
    weighted_f1 = _safe_div((f1 * support).sum(), total)

    return {
        'cm': cm,
        'support': support,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'accuracy': accuracy,
        'macro_precision': macro_precision,
        'macro_recall': macro_recall,
        'macro_f1': macro_f1,
        'weighted_precision': weighted_precision,
        'weighted_recall': weighted_recall,
        'weighted_f1': weighted_f1,
    }

def eval_complex_natural_aa(model, loader, device, eval_chains='masked'):
    """
    复合物天然 AA 指标：
    - 只看 W_out_base 的 20 类天然氨基酸输出。
    - 真实标签 S 里如果是小写/甲基化 AA，先映射回对应大写天然 AA。
    - 不计算、不惩罚、不统计甲基化专家头的二分类结果。
    - eval_chains='masked': 只评估 masked_list 中的设计链；如果数据没有 masked_list/visible_list，则自动全评估。
      eval_chains='all': 复合物所有有坐标且非 X 的残基都评估。
    """
    model.eval()
    print("🧪 正在评估复合物天然氨基酸恢复率 / AA-level F1（忽略甲基化标签）...")

    all_true = []
    all_pred = []
    all_top5_hit = []

    with torch.no_grad():
        for batch in loader:
            f = featurize_batch(batch, device)
            if f is None:
                continue
            X, S, mask, chain_M, residue_idx, chain_encoding_all = f

            logits_base, _ = model(X, S, mask, chain_M, residue_idx, chain_encoding_all)
            pred = torch.argmax(logits_base, dim=-1)
            top5 = torch.topk(logits_base, k=min(5, logits_base.shape[-1]), dim=-1).indices

            S_np = S.detach().cpu().numpy()
            pred_np = pred.detach().cpu().numpy()
            top5_np = top5.detach().cpu().numpy()
            mask_np = mask.detach().cpu().numpy().astype(bool)
            chain_M_np = chain_M.detach().cpu().numpy() > 0.5

            # extended target -> natural target；小写甲基化残基在这里被还原。
            true_nat = np.full_like(S_np, -1, dtype=np.int64)
            valid_s = (S_np >= 0) & (S_np < len(EXTENDED_TO_NATURAL_INDEX))
            true_nat[valid_s] = EXTENDED_TO_NATURAL_INDEX[S_np[valid_s]]

            valid = mask_np & (true_nat >= 0)
            if eval_chains == 'masked':
                valid = valid & chain_M_np
            elif eval_chains != 'all':
                raise ValueError("eval_chains 只能是 'masked' 或 'all'")

            all_true.extend(true_nat[valid].tolist())
            all_pred.extend(pred_np[valid].tolist())
            all_top5_hit.extend([int(t in hits) for t, hits in zip(true_nat[valid], top5_np[valid])])

    y_true = np.array(all_true, dtype=np.int64)
    y_pred = np.array(all_pred, dtype=np.int64)
    top5_hit = np.array(all_top5_hit, dtype=np.int64)

    if len(y_true) == 0:
        print("⚠️ 没有找到可评估残基。请检查 jsonl 是否有坐标、seq_chain_*，以及 masked_list/visible_list 设置。")
        print("   如果你想评估所有链，可以加：--eval_chains all")
        return

    m = compute_multiclass_metrics(y_true, y_pred, num_classes=len(NATURAL_AA_ALPHABET))
    top1_recovery = m['accuracy'] * 100.0
    top5_recovery = top5_hit.mean() * 100.0 if len(top5_hit) > 0 else 0.0

    print("\n" + "=" * 105)
    print("📌 复合物天然氨基酸评估结果（甲基化字符已还原为天然 AA，不计算甲基化二分类）")
    print("-" * 105)
    print(f"Evaluated residues       : {len(y_true)}")
    print(f"Eval chains mode         : {eval_chains}")
    print(f"Top-1 Recovery / Accuracy: {top1_recovery:8.2f}%")
    print(f"Top-5 Recovery           : {top5_recovery:8.2f}%")
    print(f"Macro Precision          : {m['macro_precision'] * 100:8.2f}%")
    print(f"Macro Recall             : {m['macro_recall'] * 100:8.2f}%")
    print(f"Macro F1                 : {m['macro_f1'] * 100:8.2f}%")
    print(f"Weighted Precision       : {m['weighted_precision'] * 100:8.2f}%")
    print(f"Weighted Recall          : {m['weighted_recall'] * 100:8.2f}%")
    print(f"Weighted F1              : {m['weighted_f1'] * 100:8.2f}%")
    print("=" * 105)

    print("\n" + "=" * 105)
    print(f"{'AA':<4} | {'Support':>8} | {'Recovery/Recall':>16} | {'Precision':>12} | {'F1':>12}")
    print("-" * 105)
    for i, aa in enumerate(NATURAL_AA_ALPHABET):
        sup = int(m['support'][i])
        if sup == 0:
            continue
        print(f"{aa:<4} | {sup:>8d} | {m['recall'][i] * 100:>15.2f}% | {m['precision'][i] * 100:>11.2f}% | {m['f1'][i] * 100:>11.2f}%")
    print("=" * 105)

    print("\n💡 论文/汇报说明：")
    print("  1. 【AA Recovery】这里等价于 20 类天然氨基酸 top-1 accuracy，只评价天然 AA 是否恢复正确。")
    print("  2. 【忽略甲基化】真实序列中的小写/甲基化 AA 已先映射回对应天然 AA，因此不会因为是否甲基化而扣分。")
    print("  3. 【Macro F1】每种 AA 等权平均；【Weighted F1】按各 AA 出现次数加权，更接近整体恢复表现。")
    print("  4. 复合物多链已按 masked_list + visible_list 拼接；默认只评估 masked_list 设计链，没有该字段时自动评估所有链。")

# =============================================================================
# 5. 主程序
# =============================================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--test_data", type=str, required=True)
    parser.add_argument(
        "--eval_mode",
        type=str,
        default="complex_aa",
        choices=["complex_aa", "methyl_binary"],
        help="complex_aa: 复合物天然 AA 恢复率/F1；methyl_binary: 保留原来的甲基化二分类评估"
    )
    parser.add_argument(
        "--eval_chains",
        type=str,
        default="masked",
        choices=["masked", "all"],
        help="complex_aa 模式下：masked=只评估 masked_list 设计链；all=评估复合物所有链"
    )
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument(
        "--thresholds",
        type=float,
        nargs="*",
        default=[0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.98, 0.99],
        help="methyl_binary 模式下使用的阈值列表"
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = RobustHierarchicalProteinMPNN(augment_eps=0.0).to(device)

    bulletproof_load_weights(model, args.model_path, device)

    test_ds = JSONLDataset(args.test_data)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, collate_fn=lambda x: x)

    if args.eval_mode == "complex_aa":
        eval_complex_natural_aa(model, test_loader, device, eval_chains=args.eval_chains)
    elif args.eval_mode == "methyl_binary":
        eval_binary_classifier(model, test_loader, device, args.thresholds)
    else:
        raise ValueError(f"Unknown eval_mode: {args.eval_mode}")

if __name__ == "__main__":
    main()

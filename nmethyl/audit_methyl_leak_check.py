#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
audit_methyl_leak_check.py

目的：
1. 只评估“真实残基位点”，不把 padding 算进指标。
2. 明确区分两种甲基化评估：
   A) known_sequence_methyl:
      已知天然 AA 序列时，用 true_base -> expert(true_base) 判断是否甲基化。
      适合论文里表述为“methylation site classifier / given peptide sequence”。
   B) end_to_end_methyl:
      模型先 pred_base，再 expert(pred_base) 判断是否甲基化。
      适合表述为“joint AA design + methylation prediction”。
3. 同时打印 padding audit，说明 legacy 38% 为什么会被 padding A 抬高。

保存到：
    ProteinMPNN-main/nmethyl/audit_methyl_leak_check.py

单体：
    python nmethyl/audit_methyl_leak_check.py \
        --model_path "./frankenstein_v28.pt" \
        --test_data "nmethyl_data/test_set/test.jsonl" \
        --eval_chains masked \
        --batch_size 16

复合物短肽：
    python nmethyl/audit_methyl_leak_check.py \
        --model_path "./frankenstein_v28.pt" \
        --test_data "17_complexes_native.jsonl" \
        --eval_chains short \
        --max_peptide_len 30 \
        --batch_size 1
"""

import os
os.environ["OMP_NUM_THREADS"] = "1"

import sys
import json
import copy
import argparse
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint
from torch.utils.data import Dataset, DataLoader

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

try:
    from model_utils import ProteinMPNN, gather_nodes, cat_neighbors_nodes
    from nmethyl.utils.nmethyl_config import (
        EXTENDED_AA_ALPHABET,
        NATURAL_AA_ALPHABET,
        NMETHYL_TO_NATURAL_MAPPING,
        EXTENDED_AA_TO_INDEX,
    )
except Exception as e:
    print("❌ import failed. 请把脚本放在 ProteinMPNN-main/nmethyl/ 下，并从 ProteinMPNN-main 根目录运行。")
    print(repr(e))
    sys.exit(1)


# =============================================================================
# 1. Model: 与 sweep_threshold_robust.py 对齐
# =============================================================================
class RobustHierarchicalProteinMPNN(ProteinMPNN):
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
        self.W_out_base = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, len(NATURAL_AA_ALPHABET)),
        )
        self.experts = nn.ModuleList([
            nn.Linear(hidden_dim, 1) for _ in range(len(NATURAL_AA_ALPHABET))
        ])

    def forward(self, X, S, mask, chain_M, residue_idx, chain_encoding_all):
        E, E_idx = self.features(X, mask, residue_idx, chain_encoding_all)
        h_V = torch.zeros((E.shape[0], E.shape[1], E.shape[-1]), device=E.device)
        h_E = self.W_e(E)

        mask_attend = gather_nodes(mask.unsqueeze(-1), E_idx).squeeze(-1)
        mask_attend = mask.unsqueeze(-1) * mask_attend

        for layer in self.encoder_layers:
            h_V, h_E = torch.utils.checkpoint.checkpoint(
                layer, h_V, h_E, E_idx, mask, mask_attend
            )

        h_S = self.W_s(S)
        h_ES = cat_neighbors_nodes(h_S, h_E, E_idx)
        h_EX_encoder = cat_neighbors_nodes(torch.zeros_like(h_S), h_E, E_idx)
        h_EXV_encoder = cat_neighbors_nodes(h_V, h_EX_encoder, E_idx)

        chain_M = chain_M * mask
        decoding_order = torch.argsort(chain_M + 0.0001)

        mask_size = E_idx.shape[1]
        permutation_matrix_reverse = F.one_hot(decoding_order, num_classes=mask_size).float()

        order_mask_backward = torch.einsum(
            "ij, biq, bjp->bqp",
            (1 - torch.triu(torch.ones(mask_size, mask_size, device=X.device))),
            permutation_matrix_reverse,
            permutation_matrix_reverse,
        )

        mask_attend = torch.gather(order_mask_backward, 2, E_idx).unsqueeze(-1)
        mask_1D = mask.view([mask.size(0), mask.size(1), 1, 1])
        mask_bw = mask_1D * mask_attend
        mask_fw = mask_1D * (1.0 - mask_attend)
        h_EXV_encoder_fw = mask_fw * h_EXV_encoder

        for layer in self.decoder_layers:
            h_ESV = cat_neighbors_nodes(h_V, h_ES, E_idx)
            h_ESV = mask_bw * h_ESV + h_EXV_encoder_fw
            h_V = torch.utils.checkpoint.checkpoint(layer, h_V, h_ESV, mask)

        logits_base = self.W_out_base(h_V)
        logits_experts = torch.cat([expert(h_V) for expert in self.experts], dim=-1)
        return logits_base, logits_experts


# =============================================================================
# 2. Dataset / featurize
# =============================================================================
class JSONLDataset(Dataset):
    def __init__(self, jsonl_file):
        self.data = []
        with open(jsonl_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    self.data.append(json.loads(line))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def collate_fn(batch):
    return batch


def get_name(b, i=0):
    return b.get("name") or b.get("pdb") or b.get("pdb_id") or b.get("id") or f"sample_{i}"


def chain_ids_from_record(b):
    ordered = []
    for cid in b.get("masked_list", []):
        if f"seq_chain_{cid}" in b and cid not in ordered:
            ordered.append(cid)
    for cid in b.get("visible_list", []):
        if f"seq_chain_{cid}" in b and cid not in ordered:
            ordered.append(cid)
    for k in sorted(b.keys()):
        if k.startswith("seq_chain_"):
            cid = k.replace("seq_chain_", "")
            if cid not in ordered:
                ordered.append(cid)
    if not ordered and "seq_chain_A" in b:
        ordered = ["A"]
    return ordered


def choose_chains(b, all_chains, eval_chains, max_peptide_len, chain_ids):
    if eval_chains == "masked":
        selected = [c for c in b.get("masked_list", []) if c in all_chains]
        if selected:
            return selected
        return ["A"] if "A" in all_chains else list(all_chains)

    if eval_chains == "short":
        return [c for c in all_chains if 0 < len(b.get(f"seq_chain_{c}", "")) <= max_peptide_len]

    if eval_chains == "all":
        return list(all_chains)

    if eval_chains == "chain":
        wanted = [x.strip() for x in (chain_ids or "").split(",") if x.strip()]
        return [c for c in wanted if c in all_chains]

    raise ValueError(eval_chains)


def prepare_batch(batch, eval_chains, max_peptide_len, chain_ids):
    out = []
    meta = []

    for i, b0 in enumerate(batch):
        b = copy.deepcopy(b0)
        all_chains = chain_ids_from_record(b)
        selected = choose_chains(b, all_chains, eval_chains, max_peptide_len, chain_ids)
        visible = [c for c in all_chains if c not in set(selected)]

        # 单体 masked 模式不动原 masked_list；复合物 short/chain/all 重写 masked/visible
        if eval_chains != "masked" or chain_ids:
            b["masked_list"] = selected
            b["visible_list"] = visible

        order = b.get("masked_list", []) + b.get("visible_list", [])
        if not order:
            order = all_chains
        b["seq"] = "".join(b.get(f"seq_chain_{c}", "") for c in order)

        chain_info = []
        for c in order:
            flag = "M" if c in b.get("masked_list", []) else "V"
            chain_info.append(f"{c}:{len(b.get(f'seq_chain_{c}', ''))}{flag}")

        meta.append({
            "name": get_name(b, i),
            "chains": ",".join(chain_info),
            "selected": list(b.get("masked_list", [])),
        })
        out.append(b)

    return out, meta


def featurize_batch(batch, device, eval_chains, max_peptide_len, chain_ids):
    batch, meta = prepare_batch(batch, eval_chains, max_peptide_len, chain_ids)

    batch2, meta2 = [], []
    for b, m in zip(batch, meta):
        if any(k.startswith("seq_chain_") and len(str(v)) > 0 for k, v in b.items()):
            batch2.append(b)
            meta2.append(m)
    batch, meta = batch2, meta2

    if not batch:
        return None, None

    B = len(batch)
    L_max = max(len(b["seq"]) if "seq" in b else len(b.get("seq_chain_A", "")) for b in batch)

    X = np.zeros([B, L_max, 4, 3], dtype=np.float32)
    S = np.zeros([B, L_max], dtype=np.int32)
    chain_M = np.zeros([B, L_max], dtype=np.float32)
    residue_idx = -100 * np.ones([B, L_max], dtype=np.int32)
    chain_encoding_all = np.zeros([B, L_max], dtype=np.int32)
    real_pos = np.zeros([B, L_max], dtype=np.float32)

    x_default = EXTENDED_AA_TO_INDEX.get("X", 40)

    for i, b in enumerate(batch):
        all_chains = b.get("masked_list", []) + b.get("visible_list", [])
        if not all_chains:
            all_chains = chain_ids_from_record(b)
        if not all_chains:
            all_chains = ["A"]

        l_p = 0
        for c_i, c_id in enumerate(all_chains):
            seq = b.get(f"seq_chain_{c_id}", "")

            N = np.array(b.get(f"N_chain_{c_id}", []), dtype=np.float32)
            CA = np.array(b.get(f"CA_chain_{c_id}", []), dtype=np.float32)
            C = np.array(b.get(f"C_chain_{c_id}", []), dtype=np.float32)
            O = np.array(b.get(f"O_chain_{c_id}", []), dtype=np.float32)

            l = min(len(seq), len(CA))
            if l == 0:
                continue

            if len(N) > 0 and len(CA) > 0 and len(O) > 0:
                dist_n_ca = np.linalg.norm(N[:1] - CA[:1])
                dist_n_o = np.linalg.norm(N[:1] - O[:1])
                if dist_n_o < dist_n_ca and dist_n_o < 1.6:
                    CA, O = O, CA

            X[i, l_p:l_p+l, 0, :] = N[:l]
            X[i, l_p:l_p+l, 1, :] = CA[:l]
            X[i, l_p:l_p+l, 2, :] = C[:l]
            X[i, l_p:l_p+l, 3, :] = O[:l]

            S[i, l_p:l_p+l] = [EXTENDED_AA_TO_INDEX.get(aa, x_default) for aa in seq[:l]]

            if c_id in b.get("masked_list", []):
                chain_M[i, l_p:l_p+l] = 1.0

            real_pos[i, l_p:l_p+l] = 1.0
            residue_idx[i, l_p:l_p+l] = np.arange(l) + c_i * 100
            chain_encoding_all[i, l_p:l_p+l] = c_i

            l_p += l

    isnan = np.isnan(X)
    legacy_mask = np.isfinite(np.sum(X, (2, 3))).astype(np.float32)
    X[isnan] = 0.0

    # clean valid 会用 real_pos & chain_M；legacy_mask 只用于保持模型输入兼容
    features = [
        torch.from_numpy(X).to(device=device, dtype=torch.float32),
        torch.from_numpy(S).to(device=device, dtype=torch.long),
        torch.from_numpy(legacy_mask).to(device=device, dtype=torch.float32),
        torch.from_numpy(chain_M).to(device=device, dtype=torch.float32),
        torch.from_numpy(residue_idx).to(device=device, dtype=torch.long),
        torch.from_numpy(chain_encoding_all).to(device=device, dtype=torch.long),
        torch.from_numpy(real_pos).to(device=device, dtype=torch.float32),
    ]
    return features, meta


# =============================================================================
# 3. metrics
# =============================================================================
def naturalize_tensor(S):
    out = S.clone()
    offset = len(NATURAL_AA_ALPHABET)
    for m_rel, n_idx in NMETHYL_TO_NATURAL_MAPPING.items():
        out[out == int(m_rel) + offset] = int(n_idx)
    out[out >= offset] = 0
    return out


def safe_div(a, b):
    return float(a) / float(b) if b else 0.0


def binary_metrics(y_true, prob, thresholds):
    y_true = np.asarray(y_true, dtype=np.int64)
    prob = np.asarray(prob, dtype=np.float32)

    rows = []
    for thr in thresholds:
        pred = (prob > thr).astype(np.int64)

        TP = int(np.sum((y_true == 1) & (pred == 1)))
        TN = int(np.sum((y_true == 0) & (pred == 0)))
        FP = int(np.sum((y_true == 0) & (pred == 1)))
        FN = int(np.sum((y_true == 1) & (pred == 0)))

        acc = safe_div(TP + TN, TP + TN + FP + FN) * 100
        precision = safe_div(TP, TP + FP) * 100
        recall = safe_div(TP, TP + FN) * 100
        f1 = safe_div(2 * precision * recall, precision + recall)
        pred_ratio = float(np.mean(pred)) * 100 if len(pred) else 0.0
        fpr = safe_div(FP, FP + TN) * 100

        rows.append({
            "thr": thr,
            "acc": acc,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "pred_ratio": pred_ratio,
            "fpr": fpr,
            "TP": TP, "TN": TN, "FP": FP, "FN": FN,
        })
    return rows


def print_binary_table(title, y_true, prob, thresholds):
    y_true = np.asarray(y_true, dtype=np.int64)
    prob = np.asarray(prob, dtype=np.float32)

    print("\n" + "=" * 118)
    print(title)
    print("=" * 118)
    print(f"n_positions = {len(y_true)}")
    print(f"positive methyl = {int(y_true.sum())}, negative = {int((1-y_true).sum())}, pos_ratio = {np.mean(y_true)*100 if len(y_true) else 0:.2f}%")
    print(f"prob: mean={prob.mean() if len(prob) else 0:.4f}, median={np.median(prob) if len(prob) else 0:.4f}, min={prob.min() if len(prob) else 0:.4f}, max={prob.max() if len(prob) else 0:.4f}")

    if int(y_true.sum()) == 0:
        print("⚠️ 没有甲基化正样本：Precision/Recall/F1 对 positive 没意义；主要看 FPR/Pred methyl%。")

    print("-" * 118)
    print(f"{'Thr':<8} | {'Acc':>8} | {'Precision':>10} | {'Recall':>8} | {'F1':>8} | {'Pred methyl%':>12} | {'FPR%':>8} | TP/TN/FP/FN")
    print("-" * 118)

    rows = binary_metrics(y_true, prob, thresholds)
    for r in rows:
        print(
            f"{r['thr']:<8.2f} | {r['acc']:>7.2f}% | {r['precision']:>9.2f}% | "
            f"{r['recall']:>7.2f}% | {r['f1']:>7.2f}% | {r['pred_ratio']:>11.2f}% | "
            f"{r['fpr']:>7.2f}% | {r['TP']}/{r['TN']}/{r['FP']}/{r['FN']}"
        )

    best = max(rows, key=lambda x: x["f1"]) if rows else None
    if best and int(y_true.sum()) > 0:
        print("-" * 118)
        print(f"Best F1 threshold = {best['thr']:.2f}, F1 = {best['f1']:.2f}%, Precision = {best['precision']:.2f}%, Recall = {best['recall']:.2f}%")
    elif rows:
        best_fpr = min(rows, key=lambda x: (x["fpr"], x["pred_ratio"]))
        print("-" * 118)
        print(f"All-negative set: lowest FPR threshold = {best_fpr['thr']:.2f}, FPR = {best_fpr['fpr']:.2f}%, Pred methyl = {best_fpr['pred_ratio']:.2f}%")


def idx_seq(indices, alphabet):
    s = []
    for x in indices:
        x = int(x)
        s.append(alphabet[x] if 0 <= x < len(alphabet) else "?")
    return "".join(s)



def naturalize_for_input(S):
    """
    用于 forward 输入：
    - 甲基化 token -> 对应天然 AA
    - X 保持 X
    - padding 仍然是 0
    """
    out = S.clone()
    offset = len(NATURAL_AA_ALPHABET)
    x_idx = EXTENDED_AA_TO_INDEX.get("X", None)

    for m_rel, n_idx in NMETHYL_TO_NATURAL_MAPPING.items():
        out[out == int(m_rel) + offset] = int(n_idx)

    # 保险：除了 X 以外，所有 >=20 的异常 token 都转成 A
    bad = out >= offset
    if x_idx is not None:
        bad = bad & (out != x_idx)
    out[bad] = 0
    return out


def evaluate_one_input_mode(model, loader, device, thresholds, eval_chains, max_peptide_len, chain_ids, max_examples, input_mode):
    """
    input_mode:
      leaky_original:
          forward 用原始 S。S 里含小写甲基 token。
          这个会把甲基标签送进 W_s(S)，只能作为诊断，不能作为严格论文指标。
      strict_naturalized:
          forward 前把所有甲基 token 转成天然 AA。
          label 仍然用原始 S 判断 methyl/non-methyl。
          这是“已知序列条件下甲基化预测”的推荐口径。
      strict_x_selected:
          forward 前先 naturalize，再把 selected target positions 置 X。
          这是更接近“未知设计位点”的压力测试。
    """
    offset = len(NATURAL_AA_ALPHABET)
    x_idx = EXTENDED_AA_TO_INDEX.get("X", 40)

    all_y = []
    all_prob_known = []
    all_prob_e2e = []
    all_pred_base = []
    all_true_base = []
    all_target_ext = []

    n_legacy_valid = 0
    n_clean_valid = 0
    n_padding_like = 0

    examples = []
    summary = []

    model.eval()
    with torch.no_grad():
        for batch in loader:
            f, meta = featurize_batch(batch, device, eval_chains, max_peptide_len, chain_ids)
            if f is None:
                continue

            X, S_label, mask, chain_M, residue_idx, chain_encoding_all, real_pos = f

            clean_valid_pre = (mask > 0) & (real_pos > 0) & (chain_M > 0) & (S_label != x_idx)

            if input_mode == "leaky_original":
                S_forward = S_label
            elif input_mode == "strict_naturalized":
                S_forward = naturalize_for_input(S_label)
            elif input_mode == "strict_x_selected":
                S_forward = naturalize_for_input(S_label)
                S_forward = S_forward.clone()
                S_forward[clean_valid_pre] = x_idx
            else:
                raise ValueError(input_mode)

            logits_base, logits_experts = model(X, S_forward, mask, chain_M, residue_idx, chain_encoding_all)

            pred_base = torch.argmax(logits_base, dim=-1)
            true_base = naturalize_tensor(S_label)

            prob_e2e = torch.sigmoid(torch.gather(logits_experts, -1, pred_base.unsqueeze(-1)).squeeze(-1))
            prob_known = torch.sigmoid(torch.gather(logits_experts, -1, true_base.unsqueeze(-1)).squeeze(-1))

            legacy_valid = (mask > 0) & (S_label != x_idx)
            clean_valid = clean_valid_pre

            n_legacy_valid += int(legacy_valid.sum().item())
            n_clean_valid += int(clean_valid.sum().item())
            n_padding_like += int((legacy_valid & (~clean_valid)).sum().item())

            if clean_valid.sum().item() == 0:
                continue

            target_ext = S_label[clean_valid].cpu().numpy()
            y = (target_ext >= offset).astype(np.int64)

            all_target_ext.extend(target_ext.tolist())
            all_y.extend(y.tolist())
            all_true_base.extend(true_base[clean_valid].cpu().numpy().tolist())
            all_pred_base.extend(pred_base[clean_valid].cpu().numpy().tolist())
            all_prob_known.extend(prob_known[clean_valid].cpu().numpy().tolist())
            all_prob_e2e.extend(prob_e2e[clean_valid].cpu().numpy().tolist())

            if len(summary) < 40:
                for m in meta:
                    summary.append(m)
                    if len(summary) >= 40:
                        break

            if len(examples) < max_examples:
                S_np = S_label.cpu().numpy()
                tb_np = true_base.cpu().numpy()
                pb_np = pred_base.cpu().numpy()
                cv_np = clean_valid.cpu().numpy().astype(bool)

                for bi in range(S_np.shape[0]):
                    if len(examples) >= max_examples:
                        break
                    pos = cv_np[bi]
                    if pos.sum() == 0:
                        continue
                    true_s = idx_seq(tb_np[bi][pos], NATURAL_AA_ALPHABET)
                    pred_s = idx_seq(pb_np[bi][pos], NATURAL_AA_ALPHABET)
                    match = "".join("|" if a == b else "." for a, b in zip(true_s, pred_s))
                    examples.append({
                        "name": meta[bi]["name"] if bi < len(meta) else "sample",
                        "selected": meta[bi]["selected"] if bi < len(meta) else [],
                        "true": true_s,
                        "pred": pred_s,
                        "match": match,
                    })

    all_y = np.asarray(all_y, dtype=np.int64)
    all_true_base = np.asarray(all_true_base, dtype=np.int64)
    all_pred_base = np.asarray(all_pred_base, dtype=np.int64)
    all_target_ext = np.asarray(all_target_ext, dtype=np.int64)
    all_prob_known = np.asarray(all_prob_known, dtype=np.float32)
    all_prob_e2e = np.asarray(all_prob_e2e, dtype=np.float32)

    return {
        "input_mode": input_mode,
        "summary": summary,
        "examples": examples,
        "n_legacy_valid": n_legacy_valid,
        "n_clean_valid": n_clean_valid,
        "n_padding_like": n_padding_like,
        "all_y": all_y,
        "all_true_base": all_true_base,
        "all_pred_base": all_pred_base,
        "all_target_ext": all_target_ext,
        "all_prob_known": all_prob_known,
        "all_prob_e2e": all_prob_e2e,
    }


def summarize_best_f1(y_true, prob, thresholds):
    rows = binary_metrics(y_true, prob, thresholds)
    if not rows:
        return None
    return max(rows, key=lambda x: x["f1"])


def print_mode_report(result, thresholds, print_examples=True):
    input_mode = result["input_mode"]
    all_y = result["all_y"]
    all_true_base = result["all_true_base"]
    all_pred_base = result["all_pred_base"]
    all_prob_known = result["all_prob_known"]
    all_prob_e2e = result["all_prob_e2e"]

    print("\n" + "#" * 120)
    print(f"# INPUT MODE = {input_mode}")
    print("#" * 120)

    if input_mode == "leaky_original":
        print("⚠️ 诊断模式：forward 输入 S 里含甲基 token，小写甲基标签会进入 W_s(S)。不能作为严格论文指标。")
    elif input_mode == "strict_naturalized":
        print("✅ 推荐模式：forward 前把甲基 token 转成天然 AA；label 仍用原始 S。适合 known-sequence methylation classifier。")
    elif input_mode == "strict_x_selected":
        print("🧪 压力测试：forward 前把 selected positions 置 X，更接近未知设计位点。")

    print("\n" + "=" * 100)
    print("🧾 Padding audit")
    print("=" * 100)
    print(f"legacy_valid_positions = {result['n_legacy_valid']}")
    print(f"clean_selected_real_positions = {result['n_clean_valid']}")
    print(f"legacy_extra_positions = {result['n_padding_like']}")
    if result["n_legacy_valid"] > 0:
        print(f"legacy_extra_ratio = {result['n_padding_like'] / result['n_legacy_valid'] * 100:.2f}%")

    print("\n" + "=" * 100)
    print("🧬 Base AA recovery on clean selected real positions")
    print("=" * 100)
    if len(all_true_base):
        base_acc = np.mean(all_true_base == all_pred_base) * 100
        print(f"n_positions = {len(all_true_base)}")
        print(f"Base AA recovery = {base_acc:.2f}%")
    else:
        print("没有 clean selected real positions。")

    if print_examples:
        print("\n前几个 true/pred：")
        for ex in result["examples"]:
            print(f"  > {ex['name']} selected={ex['selected']}")
            print(f"    true : {ex['true']}")
            print(f"    pred : {ex['pred']}")
            print(f"    match: {ex['match']}")

    print_binary_table(
        f"A) known_sequence_methyl under {input_mode}",
        all_y,
        all_prob_known,
        thresholds,
    )

    print_binary_table(
        f"B) end_to_end_methyl under {input_mode}",
        all_y,
        all_prob_e2e,
        thresholds,
    )


def evaluate(model, loader, device, thresholds, eval_chains, max_peptide_len, chain_ids, max_examples=6):
    # 同时跑 leaky 和 strict，直接解释“为什么会变高”
    modes = ["leaky_original", "strict_naturalized"]
    # x_selected 是额外压力测试；默认也跑，方便看生成式任务会不会更差
    modes.append("strict_x_selected")

    results = []
    for mode in modes:
        results.append(
            evaluate_one_input_mode(
                model=model,
                loader=loader,
                device=device,
                thresholds=thresholds,
                eval_chains=eval_chains,
                max_peptide_len=max_peptide_len,
                chain_ids=chain_ids,
                max_examples=max_examples,
                input_mode=mode,
            )
        )

    print("\n" + "=" * 100)
    print("📋 Chain summary")
    print("=" * 100)
    for m in results[0]["summary"]:
        print(f"  {m['name']:<36} chains={m['chains']} | selected={m['selected']}")
    if len(results[0]["summary"]) >= 40:
        print("  ...")

    print("\n" + "=" * 120)
    print("🔍 QUICK COMPARISON")
    print("=" * 120)
    print(f"{'input_mode':<22} | {'BaseRec':>8} | {'Known best F1':>14} | {'Known best thr':>14} | {'E2E best F1':>12} | {'E2E best thr':>12}")
    print("-" * 120)
    for r in results:
        base = np.mean(r["all_true_base"] == r["all_pred_base"]) * 100 if len(r["all_true_base"]) else 0
        bk = summarize_best_f1(r["all_y"], r["all_prob_known"], thresholds)
        be = summarize_best_f1(r["all_y"], r["all_prob_e2e"], thresholds)
        print(
            f"{r['input_mode']:<22} | "
            f"{base:>7.2f}% | "
            f"{(bk['f1'] if bk else 0):>13.2f}% | "
            f"{(bk['thr'] if bk else 0):>14.2f} | "
            f"{(be['f1'] if be else 0):>11.2f}% | "
            f"{(be['thr'] if be else 0):>12.2f}"
        )

    print("\n解释：")
    print("  - leaky_original 高，说明原始 S 里的甲基 token 进入了 W_s(S)，模型看到了标签信息。")
    print("  - strict_naturalized 才是推荐论文口径：输入只有天然 AA，不包含 methyl label。")
    print("  - strict_x_selected 是更严格的生成式压力测试。")

    for r in results:
        print_mode_report(r, thresholds, print_examples=(r["input_mode"] != "leaky_original"))

    print("\n" + "=" * 100)
    print("✅ 最终读法")
    print("=" * 100)
    print("1) 论文里不要用 leaky_original。它用于证明为什么分数会异常高。")
    print("2) 已知序列甲基化预测：用 strict_naturalized 的 A) known_sequence_methyl。")
    print("3) 端到端设计+甲基化：用 strict_naturalized 或 strict_x_selected 的 B) end_to_end_methyl，并同时报告 Base AA recovery。")
    print("4) 如果复合物 positive methyl = 0，只能报告 FPR/Pred methyl%，不能报告 recall/F1。")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--test_data", required=True)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--eval_chains", choices=["masked", "short", "all", "chain"], default="masked")
    parser.add_argument("--max_peptide_len", type=int, default=30)
    parser.add_argument("--chain_ids", type=str, default=None)
    parser.add_argument("--thresholds", type=str, default="0.3,0.4,0.5,0.6,0.7,0.8,0.9,0.95,0.98,0.99")
    parser.add_argument("--max_examples", type=int, default=6)
    args = parser.parse_args()

    thresholds = [float(x) for x in args.thresholds.split(",") if x.strip()]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("🚀 device =", device)
    print("📄 test_data =", args.test_data)
    print("🎯 eval_chains =", args.eval_chains)
    print("📏 max_peptide_len =", args.max_peptide_len)
    print("🔗 chain_ids =", args.chain_ids)
    print("🔤 EXTENDED_AA_ALPHABET:", len(EXTENDED_AA_ALPHABET), EXTENDED_AA_ALPHABET)
    print("🔤 X_idx:", EXTENDED_AA_TO_INDEX.get("X", None))
    print("🔤 NMETHYL_TO_NATURAL_MAPPING:", NMETHYL_TO_NATURAL_MAPPING)

    model = RobustHierarchicalProteinMPNN(augment_eps=0.0).to(device)

    checkpoint = torch.load(args.model_path, map_location=device)
    state_dict = checkpoint["model_state_dict"] if "model_state_dict" in checkpoint else checkpoint
    model.load_state_dict(state_dict)
    print("✅ strict load 成功")

    ds = JSONLDataset(args.test_data)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)

    evaluate(
        model=model,
        loader=loader,
        device=device,
        thresholds=thresholds,
        eval_chains=args.eval_chains,
        max_peptide_len=args.max_peptide_len,
        chain_ids=args.chain_ids,
        max_examples=args.max_examples,
    )


if __name__ == "__main__":
    main()

#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
clean_v28_common.py

这个文件只服务于 paper_clean_v28 工作区。
原则：
1. 不改旧代码。
2. 统一使用 nmethyl/utils/nmethyl_config.py 的字母表。
3. 填充位点一律不参与评价。
4. 复合物评价时，模型输入可以包含受体链，但指标只统计被选中的短肽链。
"""

import os
import sys
import json
import copy
import csv
import math
from typing import Dict, List, Tuple, Any, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from model_utils import ProteinMPNN, gather_nodes, cat_neighbors_nodes
from nmethyl.utils.nmethyl_config import (
    EXTENDED_AA_ALPHABET,
    NATURAL_AA_ALPHABET,
    NMETHYL_TO_NATURAL_MAPPING,
    EXTENDED_AA_TO_INDEX,
)

X_TOKEN = "X"
X_INDEX = EXTENDED_AA_TO_INDEX[X_TOKEN]
N_NATURAL = len(NATURAL_AA_ALPHABET)

METHYL_ABS_TO_NAT = {
    int(m_rel) + N_NATURAL: int(n_idx)
    for m_rel, n_idx in NMETHYL_TO_NATURAL_MAPPING.items()
}

NAT_TO_METHYL_ABS = {}
for m_abs, n_idx in METHYL_ABS_TO_NAT.items():
    # 如果同一个天然氨基酸有多个甲基 token，保留第一个。
    NAT_TO_METHYL_ABS.setdefault(int(n_idx), int(m_abs))


class RobustHierarchicalProteinMPNN(ProteinMPNN):
    """与 v28 / frankenstein_v28.pt 对齐的模型结构。"""

    def __init__(self, hidden_dim: int = 128, augment_eps: float = 0.0, **kwargs):
        super().__init__(
            num_letters=21,
            hidden_dim=hidden_dim,
            vocab=21,
            k_neighbors=48,
            augment_eps=augment_eps,
            **kwargs,
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
                layer, h_V, h_E, E_idx, mask, mask_attend, use_reentrant=False
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
            h_V = torch.utils.checkpoint.checkpoint(
                layer, h_V, h_ESV, mask, use_reentrant=False
            )

        logits_base = self.W_out_base(h_V)
        logits_experts = torch.cat([expert(h_V) for expert in self.experts], dim=-1)
        return logits_base, logits_experts


def load_v28_model(model_path: str, device: torch.device) -> RobustHierarchicalProteinMPNN:
    """严格加载 frankenstein_v28.pt。不做切片，不做防弹加载。"""
    model = RobustHierarchicalProteinMPNN(augment_eps=0.0).to(device)
    checkpoint = torch.load(model_path, map_location=device)
    state_dict = checkpoint.get("model_state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing or unexpected:
        raise RuntimeError(
            "模型权重没有严格对齐。\n"
            f"missing={missing}\n"
            f"unexpected={unexpected}\n"
            "最终论文流程不能用防弹切片加载。请确认 frankenstein_v28.pt 和模型结构一致。"
        )
    model.eval()
    return model


def read_jsonl(path: str) -> List[Dict[str, Any]]:
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


def write_csv(path: str, rows: List[Dict[str, Any]], fieldnames: Optional[List[str]] = None) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if fieldnames is None:
        keys = []
        for r in rows:
            for k in r.keys():
                if k not in keys:
                    keys.append(k)
        fieldnames = keys
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)


def write_json(path: str, obj: Any) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def get_record_name(record: Dict[str, Any], idx: int = 0) -> str:
    return str(record.get("name") or record.get("pdb") or record.get("pdb_id") or record.get("id") or f"sample_{idx}")


def chain_ids_from_record(record: Dict[str, Any]) -> List[str]:
    ordered = []
    for cid in record.get("masked_list", []):
        if f"seq_chain_{cid}" in record and cid not in ordered:
            ordered.append(cid)
    for cid in record.get("visible_list", []):
        if f"seq_chain_{cid}" in record and cid not in ordered:
            ordered.append(cid)
    for key in sorted(record.keys()):
        if key.startswith("seq_chain_"):
            cid = key.replace("seq_chain_", "")
            if cid not in ordered:
                ordered.append(cid)
    return ordered


def choose_eval_chains(
    record: Dict[str, Any],
    eval_chains: str,
    max_peptide_len: int = 30,
    chain_ids: Optional[str] = None,
) -> List[str]:
    all_chains = chain_ids_from_record(record)
    if eval_chains == "masked":
        selected = [c for c in record.get("masked_list", []) if c in all_chains]
        return selected if selected else (["A"] if "A" in all_chains else list(all_chains))
    if eval_chains == "short":
        return [c for c in all_chains if 0 < len(record.get(f"seq_chain_{c}", "")) <= max_peptide_len]
    if eval_chains == "all":
        return list(all_chains)
    if eval_chains == "chain":
        wanted = [x.strip() for x in (chain_ids or "").split(",") if x.strip()]
        return [c for c in wanted if c in all_chains]
    raise ValueError(f"未知 eval_chains: {eval_chains}")


def prepare_record_for_eval(
    record: Dict[str, Any],
    eval_chains: str,
    max_peptide_len: int = 30,
    chain_ids: Optional[str] = None,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    b = copy.deepcopy(record)
    all_chains = chain_ids_from_record(b)
    selected = choose_eval_chains(b, eval_chains, max_peptide_len, chain_ids)
    visible = [c for c in all_chains if c not in set(selected)]

    # 为了复合物短肽评价，短肽设为 masked，其余受体链作为 visible。
    # 单体 masked 模式也这样明确写出，避免旧文件里 masked_list 缺失。
    b["masked_list"] = selected
    b["visible_list"] = visible
    order = selected + visible
    b["seq"] = "".join(b.get(f"seq_chain_{c}", "") for c in order)

    chain_info = []
    for c in order:
        tag = "M" if c in selected else "V"
        chain_info.append(f"{c}:{len(b.get(f'seq_chain_{c}', ''))}{tag}")

    meta = {
        "name": get_record_name(b),
        "selected_chains": ",".join(selected),
        "visible_chains": ",".join(visible),
        "chain_info": ";".join(chain_info),
        "selected_length": sum(len(b.get(f"seq_chain_{c}", "")) for c in selected),
        "total_length": sum(len(b.get(f"seq_chain_{c}", "")) for c in order),
    }
    return b, meta


def naturalize_indices_np(arr: np.ndarray) -> np.ndarray:
    out = np.asarray(arr, dtype=np.int64).copy()
    for m_abs, n_idx in METHYL_ABS_TO_NAT.items():
        out[out == m_abs] = n_idx
    out[(out >= N_NATURAL) & (out != X_INDEX)] = 0
    return out


def naturalize_tensor_for_input(S: torch.Tensor) -> torch.Tensor:
    out = S.clone()
    for m_abs, n_idx in METHYL_ABS_TO_NAT.items():
        out[out == int(m_abs)] = int(n_idx)
    bad = (out >= N_NATURAL) & (out != X_INDEX)
    out[bad] = 0
    return out


def token_to_natural_token(token: str) -> str:
    if token in NATURAL_AA_ALPHABET:
        return token
    if token in EXTENDED_AA_TO_INDEX and token != X_TOKEN:
        idx = EXTENDED_AA_TO_INDEX[token]
        if idx in METHYL_ABS_TO_NAT:
            return NATURAL_AA_ALPHABET[METHYL_ABS_TO_NAT[idx]]
    return X_TOKEN


def naturalize_sequence(seq: str) -> str:
    return "".join(token_to_natural_token(ch) for ch in seq)


def methyl_binary_from_indices(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr, dtype=np.int64)
    return (arr >= N_NATURAL).astype(np.int64)


def idx_to_sequence(indices: List[int], natural_only: bool = False) -> str:
    alphabet = NATURAL_AA_ALPHABET if natural_only else EXTENDED_AA_ALPHABET
    chars = []
    for x in indices:
        x = int(x)
        chars.append(alphabet[x] if 0 <= x < len(alphabet) else "?")
    return "".join(chars)


def featurize_records(
    records: List[Dict[str, Any]],
    device: torch.device,
    eval_chains: str,
    max_peptide_len: int = 30,
    chain_ids: Optional[str] = None,
):
    """
    干净特征构建。

    返回：
    X, S, mask, chain_M, residue_idx, chain_encoding_all, real_pos, meta

    重要定义：
    - mask：所有真实残基为 1，包括受体链和短肽链。只给模型看真实位置。
    - chain_M：被评价或被设计的链为 1。
    - real_pos：所有真实残基为 1。主要用于审计。
    - 指标统计位点：mask & chain_M & S != X。
    """
    prepared = []
    metas = []
    for r in records:
        b, m = prepare_record_for_eval(r, eval_chains, max_peptide_len, chain_ids)
        if m["total_length"] > 0 and m["selected_length"] > 0:
            prepared.append(b)
            metas.append(m)

    if not prepared:
        return None

    B = len(prepared)
    L_max = max(len(b["seq"]) for b in prepared)

    X = np.zeros([B, L_max, 4, 3], dtype=np.float32)
    S = np.full([B, L_max], X_INDEX, dtype=np.int32)
    mask = np.zeros([B, L_max], dtype=np.float32)
    chain_M = np.zeros([B, L_max], dtype=np.float32)
    residue_idx = -100 * np.ones([B, L_max], dtype=np.int32)
    chain_encoding_all = np.zeros([B, L_max], dtype=np.int32)
    real_pos = np.zeros([B, L_max], dtype=np.float32)

    for i, b in enumerate(prepared):
        chain_order = b.get("masked_list", []) + b.get("visible_list", [])
        l_p = 0
        for c_i, c_id in enumerate(chain_order):
            seq = b.get(f"seq_chain_{c_id}", "")
            if not seq:
                continue

            coords_by_atom = []
            min_len = len(seq)
            for atom_name in ["N", "CA", "C", "O"]:
                arr = np.asarray(b.get(f"{atom_name}_chain_{c_id}", []), dtype=np.float32)
                coords_by_atom.append(arr)
                min_len = min(min_len, len(arr))

            if min_len <= 0:
                continue

            N, CA, C, O = [a[:min_len] for a in coords_by_atom]

            # 有些数据 CA 和 O 可能交换，保留旧代码里的修正。
            if len(N) > 0 and len(CA) > 0 and len(O) > 0:
                dist_n_ca = np.linalg.norm(N[:1] - CA[:1])
                dist_n_o = np.linalg.norm(N[:1] - O[:1])
                if dist_n_o < dist_n_ca and dist_n_o < 1.6:
                    CA, O = O, CA

            X[i, l_p:l_p + min_len, 0, :] = N
            X[i, l_p:l_p + min_len, 1, :] = CA
            X[i, l_p:l_p + min_len, 2, :] = C[:min_len]
            X[i, l_p:l_p + min_len, 3, :] = O

            seq_part = seq[:min_len]
            S[i, l_p:l_p + min_len] = [EXTENDED_AA_TO_INDEX.get(aa, X_INDEX) for aa in seq_part]
            mask[i, l_p:l_p + min_len] = 1.0
            real_pos[i, l_p:l_p + min_len] = 1.0
            if c_id in b.get("masked_list", []):
                chain_M[i, l_p:l_p + min_len] = 1.0
            residue_idx[i, l_p:l_p + min_len] = np.arange(min_len) + c_i * 100
            chain_encoding_all[i, l_p:l_p + min_len] = c_i
            l_p += min_len

    tensors = [
        torch.from_numpy(X).to(device=device, dtype=torch.float32),
        torch.from_numpy(S).to(device=device, dtype=torch.long),
        torch.from_numpy(mask).to(device=device, dtype=torch.float32),
        torch.from_numpy(chain_M).to(device=device, dtype=torch.float32),
        torch.from_numpy(residue_idx).to(device=device, dtype=torch.long),
        torch.from_numpy(chain_encoding_all).to(device=device, dtype=torch.long),
        torch.from_numpy(real_pos).to(device=device, dtype=torch.float32),
    ]
    return tensors, metas


def binary_metrics(y_true: np.ndarray, prob: np.ndarray, thresholds: List[float]) -> List[Dict[str, Any]]:
    y_true = np.asarray(y_true, dtype=np.int64)
    prob = np.asarray(prob, dtype=np.float32)
    rows = []
    for thr in thresholds:
        pred = (prob > thr).astype(np.int64)
        tp = int(np.sum((y_true == 1) & (pred == 1)))
        tn = int(np.sum((y_true == 0) & (pred == 0)))
        fp = int(np.sum((y_true == 0) & (pred == 1)))
        fn = int(np.sum((y_true == 1) & (pred == 0)))
        total = tp + tn + fp + fn
        acc = (tp + tn) / total if total else 0.0
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        fpr = fp / (fp + tn) if (fp + tn) else 0.0
        rows.append({
            "threshold": thr,
            "accuracy": acc,
            "precision": prec,
            "recall": rec,
            "f1": f1,
            "false_positive_rate": fpr,
            "pred_methyl_rate": float(pred.mean()) if len(pred) else 0.0,
            "tp": tp,
            "tn": tn,
            "fp": fp,
            "fn": fn,
        })
    return rows


def roc_auc_score_simple(y_true: np.ndarray, scores: np.ndarray) -> Optional[float]:
    """不用 sklearn 的二分类曲线下面积。没有正负两类时返回 None。"""
    y_true = np.asarray(y_true, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float64)
    n_pos = int(np.sum(y_true == 1))
    n_neg = int(np.sum(y_true == 0))
    if n_pos == 0 or n_neg == 0:
        return None
    order = np.argsort(scores)
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(scores) + 1)

    # 处理并列分数：同分取平均 rank。
    sorted_scores = scores[order]
    start = 0
    while start < len(scores):
        end = start + 1
        while end < len(scores) and sorted_scores[end] == sorted_scores[start]:
            end += 1
        avg_rank = (start + 1 + end) / 2.0
        ranks[order[start:end]] = avg_rank
        start = end

    sum_ranks_pos = float(np.sum(ranks[y_true == 1]))
    auc = (sum_ranks_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    return float(auc)


def parse_fasta(path: str) -> List[Tuple[str, str]]:
    records = []
    header = None
    seq_chunks = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if header is not None:
                    records.append((header, "".join(seq_chunks)))
                header = line[1:]
                seq_chunks = []
            else:
                seq_chunks.append(line)
    if header is not None:
        records.append((header, "".join(seq_chunks)))
    return records


def sequence_recovery(true_seq: str, pred_seq: str, naturalize: bool = True) -> Optional[float]:
    if naturalize:
        true_seq = naturalize_sequence(true_seq)
        pred_seq = naturalize_sequence(pred_seq)
    if len(true_seq) != len(pred_seq) or len(true_seq) == 0:
        return None
    return sum(a == b for a, b in zip(true_seq, pred_seq)) / len(true_seq)


def methyl_count(seq: str) -> int:
    return sum(1 for ch in seq if ch in EXTENDED_AA_TO_INDEX and EXTENDED_AA_TO_INDEX[ch] >= N_NATURAL and ch != X_TOKEN)

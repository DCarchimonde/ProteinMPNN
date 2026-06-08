#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
eval_methyl_sweep_legacy_vs_selected.py

为什么要有这个脚本：
- 你的 nmethyl/sweep_threshold_robust.py 的 38% 是 legacy 统计口径。
- legacy 口径使用：
      valid = mask_flat & (tgts != x_idx)
  它不限制 chain_M，也不区分真实填充位点和 padding 位点。
- 这个脚本同时输出两个口径：
      1) legacy     : 尽量复现你原来 sweep_threshold_robust.py 的 38%
      2) selected   : 只统计真正被设计/评估的链和真实填充残基

保存到：
    ProteinMPNN-main/nmethyl/eval_methyl_sweep_legacy_vs_selected.py

单体复现 38%：
    python nmethyl/eval_methyl_sweep_legacy_vs_selected.py \
        --model_path "./frankenstein_v28.pt" \
        --test_data "nmethyl_data/test_set/test.jsonl" \
        --eval_chains masked \
        --batch_size 16

复合物只看短肽链：
    python nmethyl/eval_methyl_sweep_legacy_vs_selected.py \
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
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("请确认脚本放在 ProteinMPNN-main/nmethyl/ 下，并从 ProteinMPNN-main 根目录运行。")
    sys.exit(1)


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
            nn.Linear(hidden_dim, len(NATURAL_AA_ALPHABET))
        )

        self.experts = nn.ModuleList([
            nn.Linear(hidden_dim, 1) for _ in range(len(NATURAL_AA_ALPHABET))
        ])

    def forward(self, X, S, mask, chain_M, residue_idx, chain_encoding_all):
        # 和 sweep_threshold_robust.py 一样，用 ProteinMPNN.features
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

        mask_size = E_idx.shape[1]
        permutation_matrix_reverse = torch.nn.functional.one_hot(decoding_order, num_classes=mask_size).float()
        order_mask_backward = torch.einsum(
            'ij, biq, bjp->bqp',
            (1 - torch.triu(torch.ones(mask_size, mask_size, device=X.device))),
            permutation_matrix_reverse,
            permutation_matrix_reverse
        )
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
        logits_experts = torch.cat([expert(h_V) for expert in self.experts], dim=-1)
        return logits_base, logits_experts


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


def get_name(b, idx=0):
    return b.get("name") or b.get("pdb") or b.get("pdb_id") or b.get("id") or f"sample_{idx}"


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

        # 单体 masked 模式：尽量不改原数据
        # 复合物 short/chain/all：把要评估的链设为 masked，其余 visible
        if eval_chains != "masked" or chain_ids:
            b["masked_list"] = selected
            b["visible_list"] = visible

        # 这个 seq 只用于 L_max；顺序和实际 masked+visible 一致
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
    # 这里保留 sweep_threshold_robust.py 的核心行为：
    # 1) L_max 取 len(seq)。
    # 2) legacy mask 依然用 np.isfinite(np.sum(X))，padding 0 坐标会被当成 valid。
    # 3) 额外返回 real_pos，用于真正残基统计。
    batch, meta = prepare_batch(batch, eval_chains, max_peptide_len, chain_ids)

    # 和原代码一致：要求有 seq_chain_A。复合物都有 A；单体也有 A。
    batch2, meta2 = [], []
    for b, m in zip(batch, meta):
        if "seq_chain_A" in b and len(b["seq_chain_A"]) > 0:
            batch2.append(b)
            meta2.append(m)
    batch, meta = batch2, meta2

    if not batch:
        return None, None

    B = len(batch)
    lengths = [len(b["seq"]) if "seq" in b else len(b["seq_chain_A"]) for b in batch]
    L_max = max(lengths)

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

    # legacy mask: 这是原 sweep_threshold_robust.py 的行为
    mask = np.isfinite(np.sum(X, (2, 3))).astype(np.float32)
    X[isnan] = 0.0

    features = [
        torch.from_numpy(X).to(dtype=torch.float32, device=device),
        torch.from_numpy(S).to(dtype=torch.long, device=device),
        torch.from_numpy(mask).to(dtype=torch.float32, device=device),
        torch.from_numpy(chain_M).to(dtype=torch.float32, device=device),
        torch.from_numpy(residue_idx).to(dtype=torch.long, device=device),
        torch.from_numpy(chain_encoding_all).to(dtype=torch.long, device=device),
        torch.from_numpy(real_pos).to(dtype=torch.float32, device=device),
    ]
    return features, meta


def naturalize_np(t):
    t = np.array(t, dtype=np.int64)
    out = t.copy()
    offset = len(NATURAL_AA_ALPHABET)
    for m_rel, n_idx in NMETHYL_TO_NATURAL_MAPPING.items():
        out[out == int(m_rel) + offset] = int(n_idx)
    return out


def idx_seq(indices, alphabet):
    s = []
    for x in indices:
        x = int(x)
        s.append(alphabet[x] if 0 <= x < len(alphabet) else "?")
    return "".join(s)


def collect_predictions(model, loader, device, eval_chains, max_peptide_len, chain_ids, max_examples):
    model.eval()

    offset = len(NATURAL_AA_ALPHABET)
    x_idx = EXTENDED_AA_TO_INDEX.get("X", 40)

    # 原 sweep 里的映射写法
    methyl_idx_to_nat_idx = {
        int(m) + offset: int(n)
        for m, n in NMETHYL_TO_NATURAL_MAPPING.items()
    }

    buckets = {
        "legacy": {"t": [], "pb": [], "pm": [], "examples": [], "n_batches": 0},
        "selected": {"t": [], "pb": [], "pm": [], "examples": [], "n_batches": 0},
    }
    summary = []

    with torch.no_grad():
        for batch in loader:
            f, meta = featurize_batch(batch, device, eval_chains, max_peptide_len, chain_ids)
            if f is None:
                continue

            X, S, mask, chain_M, residue_idx, chain_encoding_all, real_pos = f

            l_base, l_experts = model(X, S, mask, chain_M, residue_idx, chain_encoding_all)

            pred_base_idx = torch.argmax(l_base, -1)
            expert_logit = torch.gather(l_experts, -1, pred_base_idx.unsqueeze(-1)).squeeze(-1)
            prob_methyl = torch.sigmoid(expert_logit)

            S_np = S.cpu().numpy()
            pb_np = pred_base_idx.cpu().numpy()
            pm_np = prob_methyl.cpu().numpy()

            mask_np = mask.cpu().numpy().astype(bool)
            chain_M_np = chain_M.cpu().numpy().astype(bool)
            real_np = real_pos.cpu().numpy().astype(bool)

            valid_legacy = mask_np & (S_np != x_idx)
            valid_selected = mask_np & real_np & chain_M_np & (S_np != x_idx)

            for mode, valid in [("legacy", valid_legacy), ("selected", valid_selected)]:
                buckets[mode]["t"].extend(S_np[valid].reshape(-1).tolist())
                buckets[mode]["pb"].extend(pb_np[valid].reshape(-1).tolist())
                buckets[mode]["pm"].extend(pm_np[valid].reshape(-1).tolist())
                buckets[mode]["n_batches"] += 1

                if len(buckets[mode]["examples"]) < max_examples:
                    for bi in range(S_np.shape[0]):
                        if len(buckets[mode]["examples"]) >= max_examples:
                            break
                        pos = valid[bi]
                        if pos.sum() == 0:
                            continue
                        true_base = naturalize_np(S_np[bi][pos])
                        pred_base = pb_np[bi][pos]
                        true_s = idx_seq(true_base, NATURAL_AA_ALPHABET)
                        pred_s = idx_seq(pred_base, NATURAL_AA_ALPHABET)
                        match = "".join("|" if a == b else "." for a, b in zip(true_s, pred_s))
                        name = meta[bi]["name"] if bi < len(meta) else "sample"
                        selected = meta[bi]["selected"] if bi < len(meta) else []
                        buckets[mode]["examples"].append({
                            "name": name,
                            "selected": selected,
                            "true": true_s,
                            "pred": pred_s,
                            "match": match,
                        })

            if len(summary) < 40:
                for m in meta:
                    summary.append(m)
                    if len(summary) >= 40:
                        break

    return buckets, summary, methyl_idx_to_nat_idx


def report_one_mode(name, data, thresholds, methyl_idx_to_nat_idx):
    offset = len(NATURAL_AA_ALPHABET)
    t = np.array(data["t"], dtype=np.int64)
    pb = np.array(data["pb"], dtype=np.int64)
    pm = np.array(data["pm"], dtype=np.float32)

    if len(t) == 0:
        print(f"\n❌ {name}: 没有有效位点")
        return

    true_base = naturalize_np(t)
    base_acc = np.mean(pb == true_base) * 100.0
    true_methyl_ratio = np.mean(t >= offset) * 100.0

    print("\n" + "=" * 96)
    print(f"📌 MODE = {name}")
    print("=" * 96)

    if name == "legacy":
        print("说明：legacy 口径尽量复现原 sweep_threshold_robust.py：valid = mask & non-X。")
        print("      它可能包含 padding/未填充位点，适合复现旧表，但不建议当真实残基 recovery。")
    else:
        print("说明：selected 口径只统计 real_pos & chain_M & non-X，即真正选择的真实残基。")

    print(f"评估位点数: {len(t)}")
    print(f"真实 methyl ratio: {true_methyl_ratio:.2f}%")
    print(f"Base AA recovery: {base_acc:.2f}%")

    print("\n前几个例子：")
    for ex in data["examples"]:
        print(f"  > {ex['name']} selected={ex['selected']}")
        print(f"    true : {ex['true']}")
        print(f"    pred : {ex['pred']}")
        print(f"    match: {ex['match']}")

    print("\n" + "-" * 96)
    print(f"{'Threshold':<10} | {'Pred methyl%':>12} | {'Base Rec':>10} | {'Total End-to-End Acc':>22}")
    print("-" * 96)

    best_acc = -1.0
    best_thr = None

    for thr in thresholds:
        pred_is_methyl = (pm > thr).astype(int)
        final_pred = pb.copy()

        for m_abs_idx, n_idx in methyl_idx_to_nat_idx.items():
            update = (pred_is_methyl == 1) & (pb == n_idx)
            final_pred[update] = m_abs_idx

        pred_ratio = np.mean(pred_is_methyl) * 100.0
        total_acc = np.mean(final_pred == t) * 100.0

        if total_acc > best_acc:
            best_acc = total_acc
            best_thr = thr

        marker = "⭐" if abs(pred_ratio - true_methyl_ratio) < 10 else ""
        print(f"Thr={thr:<5.2f} | {pred_ratio:>11.2f}% {marker:<2} | {base_acc:>9.2f}% | {total_acc:>21.2f}%")

    print("-" * 96)
    print(f"Best threshold = {best_thr:.2f}, Best Total Acc = {best_acc:.2f}%")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--test_data", required=True)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--eval_chains", choices=["masked", "short", "all", "chain"], default="masked")
    parser.add_argument("--max_peptide_len", type=int, default=30)
    parser.add_argument("--chain_ids", type=str, default=None)
    parser.add_argument("--thresholds", type=str, default="0.3,0.4,0.5,0.6,0.7,0.8,0.9,0.95,0.98,0.99")
    parser.add_argument("--max_examples", type=int, default=5)
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
    print("✅ strict load 成功，加载方式和 sweep_threshold_robust.py 一致")

    ds = JSONLDataset(args.test_data)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)

    buckets, summary, methyl_idx_to_nat_idx = collect_predictions(
        model=model,
        loader=loader,
        device=device,
        eval_chains=args.eval_chains,
        max_peptide_len=args.max_peptide_len,
        chain_ids=args.chain_ids,
        max_examples=args.max_examples,
    )

    print("\n" + "=" * 96)
    print("📋 Chain summary")
    print("=" * 96)
    for m in summary:
        print(f"  {m['name']:<36} chains={m['chains']} | selected={m['selected']}")
    if len(ds) > len(summary):
        print("  ...")

    # legacy 一定先打印：用于验证能不能复现旧 sweep 的 38%
    report_one_mode("legacy", buckets["legacy"], thresholds, methyl_idx_to_nat_idx)

    # selected 是真实残基/选中链结果
    report_one_mode("selected", buckets["selected"], thresholds, methyl_idx_to_nat_idx)

    print("\n✅ 读数方式：")
    print("   - 单体如果 legacy 接近 38%，说明已经复现你原来的 sweep_threshold_robust.py。")
    print("   - selected 才是真正只看真实残基/被选链的 recovery。")
    print("   - 复合物建议主要看 selected，不建议用 legacy。")


if __name__ == "__main__":
    main()

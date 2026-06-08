#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
01_eval_clean_model.py

针对最终模型 frankenstein_v28.pt 的干净评价脚本。

特点：
1. 严格加载模型，不允许防弹切片。
2. 只统计真实残基位置。
3. 复合物只统计被选中的短肽链，不把受体链混入指标。
4. 同时输出三种口径：
   - 原始输入：用于审计，不作为论文主结果。
   - 天然化输入：推荐的已知序列甲基化预测口径。
   - 选中位点置 X：更严格的端到端压力测试。

运行示例见 paper_clean_v28/README.md。
"""

import os
import argparse
from typing import List, Dict, Any

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from clean_v28_common import (
    EXTENDED_AA_ALPHABET,
    NATURAL_AA_ALPHABET,
    X_INDEX,
    N_NATURAL,
    METHYL_ABS_TO_NAT,
    NAT_TO_METHYL_ABS,
    load_v28_model,
    read_jsonl,
    write_csv,
    write_json,
    featurize_records,
    naturalize_indices_np,
    naturalize_tensor_for_input,
    binary_metrics,
    roc_auc_score_simple,
)


class JsonlDataset(Dataset):
    def __init__(self, path: str):
        self.data = read_jsonl(path)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def collate_fn(batch):
    return batch


def best_row(rows: List[Dict[str, Any]], key: str = "f1") -> Dict[str, Any]:
    if not rows:
        return {}
    return max(rows, key=lambda r: r.get(key, 0.0))


def run_input_mode(model, loader, device, args, input_mode: str, thresholds: List[float]):
    position_rows = []
    meta_rows = []

    all_target_ext = []
    all_true_base = []
    all_pred_base = []
    all_prob_known = []
    all_prob_e2e = []

    total_real = 0
    total_selected = 0
    total_padding = 0

    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            packed = featurize_records(
                batch,
                device=device,
                eval_chains=args.eval_chains,
                max_peptide_len=args.max_peptide_len,
                chain_ids=args.chain_ids,
            )
            if packed is None:
                continue
            tensors, metas = packed
            X, S_label, mask, chain_M, residue_idx, chain_encoding_all, real_pos = tensors

            valid_real = (mask > 0) & (real_pos > 0) & (S_label != X_INDEX)
            valid_selected = valid_real & (chain_M > 0)
            total_real += int(valid_real.sum().item())
            total_selected += int(valid_selected.sum().item())
            total_padding += int(((mask <= 0) | (real_pos <= 0)).sum().item())

            if input_mode == "raw_original_input":
                S_forward = S_label.clone()
            elif input_mode == "strict_naturalized_input":
                S_forward = naturalize_tensor_for_input(S_label)
            elif input_mode == "selected_positions_as_X":
                S_forward = naturalize_tensor_for_input(S_label)
                S_forward = S_forward.clone()
                S_forward[valid_selected] = X_INDEX
            else:
                raise ValueError(input_mode)

            logits_base, logits_experts = model(X, S_forward, mask, chain_M, residue_idx, chain_encoding_all)
            pred_base = torch.argmax(logits_base, dim=-1)
            true_base_tensor = naturalize_tensor_for_input(S_label)

            prob_known = torch.sigmoid(torch.gather(logits_experts, -1, true_base_tensor.unsqueeze(-1)).squeeze(-1))
            prob_e2e = torch.sigmoid(torch.gather(logits_experts, -1, pred_base.unsqueeze(-1)).squeeze(-1))

            for bi, meta in enumerate(metas):
                meta_rows.append({
                    "input_mode": input_mode,
                    "batch_index": batch_idx,
                    "sample_index_in_batch": bi,
                    **meta,
                })

                pos_idx = torch.where(valid_selected[bi])[0]
                for pos in pos_idx.cpu().numpy().tolist():
                    target_ext = int(S_label[bi, pos].item())
                    true_base = int(true_base_tensor[bi, pos].item())
                    pred_b = int(pred_base[bi, pos].item())
                    pk = float(prob_known[bi, pos].item())
                    pe = float(prob_e2e[bi, pos].item())
                    is_methyl = int(target_ext >= N_NATURAL)
                    position_rows.append({
                        "input_mode": input_mode,
                        "sample_name": meta["name"],
                        "selected_chains": meta["selected_chains"],
                        "position_in_model": pos,
                        "target_token_index": target_ext,
                        "target_token": EXTENDED_AA_ALPHABET[target_ext] if target_ext < len(EXTENDED_AA_ALPHABET) else "?",
                        "true_base_index": true_base,
                        "true_base_token": NATURAL_AA_ALPHABET[true_base] if true_base < len(NATURAL_AA_ALPHABET) else "?",
                        "pred_base_index": pred_b,
                        "pred_base_token": NATURAL_AA_ALPHABET[pred_b] if pred_b < len(NATURAL_AA_ALPHABET) else "?",
                        "base_correct": int(pred_b == true_base),
                        "is_methyl_true": is_methyl,
                        "prob_methyl_known_sequence": pk,
                        "prob_methyl_end_to_end": pe,
                    })
                    all_target_ext.append(target_ext)
                    all_true_base.append(true_base)
                    all_pred_base.append(pred_b)
                    all_prob_known.append(pk)
                    all_prob_e2e.append(pe)

    all_target_ext = np.asarray(all_target_ext, dtype=np.int64)
    all_true_base = np.asarray(all_true_base, dtype=np.int64)
    all_pred_base = np.asarray(all_pred_base, dtype=np.int64)
    y_true = (all_target_ext >= N_NATURAL).astype(np.int64)
    prob_known = np.asarray(all_prob_known, dtype=np.float32)
    prob_e2e = np.asarray(all_prob_e2e, dtype=np.float32)

    n_positions = int(len(all_target_ext))
    base_recovery = float(np.mean(all_true_base == all_pred_base)) if n_positions else 0.0
    methyl_rate = float(np.mean(y_true)) if n_positions else 0.0

    known_rows = binary_metrics(y_true, prob_known, thresholds)
    e2e_rows = binary_metrics(y_true, prob_e2e, thresholds)
    for r in known_rows:
        r["input_mode"] = input_mode
        r["task"] = "known_sequence_methylation"
    for r in e2e_rows:
        r["input_mode"] = input_mode
        r["task"] = "end_to_end_methylation"

    auc_known = roc_auc_score_simple(y_true, prob_known)
    auc_e2e = roc_auc_score_simple(y_true, prob_e2e)

    summary = {
        "input_mode": input_mode,
        "n_selected_positions": n_positions,
        "n_real_positions_all_chains": total_real,
        "n_selected_real_positions": total_selected,
        "n_padding_positions_in_tensor": total_padding,
        "n_methyl_positive": int(np.sum(y_true == 1)),
        "n_methyl_negative": int(np.sum(y_true == 0)),
        "true_methyl_rate": methyl_rate,
        "base_recovery": base_recovery,
        "known_sequence_auc": auc_known,
        "end_to_end_auc": auc_e2e,
        "known_sequence_best_by_f1": best_row(known_rows, "f1"),
        "end_to_end_best_by_f1": best_row(e2e_rows, "f1"),
    }

    return summary, position_rows, meta_rows, known_rows + e2e_rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--data_jsonl", required=True)
    parser.add_argument("--mode", choices=["monomer", "complex"], default="monomer")
    parser.add_argument("--eval_chains", choices=["masked", "short", "all", "chain"], default="masked")
    parser.add_argument("--max_peptide_len", type=int, default=30)
    parser.add_argument("--chain_ids", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--thresholds", type=str, default="0.3,0.4,0.5,0.6,0.7,0.8,0.9,0.95,0.98,0.99")
    parser.add_argument("--out_dir", required=True)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    thresholds = [float(x) for x in args.thresholds.split(",") if x.strip()]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 100)
    print("干净 V28 评价")
    print("=" * 100)
    print(f"模型: {args.model_path}")
    print(f"数据: {args.data_jsonl}")
    print(f"设备: {device}")
    print(f"评价链: {args.eval_chains}")
    print(f"输出目录: {args.out_dir}")

    model = load_v28_model(args.model_path, device)
    dataset = JsonlDataset(args.data_jsonl)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)

    input_modes = [
        "raw_original_input",
        "strict_naturalized_input",
        "selected_positions_as_X",
    ]

    summaries = []
    all_positions = []
    all_metas = []
    all_thresholds = []

    for input_mode in input_modes:
        print(f"\n正在评价: {input_mode}")
        summary, position_rows, meta_rows, threshold_rows = run_input_mode(
            model, loader, device, args, input_mode, thresholds
        )
        summaries.append(summary)
        all_positions.extend(position_rows)
        all_metas.extend(meta_rows)
        all_thresholds.extend(threshold_rows)
        print(f"  真实评价位点: {summary['n_selected_positions']}")
        print(f"  基础氨基酸恢复率: {summary['base_recovery'] * 100:.2f}%")
        print(f"  甲基化正样本: {summary['n_methyl_positive']}")
        print(f"  已知序列最佳 F1: {summary['known_sequence_best_by_f1'].get('f1', 0) * 100:.2f}%")
        print(f"  端到端最佳 F1: {summary['end_to_end_best_by_f1'].get('f1', 0) * 100:.2f}%")

    write_json(os.path.join(args.out_dir, "summary.json"), summaries)
    write_csv(os.path.join(args.out_dir, "position_predictions.csv"), all_positions)
    write_csv(os.path.join(args.out_dir, "sample_manifest.csv"), all_metas)
    write_csv(os.path.join(args.out_dir, "threshold_metrics.csv"), all_thresholds)

    print("\n完成。输出文件：")
    print(os.path.join(args.out_dir, "summary.json"))
    print(os.path.join(args.out_dir, "position_predictions.csv"))
    print(os.path.join(args.out_dir, "sample_manifest.csv"))
    print(os.path.join(args.out_dir, "threshold_metrics.csv"))
    print("\n论文主口径建议优先看 strict_naturalized_input 下的 known_sequence_methylation。")


if __name__ == "__main__":
    main()

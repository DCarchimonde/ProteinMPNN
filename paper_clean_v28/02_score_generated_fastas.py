#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
02_score_generated_fastas.py

评价已经生成的 FASTA 序列。
不加载模型，只比较生成短肽序列和天然短肽序列。

输出：
1. all_designs.csv：所有设计序列。
2. unique_designs.csv：去重后的设计序列。
3. summary_by_target.csv：每个目标汇总。
4. summary_by_temperature.csv：每个温度汇总。
5. best_designs.csv：每个目标每个温度的最佳序列。

注意：
- 这个脚本只算序列层面的指标。
- 甲基化小写会先映射回天然氨基酸后再算基础氨基酸恢复率。
- 甲基化数量和比例会单独统计。
"""

import os
import re
import argparse
from typing import Dict, Any, List, Tuple, Optional

import numpy as np

from clean_v28_common import (
    read_jsonl,
    write_csv,
    write_json,
    parse_fasta,
    choose_eval_chains,
    get_record_name,
    naturalize_sequence,
    sequence_recovery,
    methyl_count,
)


def collect_native_targets(native_jsonl: str, eval_chains: str, max_peptide_len: int, chain_ids: Optional[str]):
    records = read_jsonl(native_jsonl)
    targets = {}
    manifest = []
    for i, r in enumerate(records):
        name = get_record_name(r, i)
        selected = choose_eval_chains(r, eval_chains, max_peptide_len, chain_ids)
        if not selected:
            continue
        seq = "".join(r.get(f"seq_chain_{c}", "") for c in selected)
        if not seq:
            continue
        targets[name.lower()] = {
            "target_name": name,
            "selected_chains": ",".join(selected),
            "native_seq": seq,
            "native_natural_seq": naturalize_sequence(seq),
            "native_length": len(seq),
            "native_methyl_count": methyl_count(seq),
        }
        manifest.append(targets[name.lower()])
    return targets, manifest


def infer_temperature_from_text(text: str) -> str:
    patterns = [
        r"T=([0-9.]+)",
        r"temp(?:erature)?[_=\- ]+([0-9.]+)",
        r"temperature[_=\- ]+([0-9.]+)",
        r"/([0-9]+(?:\.[0-9]+)?)/",
    ]
    for p in patterns:
        m = re.search(p, text, flags=re.IGNORECASE)
        if m:
            return m.group(1).rstrip(".")
    return "unknown"


def find_target_for_fasta(fasta_path: str, header: str, targets: Dict[str, Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    base = os.path.basename(fasta_path).lower()
    h = header.lower()

    # 优先最长名字匹配，避免短名字误匹配。
    for key in sorted(targets.keys(), key=len, reverse=True):
        if key in base or key in h:
            return targets[key]
    return None


def iter_fasta_files(fasta_dir: str):
    for root, _, files in os.walk(fasta_dir):
        for fn in files:
            if fn.lower().endswith((".fa", ".fasta", ".faa", ".txt")):
                yield os.path.join(root, fn)


def summarize_group(rows: List[Dict[str, Any]], group_keys: List[str]) -> List[Dict[str, Any]]:
    groups = {}
    for r in rows:
        key = tuple(r[k] for k in group_keys)
        groups.setdefault(key, []).append(r)

    out = []
    for key, items in sorted(groups.items()):
        recs = [float(x["natural_aa_recovery"]) for x in items if x["natural_aa_recovery"] != ""]
        methyl_rates = [float(x["design_methyl_rate"]) for x in items]
        unique_seqs = set(x["design_seq"] for x in items)
        row = {k: v for k, v in zip(group_keys, key)}
        row.update({
            "n_raw": len(items),
            "n_unique": len(unique_seqs),
            "n_duplicates": len(items) - len(unique_seqs),
            "unique_rate": len(unique_seqs) / len(items) if items else 0.0,
            "mean_recovery": float(np.mean(recs)) if recs else None,
            "best_recovery": float(np.max(recs)) if recs else None,
            "mean_methyl_rate": float(np.mean(methyl_rates)) if methyl_rates else 0.0,
        })
        out.append(row)
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--native_jsonl", required=True)
    parser.add_argument("--fasta_dir", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--eval_chains", choices=["masked", "short", "all", "chain"], default="short")
    parser.add_argument("--max_peptide_len", type=int, default=30)
    parser.add_argument("--chain_ids", type=str, default=None)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    targets, native_manifest = collect_native_targets(
        args.native_jsonl, args.eval_chains, args.max_peptide_len, args.chain_ids
    )
    write_csv(os.path.join(args.out_dir, "native_manifest.csv"), native_manifest)

    all_rows = []
    warnings = []

    for fasta_path in iter_fasta_files(args.fasta_dir):
        fasta_records = parse_fasta(fasta_path)
        temp_from_path = infer_temperature_from_text(fasta_path.replace("\\", "/"))
        for rec_idx, (header, seq) in enumerate(fasta_records):
            target = find_target_for_fasta(fasta_path, header, targets)
            if target is None:
                warnings.append({
                    "fasta_path": fasta_path,
                    "header": header,
                    "warning": "无法从文件名或 header 匹配 native target",
                })
                continue
            temp = infer_temperature_from_text(header)
            if temp == "unknown":
                temp = temp_from_path

            rec = sequence_recovery(target["native_seq"], seq, naturalize=True)
            length_match = len(seq) == target["native_length"]
            m_count = methyl_count(seq)

            all_rows.append({
                "target_name": target["target_name"],
                "selected_chains": target["selected_chains"],
                "temperature": temp,
                "fasta_file": fasta_path,
                "record_index": rec_idx,
                "header": header,
                "native_seq": target["native_seq"],
                "native_natural_seq": target["native_natural_seq"],
                "design_seq": seq,
                "design_natural_seq": naturalize_sequence(seq),
                "native_length": target["native_length"],
                "design_length": len(seq),
                "length_match": int(length_match),
                "natural_aa_recovery": rec if rec is not None else "",
                "design_methyl_count": m_count,
                "design_methyl_rate": m_count / len(seq) if len(seq) else 0.0,
            })

    # 去重：同一 target、temperature、design_seq 只保留一条。
    seen = set()
    unique_rows = []
    for r in all_rows:
        key = (r["target_name"], r["temperature"], r["design_seq"])
        if key not in seen:
            seen.add(key)
            unique_rows.append(r)

    summary_by_target = summarize_group(unique_rows, ["target_name"])
    summary_by_temperature = summarize_group(unique_rows, ["temperature"])
    summary_by_target_temperature = summarize_group(unique_rows, ["target_name", "temperature"])

    best_rows = []
    groups = {}
    for r in unique_rows:
        groups.setdefault((r["target_name"], r["temperature"]), []).append(r)
    for key, items in groups.items():
        valid_items = [x for x in items if x["natural_aa_recovery"] != ""]
        if not valid_items:
            continue
        best = max(valid_items, key=lambda x: float(x["natural_aa_recovery"]))
        best_rows.append(best)

    write_csv(os.path.join(args.out_dir, "all_designs.csv"), all_rows)
    write_csv(os.path.join(args.out_dir, "unique_designs.csv"), unique_rows)
    write_csv(os.path.join(args.out_dir, "summary_by_target.csv"), summary_by_target)
    write_csv(os.path.join(args.out_dir, "summary_by_temperature.csv"), summary_by_temperature)
    write_csv(os.path.join(args.out_dir, "summary_by_target_temperature.csv"), summary_by_target_temperature)
    write_csv(os.path.join(args.out_dir, "best_designs.csv"), best_rows)
    write_csv(os.path.join(args.out_dir, "warnings.csv"), warnings)

    report = {
        "n_native_targets": len(targets),
        "n_raw_designs": len(all_rows),
        "n_unique_designs": len(unique_rows),
        "n_warnings": len(warnings),
    }
    write_json(os.path.join(args.out_dir, "report.json"), report)

    print("完成 FASTA 干净评价。")
    print(report)
    print("输出目录:", args.out_dir)


if __name__ == "__main__":
    main()

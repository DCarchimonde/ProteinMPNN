#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
02_score_generated_fastas.py

评价已经生成的 FASTA 序列。
不加载模型，只比较生成短肽序列和天然短肽序列。

重要：
- 对于有两条相同短肽链的复合物，生成 FASTA 往往只含一条短肽序列。
- 因此推荐使用 --eval_chains auto_single。
- auto_single 会根据设计序列长度自动选择长度一致的短链；如果多条短链序列完全相同，选择第一条并记录候选链。
"""

import os
os.environ["OMP_NUM_THREADS"] = "1"

import re
import argparse
from typing import Dict, Any, List, Optional

import numpy as np

from clean_v28_common import (
    read_jsonl,
    write_csv,
    write_json,
    parse_fasta,
    choose_eval_chains,
    chain_ids_from_record,
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
        all_chain_ids = chain_ids_from_record(r)
        short_candidates = []
        for c in all_chain_ids:
            seq_c = r.get(f"seq_chain_{c}", "")
            if 0 < len(seq_c) <= max_peptide_len:
                short_candidates.append({
                    "chain_id": c,
                    "seq": seq_c,
                    "natural_seq": naturalize_sequence(seq_c),
                    "length": len(seq_c),
                    "methyl_count": methyl_count(seq_c),
                })

        if eval_chains == "auto_single":
            selected = []
            seq = ""
        else:
            selected = choose_eval_chains(r, eval_chains, max_peptide_len, chain_ids)
            seq = "".join(r.get(f"seq_chain_{c}", "") for c in selected)
            if not selected or not seq:
                continue

        target = {
            "target_name": name,
            "selected_chains": ",".join(selected),
            "native_seq": seq,
            "native_natural_seq": naturalize_sequence(seq),
            "native_length": len(seq),
            "native_methyl_count": methyl_count(seq),
            "short_candidates": short_candidates,
            "all_chain_ids": all_chain_ids,
        }
        targets[name.lower()] = target
        manifest.append({
            "target_name": name,
            "selected_chains_initial": ",".join(selected),
            "native_length_initial": len(seq),
            "short_candidate_chains": ";".join(f"{x['chain_id']}:{x['length']}:{x['seq']}" for x in short_candidates),
        })
    return targets, manifest


def resolve_target_for_design(target: Dict[str, Any], design_seq: str, eval_chains: str) -> Dict[str, Any]:
    """根据设计序列长度确定用于比较的天然链。"""
    if eval_chains != "auto_single":
        out = dict(target)
        out["chain_resolution_status"] = "fixed_by_user_mode"
        out["candidate_chains_same_length"] = ""
        return out

    design_len = len(design_seq)
    candidates = [x for x in target["short_candidates"] if x["length"] == design_len]
    if not candidates:
        out = dict(target)
        out.update({
            "selected_chains": "",
            "native_seq": "",
            "native_natural_seq": "",
            "native_length": 0,
            "native_methyl_count": 0,
            "chain_resolution_status": "no_short_chain_length_match",
            "candidate_chains_same_length": "",
        })
        return out

    unique_nat_seqs = sorted(set(x["natural_seq"] for x in candidates))
    chosen = candidates[0]
    status = "unique_length_match"
    if len(candidates) > 1 and len(unique_nat_seqs) == 1:
        status = "multiple_chains_same_sequence"
    elif len(candidates) > 1 and len(unique_nat_seqs) > 1:
        status = "ambiguous_multiple_different_sequences"

    out = dict(target)
    out.update({
        "selected_chains": chosen["chain_id"],
        "native_seq": chosen["seq"],
        "native_natural_seq": chosen["natural_seq"],
        "native_length": chosen["length"],
        "native_methyl_count": chosen["methyl_count"],
        "chain_resolution_status": status,
        "candidate_chains_same_length": ";".join(f"{x['chain_id']}:{x['seq']}" for x in candidates),
    })
    return out


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
    parser.add_argument("--eval_chains", choices=["masked", "short", "all", "chain", "auto_single"], default="auto_single")
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
            raw_target = find_target_for_fasta(fasta_path, header, targets)
            if raw_target is None:
                warnings.append({
                    "fasta_path": fasta_path,
                    "header": header,
                    "warning": "无法从文件名或 header 匹配 native target",
                })
                continue

            target = resolve_target_for_design(raw_target, seq, args.eval_chains)
            temp = infer_temperature_from_text(header)
            if temp == "unknown":
                temp = temp_from_path

            rec = sequence_recovery(target["native_seq"], seq, naturalize=True)
            length_match = len(seq) == target["native_length"]
            m_count = methyl_count(seq)

            if target.get("chain_resolution_status") in ["no_short_chain_length_match", "ambiguous_multiple_different_sequences"]:
                warnings.append({
                    "target_name": target["target_name"],
                    "fasta_path": fasta_path,
                    "header": header,
                    "design_length": len(seq),
                    "warning": target.get("chain_resolution_status", "unknown_chain_resolution_problem"),
                    "candidate_chains_same_length": target.get("candidate_chains_same_length", ""),
                })

            all_rows.append({
                "target_name": target["target_name"],
                "selected_chains": target["selected_chains"],
                "chain_resolution_status": target.get("chain_resolution_status", ""),
                "candidate_chains_same_length": target.get("candidate_chains_same_length", ""),
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

    seen = set()
    unique_rows = []
    for r in all_rows:
        key = (r["target_name"], r["selected_chains"], r["temperature"], r["design_seq"])
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
        "n_best_rows": len(best_rows),
        "n_warnings": len(warnings),
    }
    write_json(os.path.join(args.out_dir, "report.json"), report)

    print("完成 FASTA 干净评价。")
    print(report)
    print("输出目录:", args.out_dir)
    print("推荐论文序列评价优先使用 --eval_chains auto_single 的输出。")


if __name__ == "__main__":
    main()

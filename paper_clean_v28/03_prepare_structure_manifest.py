#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
03_prepare_structure_manifest.py

根据 auto_single 口径下的 best_designs.csv 和 native_jsonl 准备结构预测任务清单。

这个脚本不跑结构预测，只输出给师兄或后续工具使用的表格。

关键原则：
- 结构预测清单里的 selected_chains 直接采用 best_designs.csv 里已经解析好的链。
- 不再重新用 short 把两条短肽链拼起来。
- receptor_chains 是 native_jsonl 里除 selected_chains 外的所有链。
"""

import os
os.environ["OMP_NUM_THREADS"] = "1"

import csv
import argparse
from typing import Dict, Any, List

from clean_v28_common import (
    read_jsonl,
    write_csv,
    chain_ids_from_record,
    get_record_name,
    naturalize_sequence,
    methyl_count,
)


def read_csv(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def native_index(native_jsonl: str):
    idx = {}
    for i, r in enumerate(read_jsonl(native_jsonl)):
        name = get_record_name(r, i)
        all_chains = chain_ids_from_record(r)
        seqs = {c: r.get(f"seq_chain_{c}", "") for c in all_chains}
        idx[name] = {
            "target_name": name,
            "all_chains": all_chains,
            "seqs": seqs,
        }
    return idx


def safe_name(text: str) -> str:
    keep = []
    for ch in text:
        if ch.isalnum() or ch in "_-":
            keep.append(ch)
        else:
            keep.append("_")
    return "".join(keep).strip("_")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--best_csv", required=True)
    parser.add_argument("--native_jsonl", required=True)
    parser.add_argument("--out_csv", required=True)
    args = parser.parse_args()

    natives = native_index(args.native_jsonl)
    designs = read_csv(args.best_csv)

    rows = []
    warnings = []
    for i, d in enumerate(designs):
        target_name = d.get("target_name", "")
        if target_name not in natives:
            warnings.append({"target_name": target_name, "warning": "best_csv 中的目标在 native_jsonl 里找不到"})
            continue

        n = natives[target_name]
        selected = [x.strip() for x in d.get("selected_chains", "").split(",") if x.strip()]
        if not selected:
            warnings.append({"target_name": target_name, "warning": "best_csv 里 selected_chains 为空"})
            continue

        missing = [c for c in selected if c not in n["seqs"]]
        if missing:
            warnings.append({"target_name": target_name, "warning": f"selected_chains 不在 native_jsonl 中: {missing}"})
            continue

        receptor = [c for c in n["all_chains"] if c not in set(selected)]
        native_pep = "".join(n["seqs"][c] for c in selected)
        design_seq = d.get("design_seq", "")
        temp = d.get("temperature", "unknown")
        job_name = safe_name(f"{target_name}_T{temp}_best")

        if len(native_pep) != len(design_seq):
            warnings.append({
                "target_name": target_name,
                "temperature": temp,
                "warning": "native_peptide_seq 和 design_peptide_seq 长度不一致",
                "native_length": len(native_pep),
                "design_length": len(design_seq),
            })

        rows.append({
            "suggested_job_name": job_name,
            "target_name": target_name,
            "temperature": temp,
            "selected_chains": ",".join(selected),
            "receptor_chains": ",".join(receptor),
            "native_peptide_seq": native_pep,
            "native_peptide_natural_seq": naturalize_sequence(native_pep),
            "design_peptide_seq": design_seq,
            "design_peptide_natural_seq": naturalize_sequence(design_seq),
            "design_methyl_count": methyl_count(design_seq),
            "design_methyl_rate": methyl_count(design_seq) / len(design_seq) if design_seq else 0.0,
            "natural_aa_recovery": d.get("natural_aa_recovery", ""),
            "chain_resolution_status": d.get("chain_resolution_status", ""),
            "candidate_chains_same_length": d.get("candidate_chains_same_length", ""),
            "source_fasta_file": d.get("fasta_file", ""),
            "source_header": d.get("header", ""),
            "note": "结构预测时需要确认平台如何表示 N-甲基化残基；如果平台不支持小写甲基 token，需要单独记录修饰位点。",
        })

    os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)
    write_csv(args.out_csv, rows)
    warn_csv = os.path.join(os.path.dirname(args.out_csv), "structure_manifest_warnings.csv")
    write_csv(warn_csv, warnings)

    print("结构预测清单已生成:", args.out_csv)
    print("任务数:", len(rows))
    print("警告数:", len(warnings))
    print("警告文件:", warn_csv)


if __name__ == "__main__":
    main()

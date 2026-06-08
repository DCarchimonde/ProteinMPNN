#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
03_prepare_structure_manifest.py

根据 best_designs.csv 和 native_jsonl 准备结构预测任务清单。
这个脚本不跑结构预测，只输出给师兄或后续工具使用的表格。

输出字段包括：
- target_name
- selected_chains
- receptor_chains
- native_peptide_seq
- design_peptide_seq
- design_peptide_natural_seq
- design_methyl_count
- suggested_job_name

注意：
不同结构预测平台输入格式不同，所以这里先只做清单，不强行生成某一种格式。
"""

import os
import csv
import argparse
from typing import Dict, Any, List

from clean_v28_common import (
    read_jsonl,
    write_csv,
    choose_eval_chains,
    chain_ids_from_record,
    get_record_name,
    naturalize_sequence,
    methyl_count,
)


def read_csv(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def native_index(native_jsonl: str, eval_chains: str, max_peptide_len: int, chain_ids: str):
    idx = {}
    for i, r in enumerate(read_jsonl(native_jsonl)):
        name = get_record_name(r, i)
        selected = choose_eval_chains(r, eval_chains, max_peptide_len, chain_ids)
        all_chains = chain_ids_from_record(r)
        receptor = [c for c in all_chains if c not in set(selected)]
        native_pep = "".join(r.get(f"seq_chain_{c}", "") for c in selected)
        receptor_seqs = {c: r.get(f"seq_chain_{c}", "") for c in receptor}
        idx[name] = {
            "target_name": name,
            "selected_chains": ",".join(selected),
            "receptor_chains": ",".join(receptor),
            "native_peptide_seq": native_pep,
            "receptor_sequences": receptor_seqs,
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
    parser.add_argument("--eval_chains", choices=["masked", "short", "all", "chain"], default="short")
    parser.add_argument("--max_peptide_len", type=int, default=30)
    parser.add_argument("--chain_ids", type=str, default=None)
    args = parser.parse_args()

    natives = native_index(args.native_jsonl, args.eval_chains, args.max_peptide_len, args.chain_ids)
    designs = read_csv(args.best_csv)

    rows = []
    warnings = []
    for i, d in enumerate(designs):
        target_name = d.get("target_name", "")
        if target_name not in natives:
            warnings.append({"target_name": target_name, "warning": "best_csv 中的目标在 native_jsonl 里找不到"})
            continue
        n = natives[target_name]
        design_seq = d.get("design_seq", "")
        temp = d.get("temperature", "unknown")
        job_name = safe_name(f"{target_name}_T{temp}_rank{i+1}")
        rows.append({
            "suggested_job_name": job_name,
            "target_name": target_name,
            "temperature": temp,
            "selected_chains": n["selected_chains"],
            "receptor_chains": n["receptor_chains"],
            "native_peptide_seq": n["native_peptide_seq"],
            "design_peptide_seq": design_seq,
            "design_peptide_natural_seq": naturalize_sequence(design_seq),
            "design_methyl_count": methyl_count(design_seq),
            "design_methyl_rate": methyl_count(design_seq) / len(design_seq) if design_seq else 0.0,
            "natural_aa_recovery": d.get("natural_aa_recovery", ""),
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


if __name__ == "__main__":
    main()

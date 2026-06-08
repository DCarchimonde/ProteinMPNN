#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
04_audit_native_chains.py

检查 native_jsonl 里每个复合物的链、长度、masked_list、visible_list。
用途：
- 确认生成 FASTA 到底应该和哪一条短肽链比较。
- 避免把两条短链拼起来和单条设计序列比较。
"""

import os
import argparse

from clean_v28_common import read_jsonl, write_csv, chain_ids_from_record, get_record_name


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--native_jsonl", required=True)
    parser.add_argument("--out_csv", default="paper_clean_v28_outputs/native_chain_audit.csv")
    parser.add_argument("--max_peptide_len", type=int, default=30)
    args = parser.parse_args()

    rows = []
    records = read_jsonl(args.native_jsonl)
    for i, r in enumerate(records):
        name = get_record_name(r, i)
        masked = r.get("masked_list", [])
        visible = r.get("visible_list", [])
        chain_ids = chain_ids_from_record(r)
        short_chains = []
        for c in chain_ids:
            seq = r.get(f"seq_chain_{c}", "")
            if 0 < len(seq) <= args.max_peptide_len:
                short_chains.append(c)
            rows.append({
                "target_name": name,
                "chain_id": c,
                "chain_length": len(seq),
                "sequence": seq,
                "is_short_chain": int(0 < len(seq) <= args.max_peptide_len),
                "in_masked_list": int(c in masked),
                "in_visible_list": int(c in visible),
                "masked_list": ",".join(masked),
                "visible_list": ",".join(visible),
                "all_short_chains": ",".join(short_chains),
            })

    write_csv(args.out_csv, rows)
    print("链审计完成:", args.out_csv)
    print("请重点看每个 target 的短链、masked_list、FASTA 设计长度是否一致。")

    # 终端也打印一份简表，方便直接复制给 ChatGPT。
    current = None
    for r in rows:
        if r["target_name"] != current:
            current = r["target_name"]
            print("\n" + current)
            print("  masked_list:", r["masked_list"])
            print("  visible_list:", r["visible_list"])
            print("  all_short_chains:", r["all_short_chains"])
        print(f"  chain {r['chain_id']}: len={r['chain_length']}, short={r['is_short_chain']}, masked={r['in_masked_list']}, seq={r['sequence']}")


if __name__ == "__main__":
    main()

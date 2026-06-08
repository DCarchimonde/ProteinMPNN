import json
import os
import sys
import numpy as np
from Bio.SeqUtils.ProtParam import ProteinAnalysis

def get_pI(seq):
    # 简易 pI 计算
    clean = seq.upper().replace("X", "A")
    clean = "".join([c for c in clean if c in "ACDEFGHIKLMNPQRSTVWY"])
    if not clean: return 0.0
    return ProteinAnalysis(clean).isoelectric_point()

def main():
    gt_path = "nmethyl_data/test_sets/test.jsonl"
    design_path = "final_designs_v12.fasta"
    
    print(f"Checking alignment between:\n  GT: {gt_path}\n  Design: {design_path}\n")
    
    # 1. 加载 GT
    gt_dict = {}
    with open(gt_path, 'r') as f:
        for i, line in enumerate(f):
            entry = json.loads(line)
            # 原始名字
            raw_name = entry.get('name', f'target_{i}')
            # 清理后的名字 (用于匹配)
            safe_name = "".join([c for c in raw_name if c.isalnum() or c in ('_','-')])
            
            # 获取序列
            chains = entry.get('visible_list', []) + entry.get('masked_list', [])
            seq = "".join([entry.get(f'seq_chain_{c}', '') for c in chains])
            
            gt_dict[safe_name] = seq
            
            # 打印前3个加载的 GT 名字，用于调试
            if i < 3:
                print(f"Loaded GT [{i}]: Key='{safe_name}' Len={len(seq)} Seq={seq[:10]}...")

    print(f"\nTotal GT loaded: {len(gt_dict)}")
    
    # 2. 加载 Design 并对比
    print("\n--- Comparing First 5 Pairs ---")
    
    with open(design_path, 'r') as f:
        design_content = f.read().split('>')
    
    # 过滤空项
    design_entries = [e for e in design_content if e.strip()]
    
    matched_count = 0
    
    for i, entry in enumerate(design_entries):
        lines = entry.strip().split('\n')
        header = lines[0]
        design_seq = "".join(lines[1:])
        
        # 解析名字
        if "_design_" in header:
            base_name = header.split("_design_")[0]
        else:
            base_name = header
        safe_base = "".join([c for c in base_name if c.isalnum() or c in ('_','-')])
        
        if i < 5:
            print(f"\nPair {i+1}:")
            print(f"  Header: {header}")
            print(f"  Key:    '{safe_base}'")
            
            if safe_base in gt_dict:
                gt_seq = gt_dict[safe_base]
                print(f"  GT Seq: {gt_seq}")
                print(f"  De Seq: {design_seq}")
                print(f"  Match?: {'✅ FOUND' if safe_base in gt_dict else '❌ NOT FOUND'}")
                print(f"  pI: GT={get_pI(gt_seq):.2f} vs Design={get_pI(design_seq):.2f}")
                
                # 检查序列长度
                if len(gt_seq) != len(design_seq):
                    print(f"  ⚠️ LENGTH MISMATCH: {len(gt_seq)} vs {len(design_seq)}")
                
                matched_count += 1
            else:
                print(f"  ❌ GT NOT FOUND for key '{safe_base}'")
                # 打印 GT 里的所有 key 看看是不是名字处理不对
                if i == 0:
                    print(f"  (First 5 GT keys: {list(gt_dict.keys())[:5]})")

    print(f"\nTotal Matched in FASTA: {matched_count} / {len(design_entries)}")

if __name__ == "__main__":
    main()
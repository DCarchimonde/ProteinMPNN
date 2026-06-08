import json
import sys
import os
import numpy as np  # 新增这一行，导入numpy库并起别名np

def debug_data(jsonl_file):
    """详细调试数据格式问题"""
    print(f"Debugging data file: {jsonl_file}")
    
    with open(jsonl_file, 'r') as f:
        lines = f.readlines()
    
    print(f"Total samples: {len(lines)}")
    
    valid_samples = 0
    for i, line in enumerate(lines):
        try:
            data = json.loads(line)
            print(f"\n--- Sample {i+1} ---")
            
            # 检查关键字段
            print(f"Keys: {list(data.keys())}")
            print(f"masked_list: {data.get('masked_list', 'MISSING')}")
            print(f"visible_list: {data.get('visible_list', 'MISSING')}")
            print(f"num_of_chains: {data.get('num_of_chains', 'MISSING')}")
            
            # 检查序列数据
            seq_keys = [k for k in data.keys() if k.startswith('seq_chain_')]
            print(f"Sequence keys: {seq_keys}")
            
            for seq_key in seq_keys:
                seq = data[seq_key]
                print(f"  {seq_key}: '{seq}' (length: {len(seq)})")
                
                # 检查序列字符是否在扩展字母表中
                from nmethyl.utils.nmethyl_config import EXTENDED_AA_ALPHABET
                invalid_chars = [c for c in seq if c not in EXTENDED_AA_ALPHABET]
                if invalid_chars:
                    print(f"    WARNING: Invalid characters: {set(invalid_chars)}")
            
            # 检查坐标数据
            for chain_id in data.get('masked_list', []) + data.get('visible_list', []):
                print(f"  Chain {chain_id} coordinates:")
                for atom in ['N', 'CA', 'C', 'O']:
                    coord_key = f'{atom}_chain_{chain_id}'
                    if coord_key in data:
                        coords = data[coord_key]
                        print(f"    {coord_key}: {len(coords)} coordinates")
                        if coords:
                            # 检查是否有NaN值
                            nan_count = sum(1 for coord in coords if any(np.isnan(c) for c in coord))
                            if nan_count > 0:
                                print(f"      WARNING: {nan_count} NaN coordinates")
                    else:
                        print(f"    MISSING: {coord_key}")
            
            valid_samples += 1
            if i >= 406:  # 只检查前3个样本
                break
                
        except Exception as e:
            print(f"Error parsing sample {i+1}: {e}")
    
    print(f"\nSummary: {valid_samples} valid samples found")
    return valid_samples > 0

if __name__ == "__main__":
    if len(sys.argv) > 1:
        debug_data(sys.argv[1])
    else:
        debug_data("nmethyl_data/training_sets/train.jsonl")
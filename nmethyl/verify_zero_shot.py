import os
import sys

# --- [关键修复] 必须放在所有 import nmethyl... 之前！ ---
# 获取当前脚本所在目录的上一级目录 (即 ProteinMPNN-main 根目录)
current_dir = os.path.dirname(os.path.abspath(__file__)) # nmethyl/
project_root = os.path.dirname(current_dir)              # ProteinMPNN-main/
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import argparse
import torch
import torch.nn as nn
import json
import numpy as np
from torch.utils.data import DataLoader

# 现在系统路径已经包含了项目根目录，可以安全导入了
try:
    from nmethyl.train_final_v3 import DecoupledProteinMPNN, JSONLDataset, collate_fn, final_evaluation
    from nmethyl.utils.nmethyl_config import EXTENDED_AA_TO_INDEX
except ImportError as e:
    print(f"Import Error: {e}")
    print(f"Current sys.path: {sys.path}")
    sys.exit(1)

def verify_fix():
    # 1. 设置路径
    # 请确保文件名与您上传的一致，通常是 v_48_020.pt
    PRETRAINED_PATH = "vanilla_model_weights/v_48_020.pt" 
    TEST_DATA_PATH = "nmethyl_data/test_sets/test.jsonl"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if not os.path.exists(PRETRAINED_PATH):
        print(f"Error: Pretrained weights not found at {PRETRAINED_PATH}")
        return

    print(f"Testing Fix with Pretrained Weights: {PRETRAINED_PATH}")
    
    # 2. 初始化模型
    model = DecoupledProteinMPNN().to(device)
    
    # 3. 执行【智能权重加载】逻辑
    print("[1/3] Loading Checkpoint...")
    ckpt = torch.load(PRETRAINED_PATH, map_location=device)
    pretrained_dict = ckpt.get('model_state_dict', ckpt)
    
    # A. 加载骨架
    print("[2/3] Loading Backbone...")
    model_state = model.state_dict()
    filtered = {k: v for k, v in pretrained_dict.items() if k in model_state and v.size() == model_state[k].size()}
    model.load_state_dict(filtered, strict=False)
    
    # B. 智能映射 Embedding (核心修复验证)
    print("[3/3] Applying SMART MAPPING fix...")
    if 'W_s.weight' in pretrained_dict:
        STANDARD_ALPHABET = 'ACDEFGHIKLMNPQRSTVWYX'
        pretrained_ws = pretrained_dict['W_s.weight']
        with torch.no_grad():
            count = 0
            for i, char in enumerate(STANDARD_ALPHABET):
                if char in EXTENDED_AA_TO_INDEX:
                    target_idx = EXTENDED_AA_TO_INDEX[char]
                    model.W_s.weight.data[target_idx] = pretrained_ws[i].clone()
                    count += 1
            print(f"    ✅ Successfully mapped {count}/21 characters.")
            
            # 验证 X 的位置
            x_target = EXTENDED_AA_TO_INDEX.get('X', -1)
            x_origin = 20 # 在标准MPNN中 X 是第21个 (索引20)
            print(f"    ℹ️  'X' (Mask) origin index: {x_origin} -> target index: {x_target}")
            
            # 简单验证一下权重是否真的拷贝过去了
            # 比较原模型第20行和新模型第34行(假设X是34)是否相等
            is_equal = torch.allclose(pretrained_ws[x_origin], model.W_s.weight.data[x_target])
            print(f"    🔍 Weight Integrity Check (X): {'PASSED' if is_equal else 'FAILED'}")

    # 4. 立即评估 (Zero-Shot)
    print("\n>>> Running Zero-Shot Evaluation (No Training)...")
    print("Goal: Natural AA Recovery should be > 45%.\n")
    
    test_ds = JSONLDataset(TEST_DATA_PATH)
    test_loader = DataLoader(test_ds, batch_size=8, shuffle=False, collate_fn=collate_fn)
    
    final_evaluation(model, test_loader, device)

if __name__ == "__main__":
    verify_fix()
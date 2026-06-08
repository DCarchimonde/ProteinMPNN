import torch
import os
import sys
import json
import numpy as np
from torch.utils.data import DataLoader

# 引入您的配置
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try:
    from nmethyl.utils.nmethyl_config import EXTENDED_AA_ALPHABET
    from model_utils import ProteinMPNN
    from nmethyl.train_final_v3 import DecoupledProteinMPNN, JSONLDataset, collate_fn, featurize_batch, final_evaluation
except ImportError as e:
    print(f"Error importing: {e}")
    sys.exit(1)

# ProteinMPNN 的标准字母表 (不可更改的真理)
STANDARD_MPNN_ALPHABET = 'ACDEFGHIKLMNPQRSTVWYX'

def check_alphabet():
    print("\n=== 1. 字母表对齐诊断 ===")
    print(f"Standard MPNN Alphabet (len={len(STANDARD_MPNN_ALPHABET)}): {STANDARD_MPNN_ALPHABET}")
    print(f"Your Extended Alphabet (len={len(EXTENDED_AA_ALPHABET)}):   {EXTENDED_AA_ALPHABET}")
    
    # 检查前21位是否完全一致
    your_prefix = EXTENDED_AA_ALPHABET[:21]
    if your_prefix == STANDARD_MPNN_ALPHABET:
        print("✅ 完美匹配！前21个字母顺序完全一致。")
        return True
    else:
        print("❌ 严重错误！字母表顺序不一致。")
        print(f"Mismatch details:")
        for i, (s, y) in enumerate(zip(STANDARD_MPNN_ALPHABET, your_prefix)):
            if s != y:
                print(f"  Index {i}: Standard='{s}' vs Yours='{y}'")
        print("\n警告：这就是导致准确率暴跌的根本原因。直接复制权重会导致特征错位。")
        return False

def zero_shot_test(pretrained_path, test_data_path):
    print("\n=== 2. 零样本能力测试 (Zero-Shot) ===")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 初始化模型
    model = DecoupledProteinMPNN().to(device)
    
    # 加载权重 (使用之前的修复逻辑)
    print(f"Loading weights from {pretrained_path}...")
    ckpt = torch.load(pretrained_path, map_location=device)
    state_dict = ckpt.get('model_state_dict', ckpt)
    
    # 模拟 train_final_v3 的加载过程
    # 1. Load Backbone
    model_state = model.state_dict()
    filtered = {k: v for k, v in state_dict.items() if k in model_state and v.size() == model_state[k].size()}
    model.load_state_dict(filtered, strict=False)
    
    # 2. Load Embeddings (Blind Copy)
    if 'W_s.weight' in state_dict:
        model.W_s.weight.data[:21] = state_dict['W_s.weight'][:21].clone()
        
    # 3. Load Heads
    if 'W_out.weight' in state_dict:
        model.W_out_base.weight.data = state_dict['W_out.weight'][:20].clone()
        model.W_out_base.bias.data = state_dict['W_out.bias'][:20].clone()
        
    # 运行评估
    test_ds = JSONLDataset(test_data_path)
    test_loader = DataLoader(test_ds, batch_size=8, shuffle=False, collate_fn=collate_fn)
    
    print("Running evaluation BEFORE any training...")
    final_evaluation(model, test_loader, device)

if __name__ == "__main__":
    # 请替换为您实际的路径
    PRETRAINED_WEIGHTS = "vanilla_model_weights/v_48_020.pt"
    TEST_DATA = "nmethyl_data/test_sets/test.jsonl"
    
    aligned = check_alphabet()
    if aligned:
        print("Alphabet checks out. Running Zero-shot test to check weight integrity...")
        zero_shot_test(PRETRAINED_WEIGHTS, TEST_DATA)
    else:
        print("\n【解决方案】")
        print("我们需要构建一个 '重映射索引 (Remapping Index)'。")
        print("不能直接 copy [:21]，而是要根据字母把 standard 的权重搬运到 extended 对应的位置。")
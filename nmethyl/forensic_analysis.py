import torch
import os
import sys

# 路径设置
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try:
    from nmethyl.utils.nmethyl_config import NATURAL_AA_ALPHABET
    from nmethyl.train_final_v4 import DecoupledProteinMPNN
except ImportError as e:
    print(f"Import Error: {e}")
    sys.exit(1)

PRETRAINED_PATH = "vanilla_model_weights/v_48_020.pt"

def forensic_scan():
    print("=== 1. 输出层顺序检查 (Output Layer Order) ===")
    STANDARD_NATURAL = "ACDEFGHIKLMNPQRSTVWY"
    print(f"Standard (Pretrained): {STANDARD_NATURAL}")
    print(f"Yours (Config):        {NATURAL_AA_ALPHABET}")
    
    if NATURAL_AA_ALPHABET == STANDARD_NATURAL:
        print("✅ Output Alphabet MATCHES! (Output layer logic is safe)")
    else:
        print("❌ Output Alphabet MISMATCH! (Output layer is scrambled)")
        print("   This explains why predictions are wrong even if inputs are correct.")

    print("\n=== 2. 寻找丢失的特征层 (Missing Features Search) ===")
    if not os.path.exists(PRETRAINED_PATH):
        print("Weights file not found.")
        return

    device = torch.device('cpu')
    model = DecoupledProteinMPNN().to(device)
    ckpt = torch.load(PRETRAINED_PATH, map_location=device)
    file_keys = ckpt.get('model_state_dict', ckpt).keys()
    
    # 我们的模型期望的 features key
    target_key = "features.embeddings.weight"
    print(f"Model expects: '{target_key}'")
    
    # 在文件里搜索类似的东西
    candidates = [k for k in file_keys if "features" in k or "embedding" in k]
    print(f"\nFound {len(candidates)} candidate keys in file containing 'features' or 'embedding':")
    for k in candidates:
        print(f"  - {k}")
        
    print("\n>>> Diagnosis:")
    if not any("features.embeddings" in k for k in candidates):
        print("   The pretrained file truly lacks this layer.")
        print("   It likely uses FIXED sinusoidal embeddings, while your code expects LEARNABLE ones.")
        print("   This mismatch forces random positional encoding -> Low Accuracy.")

if __name__ == "__main__":
    forensic_scan()
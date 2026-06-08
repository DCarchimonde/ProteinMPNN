import torch
import os
import sys

# 路径设置
PRETRAINED_PATH = "vanilla_model_weights/v_48_020.pt"

def inspect_checkpoint():
    if not os.path.exists(PRETRAINED_PATH):
        print("Checkpoint not found!")
        return

    print(f"Inspecting: {PRETRAINED_PATH}")
    ckpt = torch.load(PRETRAINED_PATH, map_location='cpu')
    state_dict = ckpt.get('model_state_dict', ckpt)

    print(f"\nTotal keys in checkpoint: {len(state_dict)}")
    print("Sample keys from checkpoint (First 5):")
    for i, k in enumerate(list(state_dict.keys())[:5]):
        print(f"  [{i}] {k}  \tShape: {state_dict[k].shape}")
        
    # 模拟我们的模型
    sys.path.append(os.path.dirname(os.path.abspath(__file__))) # 确保能导入
    from nmethyl.train_final_v3 import DecoupledProteinMPNN
    model = DecoupledProteinMPNN()
    model_keys = list(model.state_dict().keys())
    
    print("\nSample keys from OUR Model (First 5):")
    for i, k in enumerate(model_keys[:5]):
        print(f"  [{i}] {k}")
        
    # 检查匹配情况
    common = set(state_dict.keys()) & set(model_keys)
    print(f"\n>>> Intersection: {len(common)} keys match perfectly.")
    
    if len(common) < 10:
        print("⚠️  CRITICAL FAIL: Almost no keys match! Prefix mismatch likely.")
    else:
        print("✅ Keys look matching. Problem might be dimensions.")

if __name__ == "__main__":
    inspect_checkpoint()
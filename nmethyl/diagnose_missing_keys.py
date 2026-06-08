import torch
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from nmethyl.train_final_v4 import DecoupledProteinMPNN

PRETRAINED_PATH = "vanilla_model_weights/v_48_020.pt"

def diagnose():
    if not os.path.exists(PRETRAINED_PATH):
        print("File not found.")
        return

    print(f"Diagnosing: {PRETRAINED_PATH}")
    device = torch.device('cpu')
    model = DecoupledProteinMPNN().to(device)
    
    # 1. 读取文件里的 Keys
    ckpt = torch.load(PRETRAINED_PATH, map_location=device)
    file_state = ckpt.get('model_state_dict', ckpt)
    # 去掉 module. 前缀
    file_keys_clean = {k.replace("module.", ""): v.shape for k, v in file_state.items()}
    
    # 2. 读取模型里的 Keys
    model_state = model.state_dict()
    model_keys = {k: v.shape for k, v in model_state.items()}
    
    # 3. 对比
    print("\n=== 关键层检查 (Critical Layers) ===")
    critical_layers = ['W_e.weight', 'W_v.weight', 'features.embeddings.weight']
    for layer in critical_layers:
        in_file = layer in file_keys_clean
        in_model = layer in model_keys
        
        if in_file and in_model:
            shape_file = file_keys_clean[layer]
            shape_model = model_keys[layer]
            match = (shape_file == shape_model)
            status = "✅ MATCH" if match else f"❌ SHAPE MISMATCH ({shape_file} vs {shape_model})"
        elif not in_file:
            status = "❌ MISSING IN FILE"
        else:
            status = "❌ MISSING IN MODEL"
            
        print(f"{layer:<30} : {status}")

    print("\n=== 未加载的参数 (Missing in Model) ===")
    # 也就是说，模型里有，但文件里没给（或者形状不对被跳过）
    missing_count = 0
    for k, v_shape in model_keys.items():
        if k not in file_keys_clean:
            # W_s 和 W_out 也就是我们改过的，不匹配是正常的
            if 'W_s' not in k and 'W_out' not in k:
                print(f"  [MISSING] {k:<40} shape={v_shape}")
                missing_count += 1
        elif file_keys_clean[k] != v_shape:
            print(f"  [MISMATCH] {k:<40} file={file_keys_clean[k]} vs model={v_shape}")
            missing_count += 1
            
    if missing_count == 0:
        print("✅ All backbone keys match perfectly!")
    else:
        print(f"\n⚠️ Found {missing_count} missing/mismatched backbone keys.")
        print("   (Excluding W_s and W_out which are handled manually)")

if __name__ == "__main__":
    diagnose()
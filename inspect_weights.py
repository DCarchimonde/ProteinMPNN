import torch
import argparse
import sys

def inspect(path):
    print(f"🔍 Inspecting: {path}")
    try:
        ckpt = torch.load(path, map_location='cpu')
    except Exception as e:
        print(f"❌ Load failed: {e}")
        return

    if 'model_state_dict' in ckpt:
        sd = ckpt['model_state_dict']
    else:
        sd = ckpt

    print(f"✅ Loaded state_dict with {len(sd)} keys.")
    
    # 1. 检查前缀
    keys = list(sd.keys())
    has_module = any(k.startswith('module.') for k in keys)
    print(f"   - Prefix 'module.': {has_module}")
    
    # 2. 关键层维度探测
    # 我们需要找 Embedding, Encoder, Decoder 的维度
    targets = [
        'W_s.weight',             # [Vocab, Hidden]
        'W_e.weight',             # [Hidden, EdgeIn]
        'encoder_layers.0.W1.weight', # Encoder Input
        'decoder_layers.0.W1.weight', # Decoder Input (关键冲突点!)
        'W_out.weight'            # [Vocab, Hidden]
    ]
    
    found_any = False
    for k in keys:
        # 去掉前缀再匹配
        clean_k = k.replace('module.', '')
        if clean_k in targets:
            shape = sd[k].shape
            print(f"   - Layer '{clean_k}': {list(shape)}")
            found_any = True
            
    if not found_any:
        print("❌ 警告：未找到标准层名称。打印前10个key供参考：")
        for k in keys[:10]:
            print(f"     {k}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, required=True, help="Path to .pt file")
    args = parser.parse_args()
    inspect(args.path)
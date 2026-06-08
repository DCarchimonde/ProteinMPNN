import torch
import argparse

def merge_weights(original_path, finetuned_path, output_path):
    print(f"📦 正在加载原始躯干: {original_path}")
    orig_ckpt = torch.load(original_path, map_location='cpu')
    orig_state = orig_ckpt.get('model_state_dict', orig_ckpt)
    
    print(f"🧠 正在加载聪明的专家头: {finetuned_path}")
    finetuned_ckpt = torch.load(finetuned_path, map_location='cpu')
    finetuned_state = finetuned_ckpt.get('model_state_dict', finetuned_ckpt)
    
    # 手术开始：遍历微调后的权重，只要是 'experts' 开头的，就强制覆盖到原始权重里
    merged_state = orig_state.copy()
    expert_count = 0
    
    for key, value in finetuned_state.items():
        # 兼容一下加了 module. 的情况
        clean_key = key.replace("module.", "")
        if "experts" in clean_key:
            # 找到对应的原始 key
            target_key = clean_key
            if target_key not in merged_state and "module." + target_key in merged_state:
                target_key = "module." + target_key
                
            merged_state[target_key] = value.clone()
            expert_count += 1
            
    print(f"💉 成功移植了 {expert_count} 个专家头参数矩阵！")
    
    # 把缝合好的新权重保存下来
    torch.save({'model_state_dict': merged_state}, output_path)
    print(f"🎉 终极神级模型已保存至: {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--original", type=str, default="./run_v28_robust/best_model_v28_0.1269.pt")
    parser.add_argument("--finetuned", type=str, default="./best_finetuned_experts.pt")
    parser.add_argument("--output", type=str, default="./frankenstein_v28.pt")
    args = parser.parse_args()
    
    merge_weights(args.original, args.finetuned, args.output)
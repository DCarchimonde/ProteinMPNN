import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import argparse
import json
import sys
import os
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import classification_report, confusion_matrix

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 复用 train_surgery.py 中的模型定义，确保一致性
try:
    from nmethyl.train_surgery import SurgeryWrapper, ProteinMPNN, compute_hbond_mask, featurize_batch_standalone, JSONLDataset, NMETHYL_TO_NATURAL_MAPPING, EXTENDED_AA_TO_INDEX
except ImportError:
    # 如果导入失败，说明路径有问题，这里做个简单的错误提示
    print("❌ Error: 无法导入 nmethyl.train_surgery。请确保文件名正确且在同一目录下。")
    sys.exit(1)

def diagnose(model, loader, device):
    model.eval()
    print(">>> 正在进行深度诊断 (Deep Diagnosis)...")
    
    # 统计器
    correct_base = 0
    total_base = 0
    
    y_true_methyl = [] # 0/1
    y_pred_methyl_prob = [] # 概率
    
    with torch.no_grad():
        for batch in loader:
            inputs = featurize_batch_standalone(batch, device)
            X, S, mask = inputs[0], inputs[1], inputs[2]
            
            # Forward
            logits_base, logits_methyl = model(*inputs)
            
            # --- 1. 诊断基础氨基酸 (Base Accuracy) ---
            # 即使它是甲基化AA，我们也只看它的“底子”是不是猜对了
            # 比如 True=Me-Ala, Pred=Ala -> 算对！
            pred_base = torch.argmax(logits_base, dim=-1) # [B, L]
            
            # 把真实标签 S 映射回 Base (0-19)
            target_base = S.clone()
            for m_id, n_id in NMETHYL_TO_NATURAL_MAPPING.items():
                target_base[target_base == (m_id + 20)] = n_id
            
            # 只看有效区域
            valid_mask = mask.bool() & (S != EXTENDED_AA_TO_INDEX.get('X', -1))
            
            correct_base += (pred_base[valid_mask] == target_base[valid_mask]).sum().item()
            total_base += valid_mask.sum().item()
            
            # --- 2. 诊断甲基化二分类 (Methyl Binary Metrics) ---
            # 真实标签：是否甲基化
            target_is_methyl = (S >= 20).long()
            
            # 预测概率
            probs_methyl = F.softmax(logits_methyl, dim=-1)[:, :, 1]
            
            # 应用氢键规则
            hb_mask = compute_hbond_mask(X, mask)
            probs_methyl_ruled = probs_methyl * (1.0 - hb_mask)
            
            y_true_methyl.extend(target_is_methyl[valid_mask].cpu().numpy())
            y_pred_methyl_prob.extend(probs_methyl_ruled[valid_mask].cpu().numpy())

    # --- 生成报告 ---
    print("\n" + "="*60)
    print("🏥 模型体检报告 (MODEL HEALTH REPORT)")
    print("="*60)
    
    # Report 1: Base Accuracy
    base_acc = correct_base / total_base if total_base > 0 else 0
    print(f"1️⃣ 基础氨基酸恢复率 (Base AA Recovery): {base_acc:.2%} (理想值: >40%)")
    if base_acc < 0.4:
        print("   ⚠️ 警告: 基础能力丢失！可能是权重没加载好，或者 Embedding 映射错了。")
    else:
        print("   ✅ 正常: 模型保留了 ProteinMPNN 的基本功。")
        
    # Report 2: Methylation Detection
    y_true_methyl = np.array(y_true_methyl)
    y_pred_methyl = (np.array(y_pred_methyl_prob) > 0.5).astype(int)
    
    print("-" * 60)
    print("2️⃣ 甲基化探测能力 (Methylation Detection):")
    print(classification_report(y_true_methyl, y_pred_methyl, target_names=["Non-Methyl", "Methylified"], digits=4))
    
    tn, fp, fn, tp = confusion_matrix(y_true_methyl, y_pred_methyl).ravel()
    print(f"   - 抓到的甲基化点 (TP): {tp}")
    print(f"   - 漏掉的甲基化点 (FN): {fn}")
    print(f"   - 误判的甲基化点 (FP): {fp}")
    
    print("="*60)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_data", type=str, required=True)
    # 默认去找 train_surgery 生成的那个模型
    parser.add_argument("--model_path", type=str, default="final_hybrid_model.pt") 
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 初始化空壳
    base_model = ProteinMPNN(node_features=128, edge_features=128, hidden_dim=128)
    model = SurgeryWrapper(base_model).to(device)
    
    print(f"Loading: {args.model_path}")
    if not os.path.exists(args.model_path):
        print(f"❌ Error: 找不到模型文件 {args.model_path}。请先运行 train_surgery.py！")
        return

    # 加载刚才微调过的权重
    state = torch.load(args.model_path, map_location=device)
    model.load_state_dict(state, strict=False) 
    
    ds = JSONLDataset(args.test_data)
    loader = DataLoader(ds, batch_size=8, collate_fn=lambda x:x)
    
    diagnose(model, loader, device)

if __name__ == "__main__":
    main()
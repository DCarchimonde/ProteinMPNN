import argparse
import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import pandas as pd

# --- 系统路径 ---
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    # 引用 v12 的组件
    from nmethyl.train_final_v12_polishing import DecoupledProteinMPNN, featurize_batch, JSONLDataset, collate_fn
    from nmethyl.utils.nmethyl_config import EXTENDED_AA_ALPHABET, NATURAL_AA_ALPHABET, NMETHYL_TO_NATURAL_MAPPING, EXTENDED_AA_TO_INDEX
except ImportError as e:
    print(f"Error: {e}")
    sys.exit(1)

def analyze():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default="./final_run_v12_soup/best_model_soup.pt")
    parser.add_argument("--test_data", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="./analysis_results")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. 加载模型
    print(f"Loading v12 model from {args.checkpoint}...")
    model = DecoupledProteinMPNN(augment_eps=0.0).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device)['model_state_dict'])
    model.eval()

    # 2. 推理
    print("Running inference on test set...")
    test_ds = JSONLDataset(args.test_data)
    test_loader = torch.utils.data.DataLoader(test_ds, batch_size=16, shuffle=False, collate_fn=collate_fn)

    all_preds = []
    all_targets = []

    with torch.no_grad():
        for batch in test_loader:
            f = featurize_batch(batch, device)
            if f is None: continue
            
            # [关键修复] v12 只返回 6 个元素，直接解包即可
            # 顺序: X, S, mask, chain_M, residue_idx, chain_encoding_all
            X, S, mask, chain_M, residue_idx, chain_encoding_all = f

            logits_base, logits_methyl = model(X, S, mask, chain_M, residue_idx, chain_encoding_all)
            
            # 解码
            base = torch.argmax(logits_base, -1)
            is_me = torch.argmax(logits_methyl, -1)
            final = base.clone()
            
            # N-Me 组合逻辑
            nat_to_me_abs = {v: k + len(NATURAL_AA_ALPHABET) for k, v in NMETHYL_TO_NATURAL_MAPPING.items()}
            for n_idx, m_idx in nat_to_me_abs.items():
                mask_update = (is_me == 1) & (base == n_idx)
                final[mask_update] = m_idx
            
            # 收集有效数据 (过滤 'X')
            tgt_np = S.cpu().numpy().flatten()
            pred_np = final.cpu().numpy().flatten()
            mask_np = mask.cpu().numpy().flatten().astype(bool)
            
            x_idx = EXTENDED_AA_TO_INDEX.get('X', -1)
            valid = mask_np & (tgt_np != x_idx)
            
            all_preds.extend(pred_np[valid])
            all_targets.extend(tgt_np[valid])

    # 3. 绘制混淆矩阵 (Confusion Matrix)
    print("Generating Confusion Matrix...")
    # 获取数据中实际出现过的标签
    present_labels = sorted(list(set(all_targets) | set(all_preds)))
    
    # 过滤掉非法索引 (防止画图报错)
    present_labels = [i for i in present_labels if i < len(EXTENDED_AA_ALPHABET)]
    
    cm = confusion_matrix(all_targets, all_preds, labels=present_labels, normalize='true') # 按行归一化(Recall)
    
    # 获取标签名称
    label_names = [EXTENDED_AA_ALPHABET[i] for i in present_labels]
    
    plt.figure(figsize=(20, 18))
    sns.heatmap(cm, annot=False, cmap='Blues', xticklabels=label_names, yticklabels=label_names)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Normalized Confusion Matrix (Recall)")
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, "confusion_matrix.png"), dpi=300)
    print(f"Saved: {os.path.join(args.output_dir, 'confusion_matrix.png')}")

    # 4. 生成详细报告 CSV
    print("Generating Classification Report...")
    report_dict = classification_report(all_targets, all_preds, labels=present_labels, target_names=label_names, output_dict=True, zero_division=0)
    df = pd.DataFrame(report_dict).transpose()
    df.to_csv(os.path.join(args.output_dir, "classification_report.csv"))
    print(f"Saved: {os.path.join(args.output_dir, 'classification_report.csv')}")

    # 5. 重点关注 N-甲基化 的统计
    print("\n=== N-Methylation Specific Performance ===")
    # 找到 N-甲基化在 EXTENDED_AA_ALPHABET 中的位置 (20 到 len-2, 因为最后是X)
    n_me_chars = [char for i, char in enumerate(EXTENDED_AA_ALPHABET) if i >= 20 and char != 'X']
    
    # 筛选出 N-Me 的行 (确保它们在报告里)
    n_me_df = df.loc[df.index.isin(n_me_chars)]
    if not n_me_df.empty:
        print(n_me_df[['precision', 'recall', 'f1-score', 'support']])
        avg_f1 = np.average(n_me_df['f1-score'], weights=n_me_df['support'])
        print(f"\nWeighted F1 for N-Methyl Residues: {avg_f1:.4f}")
    else:
        print("No N-methylated residues found in test set predictions/targets.")

if __name__ == "__main__":
    analyze()
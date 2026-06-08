import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from Bio.SeqUtils.ProtParam import ProteinAnalysis
from scipy.stats import pearsonr

def get_properties(seq):
    """计算序列的 pI 和 疏水性"""
    # 转大写，且去除非法字符
    clean_seq = seq.upper().replace("X", "A") # 把X当丙氨酸处理，影响最小
    # 再次清洗，只保留标准氨基酸，防止报错
    clean_seq = "".join([c for c in clean_seq if c in "ACDEFGHIKLMNPQRSTVWY"])
    
    if len(clean_seq) == 0: return None, None
    
    try:
        X = ProteinAnalysis(clean_seq)
        pi = X.isoelectric_point()
        gravy = X.gravy() # Grand Average of Hydropathy
        return pi, gravy
    except:
        return None, None

def main():
    # 1. 读取 Ground Truth (原始序列)
    gt_path = "nmethyl_data/test_sets/test.jsonl"
    design_path = "final_designs_v12.fasta"
    output_dir = "paper_figures"
    os.makedirs(output_dir, exist_ok=True)
    
    print(">>> analyzing physicochemical properties...")
    
    # 构建 GT 字典: name -> sequence
    gt_seqs = {}
    with open(gt_path, 'r') as f:
        for i, line in enumerate(f):
            entry = json.loads(line)
            # 拼接所有链的序列作为该蛋白的总序列
            chains = entry.get('visible_list', []) + entry.get('masked_list', [])
            full_seq = "".join([entry.get(f'seq_chain_{c}', '') for c in chains])
            
            name = entry.get('name', f'target_{i}')
            # 清理名字以便匹配
            safe_name = "".join([c for c in name if c.isalnum() or c in ('_','-')])
            gt_seqs[safe_name] = full_seq

    # 2. 读取 Designed Sequences
    data = []
    
    current_name = None
    with open(design_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith(">"):
                # Header: >Me_123..._design_0
                header = line[1:]
                if "_design_" in header:
                    base_name = header.split("_design_")[0]
                else:
                    base_name = header
                
                # 再次清理 base_name 以匹配 GT
                safe_base = "".join([c for c in base_name if c.isalnum() or c in ('_','-')])
                current_name = safe_base
            else:
                design_seq = line
                if current_name in gt_seqs:
                    gt_seq = gt_seqs[current_name]
                    
                    # 计算性质
                    pi_gt, gravy_gt = get_properties(gt_seq)
                    pi_pred, gravy_pred = get_properties(design_seq)
                    
                    if pi_gt is not None and pi_pred is not None:
                        data.append({
                            "Target": current_name,
                            "pI_Native": pi_gt,
                            "pI_Design": pi_pred,
                            "Hydro_Native": gravy_gt,
                            "Hydro_Design": gravy_pred
                        })

    df = pd.DataFrame(data)
    print(f"Computed properties for {len(df)} pairs.")
    
    # 3. 绘图 1: 等电点 (pI)
    plt.figure(figsize=(8, 8))
    sns.scatterplot(data=df, x="pI_Native", y="pI_Design", alpha=0.6, color="blue")
    # 画对角线
    plt.plot([0, 14], [0, 14], 'r--', lw=2)
    
    # 计算相关系数
    corr, _ = pearsonr(df["pI_Native"], df["pI_Design"])
    plt.title(f"Isoelectric Point (pI) Correlation\nPearson r = {corr:.3f}", fontsize=14)
    plt.xlabel("Native Sequence pI", fontsize=12)
    plt.ylabel("Designed Sequence pI", fontsize=12)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.savefig(os.path.join(output_dir, "correlation_pI.png"), dpi=300)
    print(f"Saved pI plot: r = {corr:.3f}")

    # 4. 绘图 2: 疏水性 (GRAVY)
    plt.figure(figsize=(8, 8))
    sns.scatterplot(data=df, x="Hydro_Native", y="Hydro_Design", alpha=0.6, color="green")
    
    # 动态调整坐标轴范围
    min_val = min(df["Hydro_Native"].min(), df["Hydro_Design"].min()) - 0.5
    max_val = max(df["Hydro_Native"].max(), df["Hydro_Design"].max()) + 0.5
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2)
    
    corr_h, _ = pearsonr(df["Hydro_Native"], df["Hydro_Design"])
    plt.title(f"Hydrophobicity (GRAVY) Correlation\nPearson r = {corr_h:.3f}", fontsize=14)
    plt.xlabel("Native Sequence Hydrophobicity", fontsize=12)
    plt.ylabel("Designed Sequence Hydrophobicity", fontsize=12)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.savefig(os.path.join(output_dir, "correlation_hydro.png"), dpi=300)
    print(f"Saved Hydrophobicity plot: r = {corr_h:.3f}")

if __name__ == "__main__":
    main()
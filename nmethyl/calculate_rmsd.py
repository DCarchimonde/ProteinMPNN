import argparse
import os
import numpy as np
from Bio.PDB import PDBParser, Superimposer

def calculate_rmsd(ref_pdb, pred_pdb):
    """计算两个PDB文件基于CA原子的RMSD"""
    parser = PDBParser(QUIET=True)
    try:
        # 读取结构
        structure_ref = parser.get_structure("ref", ref_pdb)
        structure_pred = parser.get_structure("pred", pred_pdb)
        
        # 获取 CA 原子 (Alpha Carbon)
        # 注意：这里假设两个PDB的残基数量和顺序是一致的
        atoms_ref = []
        atoms_pred = []
        
        # 提取第一个模型
        model_ref = structure_ref[0]
        model_pred = structure_pred[0]
        
        # 遍历残基
        residues_ref = list(model_ref.get_residues())
        residues_pred = list(model_pred.get_residues())
        
        if len(residues_ref) != len(residues_pred):
            print(f"Warning: Length mismatch {len(residues_ref)} vs {len(residues_pred)}")
            return None

        for r1, r2 in zip(residues_ref, residues_pred):
            if 'CA' in r1 and 'CA' in r2:
                atoms_ref.append(r1['CA'])
                atoms_pred.append(r2['CA'])
        
        if len(atoms_ref) == 0:
            return None

        # 叠合 (Superimpose)
        sup = Superimposer()
        sup.set_atoms(atoms_ref, atoms_pred)
        sup.apply(model_pred.get_atoms())
        
        # 获取 RMSD
        return sup.rms
        
    except Exception as e:
        print(f"Error calculating RMSD: {e}")
        return None

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ref_dir", type=str, required=True, help="Directory containing original PDBs")
    parser.add_argument("--pred_dir", type=str, required=True, help="Directory containing AlphaFold predicted PDBs")
    args = parser.parse_args()
    
    rmsds = []
    
    print(f"{'PDB_ID':<20} | {'RMSD (Å)':<10}")
    print("-" * 35)
    
    # 遍历预测目录
    for pred_file in os.listdir(args.pred_dir):
        if not pred_file.endswith(".pdb"): continue
        
        # 假设文件名格式: target_name_design_0.pdb
        # 需要解析出对应的原始 PDB 文件名
        # 您可能需要根据您的实际文件名调整这里的匹配逻辑
        # 例如: pred="1abc_design_0.pdb" -> ref="1abc.pdb"
        base_name = pred_file.split("_design")[0] 
        ref_file = f"{base_name}.pdb" 
        ref_path = os.path.join(args.ref_dir, ref_file)
        
        if not os.path.exists(ref_path):
            # 尝试另一种命名可能
            ref_file = f"{base_name}_truth.pdb"
            ref_path = os.path.join(args.ref_dir, ref_file)
            
        if os.path.exists(ref_path):
            rmsd = calculate_rmsd(ref_path, os.path.join(args.pred_dir, pred_file))
            if rmsd is not None:
                rmsds.append(rmsd)
                print(f"{base_name:<20} | {rmsd:.4f}")
        else:
            # print(f"Reference not found for {pred_file}")
            pass
            
    if len(rmsds) > 0:
        print("-" * 35)
        print(f"Average scRMSD: {np.mean(rmsds):.4f} Å")
        print(f"Median scRMSD:  {np.median(rmsds):.4f} Å")
        print(f"Success Rate (<2.0Å): {np.mean(np.array(rmsds) < 2.0) * 100:.1f}%")
        
        # 生成论文数据：保存 RMSD 列表
        np.savetxt("scRMSD_results.csv", rmsds, delimiter=",")
        print("Results saved to scRMSD_results.csv")

if __name__ == "__main__":
    main()
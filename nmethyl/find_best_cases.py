import os
import shutil
import numpy as np
from Bio.PDB import PDBParser, Superimposer

def calculate_rmsd(ref_pdb, pred_pdb):
    parser = PDBParser(QUIET=True)
    try:
        s_ref = parser.get_structure("ref", ref_pdb)
        s_pred = parser.get_structure("pred", pred_pdb)
        atoms_ref = [a for a in s_ref.get_atoms() if a.name == 'CA']
        atoms_pred = [a for a in s_pred.get_atoms() if a.name == 'CA']
        min_len = min(len(atoms_ref), len(atoms_pred))
        if min_len < 3: return 999.9
        sup = Superimposer()
        sup.set_atoms(atoms_ref[:min_len], atoms_pred[:min_len])
        return sup.rms
    except:
        return 999.9

def main():
    pred_dir = "esmfold_predictions"
    gt_dir = "ground_truth_pdbs"
    output_dir = "best_structures_for_paper"
    
    os.makedirs(output_dir, exist_ok=True)
    
    results = []
    
    print(f">>> Scanning for best cases in {pred_dir}...")
    
    for pred_file in os.listdir(pred_dir):
        if not pred_file.endswith(".pdb"): continue
        
        # 文件名匹配逻辑
        # 假设 pred: TargetName_design_0.pdb
        if "_design_" in pred_file:
            base_name = pred_file.split("_design_")[0]
        else:
            base_name = pred_file.replace(".pdb", "")
            
        # 清理非法字符匹配 GT
        safe_base_name = "".join([c for c in base_name if c.isalnum() or c in ('_','-')])
        gt_file = os.path.join(gt_dir, f"{safe_base_name}.pdb")
        
        if os.path.exists(gt_file):
            rmsd = calculate_rmsd(gt_file, os.path.join(pred_dir, pred_file))
            if rmsd < 100: # 过滤失败的
                results.append((base_name, pred_file, rmsd))
    
    # 按 RMSD 从小到大排序
    results.sort(key=lambda x: x[2])
    
    print("\n" + "="*40)
    print("🏆 TOP 10 BEST FOLDING CASES")
    print("="*40)
    print(f"{'Rank':<5} | {'PDB ID':<30} | {'RMSD (Å)':<10}")
    
    # 保存前 10 个最好的
    for i, (base_name, pred_filename, rmsd) in enumerate(results[:10]):
        print(f"{i+1:<5} | {pred_filename:<30} | {rmsd:.4f}")
        
        # 复制文件到输出目录
        # 1. 复制 Ground Truth
        safe_base = "".join([c for c in base_name if c.isalnum() or c in ('_','-')])
        src_gt = os.path.join(gt_dir, f"{safe_base}.pdb")
        dst_gt = os.path.join(output_dir, f"Rank{i+1}_{base_name}_GT.pdb")
        shutil.copy(src_gt, dst_gt)
        
        # 2. 复制 Prediction
        src_pred = os.path.join(pred_dir, pred_filename)
        dst_pred = os.path.join(output_dir, f"Rank{i+1}_{base_name}_Pred_RMSD{rmsd:.2f}.pdb")
        shutil.copy(src_pred, dst_pred)
        
    print(f"\n✅ Best PDB files have been copied to: {output_dir}")
    print("Please download this folder and use PyMOL to visualize them!")

if __name__ == "__main__":
    main()
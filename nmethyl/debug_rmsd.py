import os
import sys
from Bio.PDB import PDBParser, Superimposer

def debug_pair():
    # 1. 自动寻找一对匹配的文件
    pred_dir = "esmfold_predictions"
    gt_dir = "ground_truth_pdbs"
    
    found_pair = None
    for pred_file in os.listdir(pred_dir):
        if not pred_file.endswith(".pdb"): continue
        
        # 还原文件名逻辑
        base_name = pred_file.split("_design")[0] # Me_1021...
        # 清理逻辑
        safe_name = "".join([c for c in base_name if c.isalnum() or c in ('_','-')])
        
        gt_path = os.path.join(gt_dir, f"{safe_name}.pdb")
        pred_path = os.path.join(pred_dir, pred_file)
        
        if os.path.exists(gt_path):
            found_pair = (gt_path, pred_path)
            break
    
    if not found_pair:
        print("❌ 无法找到任何一对匹配的文件！请检查文件名。")
        print(f"Pred dir sample: {os.listdir(pred_dir)[:3]}")
        print(f"GT dir sample:   {os.listdir(gt_dir)[:3]}")
        return

    ref_pdb, pred_pdb = found_pair
    print(f"🔍 正在深度诊断文件对：")
    print(f"  Reference (GT):   {ref_pdb}")
    print(f"  Prediction (ESM): {pred_pdb}")
    
    # 2. 详细解析过程
    parser = PDBParser(QUIET=False) # 打开啰嗦模式
    
    print("\n--- 解析 Reference PDB ---")
    try:
        s_ref = parser.get_structure("ref", ref_pdb)
        atoms_ref = [a for a in s_ref.get_atoms() if a.name == 'CA']
        print(f"  ✅ 解析成功。找到 {len(atoms_ref)} 个 CA 原子。")
        print(f"  前5个残基: {[r.get_resname() for r in list(s_ref.get_residues())[:5]]}")
    except Exception as e:
        print(f"  ❌ 解析失败: {e}")
        # 读取文件内容看看
        with open(ref_pdb, 'r') as f:
            print("  文件头5行内容:")
            print(f.read(200))
        return

    print("\n--- 解析 Prediction PDB ---")
    try:
        s_pred = parser.get_structure("pred", pred_pdb)
        atoms_pred = [a for a in s_pred.get_atoms() if a.name == 'CA']
        print(f"  ✅ 解析成功。找到 {len(atoms_pred)} 个 CA 原子。")
    except Exception as e:
        print(f"  ❌ 解析失败: {e}")
        return

    # 3. 尝试叠合
    print("\n--- 尝试叠合 (Superimpose) ---")
    min_len = min(len(atoms_ref), len(atoms_pred))
    print(f"  对齐长度: {min_len}")
    
    if min_len < 3:
        print("  ❌ 失败：原子数量太少 (<3)，无法计算 RMSD。")
        return
        
    try:
        sup = Superimposer()
        sup.set_atoms(atoms_ref[:min_len], atoms_pred[:min_len])
        print(f"  ✅ 叠合成功！")
        print(f"  🏆 RMSD = {sup.rms:.4f}")
    except Exception as e:
        print(f"  ❌ 叠合报错: {e}")

if __name__ == "__main__":
    debug_pair()
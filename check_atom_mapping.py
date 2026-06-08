import os
import glob
from biopandas.pdb import PandasPdb
import pandas as pd

# 设定你的原始数据目录
RAW_PDB_DIR = "nmethyl_data/raw_pdb/" 

# 所有的 HETATM 代码 (你提供的列表)
target_res_names = [
    '5JP', 'E9M', 'E9V', 'EME', 'GNC', 'IML', 'MAA', 'MEA', 'MLE', 'MME', 
    'MMO', 'MVA', 'NCY', 'NMK', 'NZC', 'SAR', 'SER', 'SOQ', 'YNM', 'ZCA'
]

def identify_parent(residue_name, atom_list):
    """
    根据侧链原子推断天然氨基酸类型
    """
    atoms = set(atom_list)
    
    # 1. 特殊原子判断
    if 'SE' in atoms: return 'MSE (Met/Se)' # 硒代蛋氨酸
    if 'SG' in atoms: return 'CYS (C)'      # 半胱氨酸 (有硫)
    if 'SD' in atoms: return 'MET (M)'      # 蛋氨酸 (有硫)
    
    # 2. 芳香族/环状判断
    if 'CZ' in atoms and 'OH' in atoms: return 'TYR (Y)' # 酪氨酸 (苯环+羟基)
    if 'CZ2' in atoms and 'CH2' in atoms: return 'TRP (W)' # 色氨酸 (双环)
    if 'CZ' in atoms and 'NH1' in atoms: return 'ARG (R)' # 精氨酸 (长链+氮)
    if 'CZ' in atoms: return 'PHE (F)'      # 苯丙氨酸 (只有苯环)
    if 'ND1' in atoms or 'NE2' in atoms: return 'HIS (H)' # 组氨酸 (咪唑环)
    
    # 3. 极性/电荷判断
    if 'OD1' in atoms and 'OD2' in atoms: return 'ASP (D)' # 天冬氨酸
    if 'OE1' in atoms and 'OE2' in atoms: return 'GLU (E)' # 谷氨酸
    if 'OD1' in atoms and 'ND2' in atoms: return 'ASN (N)' # 天冬酰胺
    if 'OE1' in atoms and 'NE2' in atoms: return 'GLN (Q)' # 谷氨酰胺
    if 'NZ' in atoms: return 'LYS (K)'      # 赖氨酸
    
    # 4. 短侧链判断
    if 'OG' in atoms or 'OG1' in atoms: 
        if 'CG2' in atoms: return 'THR (T)' # 苏氨酸 (有甲基)
        return 'SER (S)'                    # 丝氨酸
    
    if 'CG1' in atoms and 'CG2' in atoms:
        if 'CD1' in atoms: return 'ILE (I)' # 异亮氨酸
        return 'VAL (V)'                    # 缬氨酸
        
    if 'CG' in atoms and 'CD1' in atoms and 'CD2' in atoms: return 'LEU (L)' # 亮氨酸
    if 'CB' in atoms and len(atoms) <= 5: return 'ALA (A)' # 丙氨酸 (只有beta碳)
    if 'CA' in atoms: return 'GLY (G)'      # 甘氨酸 (无侧链, Sarcosine)
    
    return "UNKNOWN"

def check_mapping():
    print(f"正在分析 {RAW_PDB_DIR} 下的文件原子组成...")
    pdb_files = glob.glob(os.path.join(RAW_PDB_DIR, "*.pdb"))
    
    found_map = {}
    
    # 为了效率，我们只要找到每个残基的一个样本即可
    for pdb_file in pdb_files:
        if len(found_map) >= len(target_res_names): break
        
        try:
            ppdb = PandasPdb().read_pdb(pdb_file)
            df = pd.concat([ppdb.df['ATOM'], ppdb.df['HETATM']], ignore_index=True)
            
            # 找到当前文件里包含的 HETATM 残基
            unique_res = df['residue_name'].unique()
            for res in unique_res:
                if res in target_res_names and res not in found_map:
                    # 提取该残基的所有原子名
                    res_atoms = df[df['residue_name'] == res]['atom_name'].tolist()
                    # 过滤掉主链原子 (N, CA, C, O) 和 甲基原子 (CN, CM等)
                    side_chain_atoms = [a for a in res_atoms if a not in ['N', 'CA', 'C', 'O', 'OXT', 'H', 'CN', 'CM', 'H1', 'H2', 'H3']]
                    
                    parent = identify_parent(res, side_chain_atoms)
                    found_map[res] = {
                        'parent': parent,
                        'file': os.path.basename(pdb_file),
                        'atoms': side_chain_atoms
                    }
                    print(f"检测到: {res} -> 推测母体: {parent} (依据原子: {side_chain_atoms})")
        except Exception:
            continue

    print("\n" + "="*50)
    print("最终验证结果 (请根据此表修改 nmethyl_config.py)")
    print("="*50)
    print(f"{'HETATM':<10} | {'Parent AA':<15} | {'Verification Details'}")
    print("-" * 50)
    
    for res in sorted(target_res_names):
        if res in found_map:
            info = found_map[res]
            print(f"{res:<10} | {info['parent']:<15} | Found in {info['file']}")
        else:
            print(f"{res:<10} | {'MISSING':<15} | 未在当前PDB目录中找到")

if __name__ == "__main__":
    check_mapping()
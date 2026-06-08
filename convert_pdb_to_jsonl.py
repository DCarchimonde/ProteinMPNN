import os
import json
import numpy as np
from Bio.PDB import PDBParser
import warnings
from Bio import BiopythonWarning

warnings.simplefilter('ignore', BiopythonWarning)

# 氨基酸三字母转单字母字典
AA_3_TO_1 = {
    'ALA': 'A', 'CYS': 'C', 'ASP': 'D', 'GLU': 'E', 'PHE': 'F',
    'GLY': 'G', 'HIS': 'H', 'ILE': 'I', 'LYS': 'K', 'LEU': 'L',
    'MET': 'M', 'ASN': 'N', 'PRO': 'P', 'GLN': 'Q', 'ARG': 'R',
    'SER': 'S', 'THR': 'T', 'VAL': 'V', 'TRP': 'W', 'TYR': 'Y'
}

def parse_pdb_to_dict(pdb_path):
    parser = PDBParser()
    structure = parser.get_structure('pdb', pdb_path)
    
    pdb_dict = {}
    pdb_dict['name'] = os.path.basename(pdb_path).replace('.pdb', '')
    
    for model in structure:
        for chain in model:
            chain_id = chain.id
            seq = ""
            coords_N = []
            coords_CA = []
            coords_C = []
            coords_O = []
            
            for residue in chain:
                # 过滤掉水分子和杂原子
                if residue.id[0] != ' ':
                    continue
                
                res_name = residue.resname
                if res_name not in AA_3_TO_1:
                    continue # 遇到不认识的直接跳过
                
                # 确保 N, CA, C, O 四个骨架原子都在
                if all(atom in residue for atom in ['N', 'CA', 'C', 'O']):
                    seq += AA_3_TO_1[res_name]
                    coords_N.append(residue['N'].coord.tolist())
                    coords_CA.append(residue['CA'].coord.tolist())
                    coords_C.append(residue['C'].coord.tolist())
                    coords_O.append(residue['O'].coord.tolist())
            
            if len(seq) > 0:
                pdb_dict[f'seq_chain_{chain_id}'] = seq
                pdb_dict[f'N_chain_{chain_id}'] = coords_N
                pdb_dict[f'CA_chain_{chain_id}'] = coords_CA
                pdb_dict[f'C_chain_{chain_id}'] = coords_C
                pdb_dict[f'O_chain_{chain_id}'] = coords_O
                
                # 默认把所有的链都加入 visible_list（对于推理，我们可能需要指定哪条是多肽，这里全给）
                if 'visible_list' not in pdb_dict:
                    pdb_dict['visible_list'] = []
                pdb_dict['visible_list'].append(chain_id)
                
        break # 只处理第一个 model
    
    return pdb_dict

def main():
    # 👉 把这里的路径改成存放你 17 个 PDB 文件的文件夹路径！
    pdb_folder = "./pdb_downloads" 
    output_jsonl = "./17_complexes_native.jsonl"
    
    if not os.path.exists(pdb_folder):
        print(f"❌ 找不到文件夹: {pdb_folder}，请修改代码里的路径！")
        return

    with open(output_jsonl, 'w') as f:
        count = 0
        for filename in os.listdir(pdb_folder):
            if filename.endswith(".pdb"):
                pdb_path = os.path.join(pdb_folder, filename)
                try:
                    pdb_dict = parse_pdb_to_dict(pdb_path)
                    f.write(json.dumps(pdb_dict) + '\n')
                    count += 1
                    print(f"✅ 成功解析: {filename}")
                except Exception as e:
                    print(f"⚠️ 解析失败 {filename}: {e}")
                    
    print(f"\n🎉 大功告成！成功将 {count} 个 PDB 转换并保存为 {output_jsonl}")

if __name__ == "__main__":
    main()
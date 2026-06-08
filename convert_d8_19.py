import os
import json
import warnings
from Bio.PDB import PDBParser
from Bio import BiopythonWarning

warnings.simplefilter('ignore', BiopythonWarning)

# 氨基酸三字母转单字母字典（加入甲基化特异性映射）
AA_3_TO_1 = {
    'ALA': 'A', 'CYS': 'C', 'ASP': 'D', 'GLU': 'E', 'PHE': 'F',
    'GLY': 'G', 'HIS': 'H', 'ILE': 'I', 'LYS': 'K', 'LEU': 'L',
    'MET': 'M', 'ASN': 'N', 'PRO': 'P', 'GLN': 'Q', 'ARG': 'R',
    'SER': 'S', 'THR': 'T', 'VAL': 'V', 'TRP': 'W', 'TYR': 'Y',
    # 💡 甲基化氨基酸映射（使用模型规定的小写字母）
    'MAA': 'a',  # N-甲基丙氨酸 (Methyl-Alanine)
    'MLU': 'l',  # N-甲基亮氨酸 (Methyl-Leucine)
}

def parse_pdb_to_dict(pdb_path):
    parser = PDBParser()
    structure = parser.get_structure('pdb', pdb_path)
    
    pdb_dict = {}
    pdb_dict['name'] = os.path.basename(pdb_path).replace('.pdb', '')
    pdb_dict['visible_list'] = []
    pdb_dict['masked_list'] = [] # 模型预测必须用 masked_list
    
    for model in structure:
        for chain in model:
            # 💡 修复 1：如果链 ID 是空格，强行赋予 'A'
            raw_id = chain.id.strip()
            chain_id = "A" if raw_id == "" else raw_id
            
            seq = ""
            coords_N, coords_CA, coords_C, coords_O = [], [], [], []
            
            for residue in chain:
                # 过滤杂原子（水分子等）
                if residue.id[0] != ' ':
                    continue
                
                res_name = residue.resname
                
                # 💡 修复 2：暴露未知氨基酸，而不是悄悄删掉
                if res_name not in AA_3_TO_1:
                    print(f"⚠️ 发现非标准氨基酸: '{res_name}'！已暂存为 'X'。")
                    mapped_char = 'X'
                else:
                    mapped_char = AA_3_TO_1[res_name]
                
                # 确保主链四个原子都在
                if all(atom in residue for atom in ['N', 'CA', 'C', 'O']):
                    seq += mapped_char
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
            
            # 将这条链加入预测目标列表
            pdb_dict['masked_list'].append(chain_id)
            
        break # 只处理第一个 model
    
    return pdb_dict

def main():
    pdb_path = "d8_19.pdb"   # 你的 PDB 文件
    output_jsonl = "d8_19.jsonl"
    
    if not os.path.exists(pdb_path):
        print(f"❌ 找不到文件: {pdb_path}")
        return

    pdb_dict = parse_pdb_to_dict(pdb_path)
    
    with open(output_jsonl, 'w') as f:
        f.write(json.dumps(pdb_dict) + '\n')
        
    print(f"✅ 成功解析并保存至: {output_jsonl}")
    print(f"💡 最终序列为: {pdb_dict.get('seq_chain_A', '解析失败')}")

if __name__ == "__main__":
    main()
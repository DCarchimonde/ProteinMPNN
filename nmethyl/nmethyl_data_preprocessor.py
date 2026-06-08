import os
import json
import glob
from collections import defaultdict
import pandas as pd 

try:
    from biopandas.pdb import PandasPdb
except ImportError:
    print("错误: 本脚本需要 biopandas 库。请通过 'pip install biopandas' 进行安装。")
    PandasPdb = None

# =============================================================================
# 1. 映射表 (必须与 config 保持一致)
# =============================================================================

# N-甲基化氨基酸：基于原子结构验证的准确映射
NMETHYL_RESIDUE_MAP = {
    'MAA': 'a', 'SAR': 'g', 'MLE': 'l', 'IML': 'i', 'MVA': 'v', 
    'MME': 'm', 'MEA': 'f', 'YNM': 'y', 'E9M': 'w', '5JP': 's', 
    'SER': 's', 'NZC': 't', 'NCY': 'c', 'ZCA': 'n', 'GNC': 'q', 
    'SOQ': 'd', 'EME': 'e', 'NMK': 'k', 'MMO': 'r', 'E9V': 'h'
}

# 标准天然氨基酸
NATURAL_RESIDUE_MAP = {
    'ALA': 'A', 'ARG': 'R', 'ASN': 'N', 'ASP': 'D', 'CYS': 'C',
    'GLN': 'Q', 'GLU': 'E', 'GLY': 'G', 'HIS': 'H', 'ILE': 'I',
    'LEU': 'L', 'LYS': 'K', 'MET': 'M', 'PHE': 'F', 'PRO': 'P',
    'SER': 'S', 'THR': 'T', 'TRP': 'W', 'TYR': 'Y', 'VAL': 'V',
}

class NmethylDataProcessor:
    """
    一个基于biopandas的、经过修复和增强的PDB解析器。
    它能够正确处理包含N-甲基化氨基酸 (HETATM) 和天然氨基酸 (ATOM) 的PDB文件。
    """
    def __init__(self):
        if PandasPdb is None:
            raise ImportError("biopandas 未安装，无法初始化处理器。")
        self.residue_map = {**NATURAL_RESIDUE_MAP, **NMETHYL_RESIDUE_MAP}

    def process_pdb(self, pdb_path):
        """
        使用biopandas处理单个PDB文件，并格式化为模型输入。
        """
        try:
            # 抑制 biopandas 的一些烦人警告
            ppdb = PandasPdb().read_pdb(pdb_path)
        except Exception as e:
            print(f"  [警告] 无法读取 {os.path.basename(pdb_path)}: {e}")
            return None
        
        # --- [关键步骤] 合并 ATOM 和 HETATM ---
        df_atom = ppdb.df['ATOM']
        df_het = ppdb.df['HETATM']
        
        # 很多 N-甲基化残基 (如 MAA, SAR) 在 PDB 中完全存储为 HETATM
        # 必须将它们合并才能形成完整的蛋白质链
        if df_het.empty:
            df = df_atom
        else:
            df = pd.concat([df_atom, df_het], ignore_index=True)
        
        # 按链ID、残基编号、原子编号排序，确保序列顺序正确
        # 注意：重置索引以防止合并后的索引冲突
        df = df.sort_values(by=['chain_id', 'residue_number', 'atom_number']).reset_index(drop=True)

        output = {}
        all_chains = sorted(df['chain_id'].unique())
        
        output['name'] = os.path.basename(pdb_path).replace('.pdb', '')
        output['num_of_chains'] = len(all_chains)
        output['masked_list'] = all_chains
        output['visible_list'] = []
        
        full_sequence = ""

        for chain_id in all_chains:
            chain_df = df[df['chain_id'] == chain_id]
            
            seq_chain = []
            coords = defaultdict(list)
            
            # 使用 groupby 按残基迭代，自动保持 PDB 中的顺序
            # sort=False 很重要，防止按残基编号重新排序（有时PDB编号不连续）
            for res_num, res_df in chain_df.groupby('residue_number', sort=False):
                # 获取该残基的名称 (取第一行的 residue_name)
                res_name = res_df['residue_name'].iloc[0]
                
                # --- [序列转换] ---
                if res_name in self.residue_map:
                    token = self.residue_map[res_name]
                    seq_chain.append(token)
                    
                    # 提取主链原子坐标 N, CA, C, O
                    # 将每行转换为字典以便快速查找
                    atom_coords = {row['atom_name']: [row['x_coord'], row['y_coord'], row['z_coord']] for _, row in res_df.iterrows()}
                    
                    # 即使是 HETATM，ProteinMPNN 只需要主链骨架
                    # 对于 SAR (Sarcosine/Gly)，它没有 CB，但这不影响 backbone 提取
                    coords['N_chain_' + chain_id].append(atom_coords.get('N', [None, None, None]))
                    coords['CA_chain_' + chain_id].append(atom_coords.get('CA', [None, None, None]))
                    coords['C_chain_' + chain_id].append(atom_coords.get('C', [None, None, None]))
                    coords['O_chain_' + chain_id].append(atom_coords.get('O', [None, None, None]))
                else:
                    # 如果遇到映射表中没有的残基 (如水分子 HOH, 离子等)，跳过
                    pass

            # 只有当链不为空时才保存
            if seq_chain:
                output[f'seq_chain_{chain_id}'] = "".join(seq_chain)
                output.update(coords)
                full_sequence += "".join(seq_chain)

        output['seq'] = full_sequence
        
        # 简单校验：如果没有读到任何序列，返回 None
        if not full_sequence:
            return None
            
        return output

    def create_training_dataset(self, pdb_directory, output_jsonl):
        """
        遍历PDB目录，创建包含N-甲基化氨基酸的训练数据集。
        """
        print(f"开始从 '{pdb_directory}' 创建数据集...")
        # 支持 .pdb 结尾的文件
        pdb_files = glob.glob(os.path.join(pdb_directory, "*.pdb"))
        count = 0
        
        with open(output_jsonl, 'w') as f:
            for pdb_file in pdb_files:
                try:
                    processed_data = self.process_pdb(pdb_file)
                    if processed_data:
                        f.write(json.dumps(processed_data) + '\n')
                        count += 1
                        if count % 100 == 0:
                            print(f"  已处理 {count} 个文件...")
                except Exception as e:
                    print(f"  [失败] 处理 {os.path.basename(pdb_file)} 时发生错误: {e}")
        
        print(f"\n数据集创建完成！共处理了 {count} 个有效的PDB文件。")
        print(f"输出文件已保存至: {output_jsonl}")
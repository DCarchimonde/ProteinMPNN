import os
import glob
from collections import defaultdict

def find_unique_hetatm_residues(directory_path):
    """
    遍历指定目录下的所有PDB文件，查找并报告所有唯一的HETATM残基名称及其首次出现的记录。

    Args:
        directory_path (str): 要搜索的PDB文件所在的目录路径。
    """
    print(f"--- 开始在目录 '{directory_path}' 中搜索唯一的HETATM残基 ---\n")
    
    # 构建搜索路径以匹配所有.pdb文件
    search_path = os.path.join(directory_path, "*.pdb")
    pdb_files = glob.glob(search_path)
    
    if not pdb_files:
        print(f"错误：在目录 '{directory_path}' 中没有找到任何.pdb文件。")
        return

    # 使用一个字典来存储唯一的残基名称及其首次出现的文件和完整行信息
    unique_hetatm_residues = {}

    # 遍历找到的每个PDB文件
    for pdb_file in pdb_files:
        file_name = os.path.basename(pdb_file)
        try:
            with open(pdb_file, 'r') as f:
                for line in f:
                    if line.startswith('HETATM'):
                        # PDB格式中，残基名称在第18-20列 (索引17到19)
                        residue_name = line[17:20].strip()
                        
                        # 如果这个残基名称是第一次遇到，就记录下来
                        if residue_name not in unique_hetatm_residues:
                            unique_hetatm_residues[residue_name] = {
                                'file': file_name,
                                'line': line.strip()
                            }
        except Exception as e:
            print(f"处理文件 {file_name} 时发生错误: {e}")

    print("\n--- 搜索完成 ---")
    if not unique_hetatm_residues:
        print("在所有文件中均未找到HETATM记录。")
    else:
        print(f"总共找到了 {len(unique_hetatm_residues)} 种唯一的HETATM残基：")
        print("-" * 40)
        # 打印表头
        print(f"{'残基名称':<12} | {'首次出现的文件':<30} | {'完整记录示例'}")
        print(f"{'-'*12} | {'-'*30} | {'-'*20}")
        
        # 排序后打印结果，更清晰
        for name in sorted(unique_hetatm_residues.keys()):
            info = unique_hetatm_residues[name]
            print(f"{name:<12} | {info['file']:<30} | {info['line']}")
        print("-" * 40)

if __name__ == "__main__":
    # 设置您要搜索的PDB文件目录
    pdb_directory = 'nmethyl_data/training_set/'
    
    # 检查目录是否存在
    if not os.path.isdir(pdb_directory):
        print(f"错误: 目录 '{pdb_directory}' 不存在或不是一个有效的目录。")
        print("请确保脚本与'nmethyl_data'目录在同一级，或者提供正确的路径。")
    else:
        find_unique_hetatm_residues(pdb_directory)


import json
import argparse
from collections import Counter
import os

# --- [更新] 从您的官方配置文件导入氨基酸定义 ---
# 这确保了此诊断脚本与您的训练脚本使用完全相同的标准。
try:
    from nmethyl.utils.nmethyl_config import NATURAL_AA_ALPHABET, METHYL_AA_ALPHABET
except ImportError:
    print("警告: 无法从 'nmethyl.utils.nmethyl_config' 导入配置。")
    print("将使用脚本内定义的默认字母表。")
    NATURAL_AA_ALPHABET = 'ACDEFGHIKLMNPQRSTVWY'
    METHYL_AA_ALPHABET = 'cdefhilmqrstvy'


def analyze_data_file(file_path):
    """
    分析指定的.jsonl文件，统计各类氨基酸的数量和比例，
    并明确列出所有未知的字符。
    """
    print("\n" + "="*60)
    print(f"正在分析文件: {file_path}")
    print("="*60)

    if not os.path.exists(file_path):
        print(f"错误: 文件 '{file_path}' 不存在。请检查路径是否正确。")
        return

    residue_counter = Counter()

    try:
        with open(file_path, 'r') as f:
            for line_num, line in enumerate(f, 1):
                try:
                    item = json.loads(line)
                    # 遍历所有可能的序列键
                    for key in item:
                        if key.startswith('seq_chain_'):
                            sequence = item[key]
                            residue_counter.update(sequence)
                except json.JSONDecodeError:
                    print(f"警告: 第 {line_num} 行JSON解析失败，已跳过。")
                    continue
        
        # 初始化计数器
        natural_count = 0
        nmethyl_count = 0
        other_count = 0
        other_chars_counter = Counter() # <-- [新功能] 用于统计未知字符的具体类型和数量

        # 根据字符集分类统计
        for residue, count in residue_counter.items():
            if residue in NATURAL_AA_ALPHABET:
                natural_count += count
            elif residue in METHYL_AA_ALPHABET:
                nmethyl_count += count
            else:
                other_count += count
                other_chars_counter[residue] += count # <-- [新功能] 记录未知字符
        
        total_residues = sum(residue_counter.values())

        # 打印总结报告
        print(f"分析完成！\n")
        print(f"总残基数量: {total_residues}")
        print("-" * 30)
        
        if total_residues > 0:
            natural_perc = (natural_count / total_residues) * 100
            nmethyl_perc = (nmethyl_count / total_residues) * 100
            other_perc = (other_count / total_residues) * 100
            
            print(f"天然氨基酸      : {natural_count} ({natural_perc:.2f}%)")
            print(f"N-甲基化氨基酸: {nmethyl_count} ({nmethyl_perc:.2f}%)")
            print(f"其他/未知字符   : {other_count} ({other_perc:.2f}%)")
        else:
            print("文件中未找到任何序列数据。")

        print("-" * 30)

        # [新功能] 如果存在未知字符，则打印详细列表
        if other_count > 0:
            print("检测到的“其他/未知字符”详细列表:")
            for char, count in other_chars_counter.items():
                print(f"  - 字符 '{char}': {count} 次")
        
        print("-" * 30)

        if nmethyl_count > 0:
            print("检测到的N-甲基化氨基酸详细列表:")
            for residue, count in sorted(residue_counter.items()):
                if residue in METHYL_AA_ALPHABET:
                    print(f"  - '{residue}': {count} 次")
        else:
            print("结论: 文件中未检测到任何已定义的N-甲基化氨基酸。")

    except Exception as e:
        print(f"读取或处理文件时发生未知错误: {e}")


def main():
    parser = argparse.ArgumentParser(description="检查数据文件中各类氨基酸的含量，并识别未知字符。")
    parser.add_argument(
        "nmethyl_data/processed/all_data.jsonl",
        nargs='+',
        type=str,
        help="需要分析的一个或多个.jsonl文件的路径。"
    )
    args = parser.parse_args()

    for file_path in args.file_paths:
        analyze_data_file(file_path)

if __name__ == "__main__":
    main()


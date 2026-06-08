# 创建预处理脚本 preprocess_data.py
import os
import sys
import json
import random

# 将当前目录（'.'）添加到Python的模块搜索路径中
# 这是为了确保可以成功导入 nmethyl 包中的模块
# 假设您从项目的根目录运行此脚本
sys.path.append('.')

from nmethyl.nmethyl_data_preprocessor import NmethylDataProcessor

def create_full_dataset(processor, raw_data_dir, output_file):
    """
    使用 NmethylDataProcessor 处理所有原始 PDB 文件，并创建一个完整的数据集文件。
    
    Args:
        processor (NmethylDataProcessor): 数据处理器实例。
        raw_data_dir (str): 包含原始 .pdb 文件的目录路径。
        output_file (str): 用于保存所有处理后数据的 .jsonl 文件的路径。
    """
    print(f"开始从 '{raw_data_dir}' 目录进行数据预处理...")
    
    # 确保输出文件的目录存在，如果不存在则创建
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # 调用库提供的函数来处理目录中的所有 PDB 文件。
    # 虽然函数名叫 create_training_dataset，但在这里我们用它来处理所有数据，生成一个完整的数据集。
    processor.create_training_dataset(raw_data_dir, output_file)
    print(f"成功创建完整数据集，已保存至 '{output_file}'")

def split_dataset(full_dataset_file, train_file, test_file, train_ratio=0.8):
    """
    将包含所有数据的文件分割成训练集和测试集。
    
    Args:
        full_dataset_file (str): 包含所有已处理数据的 .jsonl 文件的路径。
        train_file (str): 用于保存训练数据的 .jsonl 文件的路径。
        test_file (str): 用于保存测试数据的 .jsonl 文件的路径。
        train_ratio (float): 数据集中用于训练的样本比例。
    """
    print(f"正在将 '{full_dataset_file}' 分割为训练集和测试集...")

    try:
        with open(full_dataset_file, 'r') as f:
            # 逐行读取json对象，构建数据列表
            data = [json.loads(line) for line in f]
    except FileNotFoundError:
        print(f"错误：输入文件 '{full_dataset_file}' 未找到。请先运行 create_full_dataset。")
        return
    except json.JSONDecodeError:
        print(f"错误：文件 '{full_dataset_file}' 包含无效的JSON格式。")
        return

    # 随机打乱数据，确保分割后的训练集和测试集具有相似的数据分布
    random.shuffle(data)
    
    # 计算分割点
    split_idx = int(len(data) * train_ratio)
    
    train_data = data[:split_idx]
    test_data = data[split_idx:]
    
    # --- 保存训练集 ---
    # 确保输出目录存在
    os.makedirs(os.path.dirname(train_file), exist_ok=True)
    with open(train_file, 'w') as f:
        for item in train_data:
            f.write(json.dumps(item) + '\n')
    
    # --- 保存测试集 ---
    # 确保输出目录存在
    os.makedirs(os.path.dirname(test_file), exist_ok=True)
    with open(test_file, 'w') as f:
        for item in test_data:
            f.write(json.dumps(item) + '\n')
            
    print("数据分割完成:")
    print(f"  - {len(train_data)} 个训练样本已保存至 '{train_file}'")
    print(f"  - {len(test_data)} 个测试样本已保存至 '{test_file}'")

def main():
    """
    主执行函数，协调整个预处理流程：
    1. 设置文件和目录的路径。
    2. 处理所有原始 PDB 数据，生成一个大文件。
    3. 将这个大文件分割成训练集和测试集。
    """
    # --- 路径配置 ---
    # 包含原始PDB文件的输入目录
    RAW_PDB_DIR = "nmethyl_data/raw_pdb/"
    
    # 临时文件路径，用于存放所有处理过的数据
    ALL_DATA_FILE = "nmethyl_data/processed/all_data.jsonl" 
    
    # 最终的训练集和测试集输出路径
    TRAIN_SET_FILE = "nmethyl_data/training_set/train.jsonl"
    TEST_SET_FILE = "nmethyl_data/test_set/test.jsonl"
    
    # --- 执行步骤 ---
    processor = NmethylDataProcessor()
    
    # 步骤 1: 预处理所有 PDB 文件
    create_full_dataset(processor, RAW_PDB_DIR, ALL_DATA_FILE)
    
    # 步骤 2: 将生成的大文件分割成独立的训练集和测试集
    split_dataset(ALL_DATA_FILE, TRAIN_SET_FILE, TEST_SET_FILE, train_ratio=0.8)

if __name__ == "__main__":
    main()

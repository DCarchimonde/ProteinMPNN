import json
from collections import defaultdict

def analyze_amino_acid_data(data_path, sample_size=5):
    """
    分析JSONL数据的结构，检测氨基酸类型存储位置及数量
    
    参数:
        data_path: JSONL文件路径
        sample_size: 抽取前N条数据样本分析结构
    """
    print(f"\n===== 分析数据文件: {data_path} =====")
    
    # 1. 检查文件是否为空
    if not os.path.exists(data_path):
        print(f"错误: 文件不存在 - {data_path}")
        return None
    if os.path.getsize(data_path) == 0:
        print(f"错误: 文件为空 - {data_path}")
        return None
    
    # 2. 读取样本数据，分析结构（字段名、嵌套关系）
    sample_data = []
    all_fields = set()  # 记录所有出现过的字段名
    nested_fields = defaultdict(set)  # 记录嵌套字段（如 key.subkey）
    
    with open(data_path, 'r') as f:
        for i, line in enumerate(f):
            if i >= sample_size:
                break  # 只看前N条样本
            try:
                item = json.loads(line.strip())
                sample_data.append(item)
                # 提取所有字段名（包括嵌套字段）
                _extract_fields(item, parent_key='', all_fields=all_fields, nested_fields=nested_fields)
            except json.JSONDecodeError:
                print(f"警告: 第{i+1}行数据格式错误，无法解析为JSON")
                continue
    
    if not sample_data:
        print("错误: 未读取到有效数据（可能全是格式错误）")
        return None
    
    # 3. 打印数据结构分析结果
    print(f"\n--- 数据结构分析（前{len(sample_data)}条样本） ---")
    print(f"所有顶层字段: {sorted(all_fields)}")
    if nested_fields:
        print("嵌套字段（key.subkey）:")
        for parent, subs in nested_fields.items():
            print(f"  {parent}: {sorted(subs)}")
    
    # 4. 尝试提取氨基酸类型（基于常见字段名猜测，结合结构分析）
    # 可能的氨基酸存储字段（扩展版，涵盖常见命名）
    possible_aa_fields = [
        # 单个氨基酸记录（如每条数据是一个残基）
        "aa", "amino_acid", "residue", "res", "aminoacid",
        # 序列列表（如每条数据是一个蛋白质，包含多个残基）
        "sequence", "seq", "aa_sequence", "residues", "res_list",
        # 嵌套字段（根据上面的nested_fields补充）
        "features.aa", "structure.residues", "data.amino_acids"
    ]
    
    # 从样本中收集可能的氨基酸字段
    candidate_fields = []
    for field in possible_aa_fields:
        if '.' in field:
            parent, sub = field.split('.', 1)
            if parent in all_fields and sub in nested_fields.get(parent, set()):
                candidate_fields.append(field)
        else:
            if field in all_fields:
                candidate_fields.append(field)
    
    print(f"\n--- 可能存储氨基酸信息的字段（{len(candidate_fields)}个） ---")
    for field in candidate_fields:
        print(f"  候选字段: {field}")
    
    # 5. 从所有数据中提取氨基酸类型
    amino_types = set()
    total_items = 0
    empty_items = 0  # 字段存在但值为空的记录数
    
    with open(data_path, 'r') as f:
        for line in f:
            total_items += 1
            try:
                item = json.loads(line.strip())
            except json.JSONDecodeError:
                continue  # 跳过格式错误的记录
            
            # 遍历候选字段，提取氨基酸类型
            for field in candidate_fields:
                # 处理嵌套字段（如 "structure.residues"）
                if '.' in field:
                    keys = field.split('.')
                    current = item
                    valid = True
                    for k in keys:
                        if k not in current:
                            valid = False
                            break
                        current = current[k]
                    value = current if valid else None
                else:
                    value = item.get(field, None)
                
                # 提取类型（处理单个值或列表）
                if value is not None:
                    if isinstance(value, list):
                        # 列表形式（如序列）：逐个添加元素
                        for v in value:
                            if isinstance(v, str) and v.strip() != '':
                                amino_types.add(v.strip())
                    elif isinstance(value, str) and value.strip() != '':
                        # 单个字符串（如单个残基）
                        amino_types.add(value.strip())
                    else:
                        # 非字符串类型（如数字编码，需要进一步处理）
                        amino_types.add(str(value))  # 先转为字符串记录
                else:
                    empty_items += 1
    
    # 6. 输出最终结果
    print(f"\n--- 氨基酸类型统计 ---")
    print(f"总记录数: {total_items}")
    print(f"有效氨基酸类型数量: {len(amino_types)}")
    if amino_types:
        print(f"具体类型（前20种，若超过）: {sorted(amino_types)[:20]}")
        if len(amino_types) > 20:
            print(f"（省略 {len(amino_types)-20} 种）")
    else:
        print("未检测到任何氨基酸类型！可能原因：")
        print("  1. 候选字段不正确（需根据数据结构自定义字段名）")
        print("  2. 数据中氨基酸字段的值为空或格式错误")
        print("  3. 预处理脚本未正确写入氨基酸信息")
    
    return amino_types

def _extract_fields(data, parent_key, all_fields, nested_fields):
    """递归提取字段名（辅助函数）"""
    if isinstance(data, dict):
        for k, v in data.items():
            current_key = f"{parent_key}.{k}" if parent_key else k
            all_fields.add(k if not parent_key else parent_key)  # 顶层字段或父字段
            if isinstance(v, (dict, list)):
                _extract_fields(v, current_key, all_fields, nested_fields)
            if parent_key:
                nested_fields[parent_key].add(k)
    elif isinstance(data, list) and len(data) > 0 and isinstance(data[0], dict):
        # 列表中的元素是字典，提取第一个元素的字段作为代表
        _extract_fields(data[0], parent_key, all_fields, nested_fields)

# 执行分析（替换为你的数据路径）
if __name__ == "__main__":
    import os
    # 分析训练集和测试集
    train_data = "nmethyl_data/processed/all_data.jsonl"
    test_data = "nmethyl_data/test_sets/test.jsonl"
    
    print("===== 开始分析训练集 =====")
    train_amino_types = analyze_amino_acid_data(train_data)
    
    print("\n\n===== 开始分析测试集 =====")
    test_amino_types = analyze_amino_acid_data(test_data)
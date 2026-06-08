import argparse
import os
import sys
import torch
import numpy as np
import json
from collections import Counter

# --- [关键] 动态添加项目根目录到Python路径 ---
# 确保能正确导入 nmethyl 和 model_utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    # --- [关键] 从您的最终训练脚本导入必要的模块 ---
    # 确保使用与训练时完全一致的模型定义、配置和特征化函数
    from train_nmethyl_mpnn import ExtendedProteinMPNN, JSONLDataset, collate_fn, featurize_batch
    from nmethyl.utils.nmethyl_config import EXTENDED_AA_ALPHABET, NATURAL_AA_ALPHABET, METHYL_AA_ALPHABET
except ImportError as e:
    print(f"错误：无法导入必需的模块。请确保此脚本与 'train_nmethyl_mpnn.py' 在同一目录下，")
    print(f"或者您的项目结构和Python路径设置正确。")
    print(f"详细错误: {e}")
    sys.exit(1)

def analyze_correct_predictions(model_path, test_data_path, batch_size=8, device_str='cuda'):
    """
    加载模型和测试数据，分析预测正确的氨基酸种类。
    """
    device = torch.device(device_str if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    # 1. 加载模型
    print(f"正在加载模型: {model_path}")
    if not os.path.exists(model_path):
        print(f"错误: 模型文件 '{model_path}' 未找到。")
        return

    # --- [关键] 使用与训练时完全一致的模型参数初始化 ---
    model = ExtendedProteinMPNN(
        num_letters=len(EXTENDED_AA_ALPHABET),
        node_features=128, edge_features=128, hidden_dim=128,
        num_encoder_layers=3, num_decoder_layers=3, dropout=0.1
    ).to(device)

    try:
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        print("模型加载成功！")
    except Exception as e:
        print(f"加载模型状态字典时出错: {e}")
        return

    # 2. 加载测试数据
    print(f"正在加载测试数据: {test_data_path}")
    if not os.path.exists(test_data_path):
        print(f"错误: 测试数据文件 '{test_data_path}' 未找到。")
        return
    
    test_dataset = JSONLDataset(test_data_path, augment=False) # 测试时不使用增强
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    print(f"测试样本数量: {len(test_dataset)}")

    # 3. 运行预测并收集结果
    all_predictions = []
    all_targets = []
    print("正在运行预测...")
    with torch.no_grad():
        for batch_idx, batch in enumerate(test_loader):
            try:
                features = featurize_batch(batch, device)
                if features is None: continue
                # features 包含: X, S, mask, chain_M, residue_idx, chain_encoding_all
                log_probs_main, _ = model(*features) # 我们只关心主任务的预测
                predictions = torch.argmax(log_probs_main, dim=-1)
                
                mask_flat = features[2].cpu().numpy().flatten().astype(bool)
                targets_flat = features[1].cpu().numpy().flatten()
                predictions_flat = predictions.cpu().numpy().flatten()

                # 只收集有效位置（非填充）的预测和目标
                all_predictions.extend(predictions_flat[mask_flat])
                all_targets.extend(targets_flat[mask_flat])
                
                if (batch_idx + 1) % 10 == 0:
                    print(f"  已处理 {batch_idx + 1}/{len(test_loader)} 个批次")

            except Exception as e:
                print(f"处理批次 {batch_idx} 时发生错误: {e}")
                continue
    print("预测完成！")

    all_predictions = np.array(all_predictions)
    all_targets = np.array(all_targets)

    if len(all_targets) == 0:
        print("错误：未能从测试数据中提取任何有效的残基。")
        return

    # 4. 分析正解预测
    correct_indices = np.where(all_predictions == all_targets)[0]
    num_correct = len(correct_indices)
    total_residues = len(all_targets)
    overall_accuracy = num_correct / total_residues if total_residues > 0 else 0

    print("\n" + "="*60)
    print("正解预测分析报告")
    print("="*60)
    print(f"总测试残基数: {total_residues}")
    print(f"总正确预测数: {num_correct} (总体准确率: {overall_accuracy*100:.2f}%)")
    print("-" * 30)

    if num_correct == 0:
        print("没有任何残基被正确预测。")
        return

    # 获取所有正确预测的氨基酸的 *真实* 索引
    correctly_predicted_targets = all_targets[correct_indices]

    # 使用 Counter 统计每种正确预测的氨基酸
    correct_counter = Counter(correctly_predicted_targets)

    natural_correct_count = 0
    nmethyl_correct_count = 0
    natural_correct_details = Counter()
    nmethyl_correct_details = Counter()

    # 将索引转换为氨基酸字符并分类统计
    natural_indices_range = range(len(NATURAL_AA_ALPHABET))
    methyl_indices_start = len(NATURAL_AA_ALPHABET)
    # 假设 'X' 是最后一个字符
    methyl_indices_end = len(EXTENDED_AA_ALPHABET) - 1 

    for target_index, count in correct_counter.items():
        if target_index in natural_indices_range:
            natural_correct_count += count
            aa_char = EXTENDED_AA_ALPHABET[target_index]
            natural_correct_details[aa_char] += count
        elif methyl_indices_start <= target_index < methyl_indices_end:
            nmethyl_correct_count += count
            aa_char = EXTENDED_AA_ALPHABET[target_index]
            nmethyl_correct_details[aa_char] += count
        # 忽略 'X' 或其他可能的无效索引

    print(f"正确预测的天然氨基酸数量: {natural_correct_count}")
    if natural_correct_count > 0:
        print("  详细列表 (天然):")
        for aa, count in sorted(natural_correct_details.items()):
            print(f"    - '{aa}': {count} 次")
            
    print("-" * 30)
    print(f"正确预测的N-甲基化氨基酸数量: {nmethyl_correct_count}")
    if nmethyl_correct_count > 0:
        print("  详细列表 (N-甲基化):")
        for aa, count in sorted(nmethyl_correct_details.items()):
            print(f"    - '{aa}': {count} 次")
        print("\n结论：模型 **成功** 预测了部分N-甲基化氨基酸！")
    else:
        print("\n结论：在本次测试中，模型未能正确预测任何N-甲基化氨基酸。")
        
    print("="*60)


def main():
    parser = argparse.ArgumentParser(description="分析ProteinMPNN模型预测正确的氨基酸种类。")
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="训练好的最佳模型文件路径 (通常是 'best_model.pt')"
    )
    parser.add_argument(
        "--test_data",
        type=str,
        required=True,
        help="用于测试的 .jsonl 文件路径。"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="进行预测时使用的批次大小。"
    )
    parser.add_argument(
        "--device",
        type=str,
        default='cuda',
        choices=['cuda', 'cpu'],
        help="指定运行设备 ('cuda' 或 'cpu')."
    )
    args = parser.parse_args()

    analyze_correct_predictions(args.model_path, args.test_data, args.batch_size, args.device)

if __name__ == "__main__":
    main()

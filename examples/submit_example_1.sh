#!/bin/bash
# 修复：确保Python命令的参数换行时，每行末尾加反斜杠\（无空格）

# 1. 配置路径
folder_with_pdbs="./inputs/PDB_complexes/pdbs/"  # 确保inputs文件夹下有PDB文件
output_dir="./outputs_test/example_1_outputs"
path_for_parsed_chains="${output_dir}/parsed_pdbs.jsonl"

# 2. 创建输出目录
if [ ! -d "$output_dir" ]; then
    mkdir -p "$output_dir"
    echo "已创建输出目录：$output_dir"
fi

# 3. 解析PDB（转为jsonl格式）
python helper_scripts/parse_multiple_chains.py \
    --input_path="$folder_with_pdbs" \
    --output_path="$path_for_parsed_chains"

if [ $? -ne 0 ]; then
    echo "PDB解析失败！检查inputs文件夹是否有PDB文件"
    exit 1
fi
echo "PDB解析成功，保存路径：$path_for_parsed_chains"

# 4. 运行ProteinMPNN（关键修复：每行参数末尾加\，且\后无空格）
python protein_mpnn_run.py \
    --jsonl_path "$path_for_parsed_chains" \
    --out_folder "$output_dir" \
    --model_name "v_48_020" \
    --num_seq_per_target 2 \
    --sampling_temp "0.1" \
    --seed 37 \
    --batch_size 1

# 5. 检查运行结果
if [ $? -eq 0 ]; then
    echo "✅ ProteinMPNN运行成功！结果在：$output_dir"
else
    echo "❌ ProteinMPNN运行失败！常见原因：显存不足（可降batch_size）、模型权重缺失"
    exit 1
fi
#!/usr/bin/env bash
set -euo pipefail

# 用法：
# bash paper_clean_v28/05_make_release_repo.sh
# 或指定输出目录：
# bash paper_clean_v28/05_make_release_repo.sh /root/autodl-tmp/ProteinMPNN_clean_v28_release

SRC_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RELEASE_DIR="${1:-$HOME/ProteinMPNN_clean_v28_release}"

if [ -e "$RELEASE_DIR" ]; then
  echo "[错误] 输出目录已存在：$RELEASE_DIR"
  echo "如果确认要重建，请先手动移动或删除它。"
  exit 1
fi

echo "[1/6] 创建干净发布目录：$RELEASE_DIR"
mkdir -p "$RELEASE_DIR"
cd "$SRC_ROOT"

echo "[2/6] 复制干净代码"
mkdir -p "$RELEASE_DIR/paper_clean_v28"
cp -r paper_clean_v28/*.py "$RELEASE_DIR/paper_clean_v28/"
cp -r paper_clean_v28/*.md "$RELEASE_DIR/paper_clean_v28/" 2>/dev/null || true
cp model_utils.py "$RELEASE_DIR/model_utils.py"

mkdir -p "$RELEASE_DIR/nmethyl/utils"
cp nmethyl/utils/nmethyl_config.py "$RELEASE_DIR/nmethyl/utils/nmethyl_config.py"
touch "$RELEASE_DIR/nmethyl/__init__.py"
touch "$RELEASE_DIR/nmethyl/utils/__init__.py"

echo "[3/6] 复制数据、模型和结果"
cp frankenstein_v28.pt "$RELEASE_DIR/frankenstein_v28.pt"
cp 17_complexes_native.jsonl "$RELEASE_DIR/17_complexes_native.jsonl"
mkdir -p "$RELEASE_DIR/nmethyl_data/test_set"
cp nmethyl_data/test_set/test.jsonl "$RELEASE_DIR/nmethyl_data/test_set/test.jsonl"
cp -r all_temperature_results "$RELEASE_DIR/all_temperature_results"

mkdir -p "$RELEASE_DIR/paper_clean_v28_outputs"
cp -r paper_clean_v28_outputs/monomer_clean "$RELEASE_DIR/paper_clean_v28_outputs/monomer_clean"
cp -r paper_clean_v28_outputs/complex_native_clean "$RELEASE_DIR/paper_clean_v28_outputs/complex_native_clean"
cp -r paper_clean_v28_outputs/generated_fasta_clean_auto_single "$RELEASE_DIR/paper_clean_v28_outputs/generated_fasta_clean_auto_single"
cp paper_clean_v28_outputs/native_chain_audit.csv "$RELEASE_DIR/paper_clean_v28_outputs/native_chain_audit.csv"
cp paper_clean_v28_outputs/af3_manifest.csv "$RELEASE_DIR/paper_clean_v28_outputs/af3_manifest.csv"
cp paper_clean_v28_outputs/structure_manifest_warnings.csv "$RELEASE_DIR/paper_clean_v28_outputs/structure_manifest_warnings.csv"

echo "[4/6] 写入 README 和复现实验脚本"
cat > "$RELEASE_DIR/README.md" <<'EOF'
# ProteinMPNN Clean V28 Evaluation Package

本仓库是 `frankenstein_v28.pt` 的干净评价和结构预测交接包。

## 目录说明

```text
paper_clean_v28/                         干净评价脚本
paper_clean_v28_outputs/monomer_clean/   单体测试集干净评价结果
paper_clean_v28_outputs/complex_native_clean/ 复合物天然短肽干净评价结果
paper_clean_v28_outputs/generated_fasta_clean_auto_single/ 生成 FASTA 干净评价结果
paper_clean_v28_outputs/af3_manifest.csv 给结构预测使用的 85 个任务清单
frankenstein_v28.pt                      最终模型
17_complexes_native.jsonl                17 个天然复合物数据
nmethyl_data/test_set/test.jsonl          单体测试集
all_temperature_results/                 已生成的 FASTA 序列
```

## 已确认结果

### 单体测试集

```text
真实评价位点：1505
基础氨基酸恢复率：16.08%
甲基化正样本数：323
已知序列甲基化 F1：80.14%
端到端甲基化 F1：60.91%
```

推荐论文主口径：`strict_naturalized_input` 下的 `known_sequence_methylation`。

### 复合物天然短肽

```text
真实短肽位点：251
基础氨基酸恢复率：21.12%
甲基化正样本数：0
```

复合物天然短肽没有甲基化正样本，因此不能报告甲基化召回率、F1 或 AUC，只能报告误报率或预测甲基化比例。

### 生成 FASTA

使用 `auto_single` 口径重新对齐：

```text
天然复合物目标数：17
原始生成序列数：4115
去重后生成序列数：4015
最佳设计条目数：85
警告数：0
```

`85 = 17 个目标 × 5 个温度`。

## 重新运行

```bash
bash run_reproduce_clean_eval.sh
```

## 结构预测

给结构预测使用的任务清单：

```text
paper_clean_v28_outputs/af3_manifest.csv
```

重要字段：

```text
design_peptide_seq           保留小写字母，表示 N-甲基化残基
design_peptide_natural_seq   将小写甲基化残基还原成普通天然氨基酸后的序列
design_methyl_count          设计序列中的甲基化位点数量
natural_aa_recovery          设计短肽相对天然短肽的天然氨基酸恢复率
```

如果结构预测平台不支持 N-甲基化残基，先用 `design_peptide_natural_seq` 预测结构，同时单独记录小写位点作为甲基化位点。
EOF

cat > "$RELEASE_DIR/run_reproduce_clean_eval.sh" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
export OMP_NUM_THREADS=1

python paper_clean_v28/01_eval_clean_model.py \
  --model_path ./frankenstein_v28.pt \
  --data_jsonl nmethyl_data/test_set/test.jsonl \
  --mode monomer \
  --eval_chains masked \
  --batch_size 16 \
  --out_dir paper_clean_v28_outputs/monomer_clean

python paper_clean_v28/01_eval_clean_model.py \
  --model_path ./frankenstein_v28.pt \
  --data_jsonl 17_complexes_native.jsonl \
  --mode complex \
  --eval_chains short \
  --max_peptide_len 30 \
  --batch_size 1 \
  --out_dir paper_clean_v28_outputs/complex_native_clean

python paper_clean_v28/02_score_generated_fastas.py \
  --native_jsonl 17_complexes_native.jsonl \
  --fasta_dir all_temperature_results \
  --out_dir paper_clean_v28_outputs/generated_fasta_clean_auto_single \
  --eval_chains auto_single \
  --max_peptide_len 30

python paper_clean_v28/03_prepare_structure_manifest.py \
  --best_csv paper_clean_v28_outputs/generated_fasta_clean_auto_single/best_designs.csv \
  --native_jsonl 17_complexes_native.jsonl \
  --out_csv paper_clean_v28_outputs/af3_manifest.csv
EOF
chmod +x "$RELEASE_DIR/run_reproduce_clean_eval.sh"

cat > "$RELEASE_DIR/requirements_minimal.txt" <<'EOF'
numpy
torch
EOF

cat > "$RELEASE_DIR/.gitignore" <<'EOF'
__pycache__/
*.pyc
.ipynb_checkpoints/
.DS_Store
*.log
EOF

cat > "$RELEASE_DIR/CHECK_FILE_SIZES.sh" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
find . -type f -size +50M -print0 | xargs -0 -r du -h
EOF
chmod +x "$RELEASE_DIR/CHECK_FILE_SIZES.sh"

echo "[5/6] 初始化 Git 仓库"
cd "$RELEASE_DIR"
if git init -b main >/dev/null 2>&1; then
  echo "已初始化 main 分支。"
else
  git init
  git symbolic-ref HEAD refs/heads/main
  echo "已设置 HEAD 到 main 分支。"
fi

echo "[6/6] 显示大文件和仓库状态"
echo "大于 50M 的文件："
./CHECK_FILE_SIZES.sh || true

git add .
git status --short

cat <<EOF

完成：$RELEASE_DIR

下一步：
1. 在 GitHub 网页新建一个空仓库，建议私有仓库，例如：proteinmpnn-clean-v28
2. 如果没有大于 100M 的文件：
   cd $RELEASE_DIR
   git commit -m "Initial clean V28 evaluation package"
   git remote add origin git@github.com:DCarchimonde/proteinmpnn-clean-v28.git
   git push -u origin main

3. 如果 frankenstein_v28.pt 或其他文件大于 100M，请先启用 Git LFS：
   git lfs install
   git lfs track "*.pt"
   git add .gitattributes frankenstein_v28.pt
   git commit -m "Initial clean V28 evaluation package with LFS"
   git remote add origin git@github.com:DCarchimonde/proteinmpnn-clean-v28.git
   git push -u origin main

EOF

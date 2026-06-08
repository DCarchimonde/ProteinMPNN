# paper_clean_v28

这是给 `frankenstein_v28.pt` 单独建立的干净评价工作区。这个文件夹不改动原来的训练、生成、评估脚本，后续论文用数据优先从这里重新产出。

## 目标

1. 统一字母表，只使用 `nmethyl/utils/nmethyl_config.py`。
2. 统一模型结构，只针对最终模型 `frankenstein_v28.pt`。
3. 严格排除填充位点。
4. 单体和复合物分开评价。
5. 明确区分：
   - 氨基酸恢复率。
   - 已知序列条件下的甲基化位点预测。
   - 端到端设计加甲基化预测。
6. 为后续结构预测和论文作图输出干净表格。

## 推荐目录

在 `ProteinMPNN-main` 根目录下运行：

```bash
mkdir -p paper_clean_v28_outputs
```

建议把所有新结果都放在：

```text
paper_clean_v28_outputs/
```

不要再和旧的 `run_v28_robust`、`generated_peptides`、`temperature_sequence_metrics` 混在一起。

## 第一步：单体甲基化与序列评价

```bash
python paper_clean_v28/01_eval_clean_model.py \
  --model_path ./frankenstein_v28.pt \
  --data_jsonl nmethyl_data/test_set/test.jsonl \
  --mode monomer \
  --eval_chains masked \
  --batch_size 16 \
  --out_dir paper_clean_v28_outputs/monomer_clean
```

重点看输出：

```text
summary.json
position_predictions.csv
threshold_metrics.csv
```

论文主口径优先看 `strict_naturalized_known_sequence`。

## 第二步：复合物短肽评价

```bash
python paper_clean_v28/01_eval_clean_model.py \
  --model_path ./frankenstein_v28.pt \
  --data_jsonl 17_complexes_native.jsonl \
  --mode complex \
  --eval_chains short \
  --max_peptide_len 30 \
  --batch_size 1 \
  --out_dir paper_clean_v28_outputs/complex_native_clean
```

注意：如果复合物天然短肽里没有甲基化正样本，只能看误报率，不能报告召回率或 F1。

## 第三步：评价已经生成的 FASTA

```bash
python paper_clean_v28/02_score_generated_fastas.py \
  --native_jsonl 17_complexes_native.jsonl \
  --fasta_dir all_temperature_results \
  --out_dir paper_clean_v28_outputs/generated_fasta_clean \
  --eval_chains short \
  --max_peptide_len 30
```

这个脚本用于评价生成序列本身，不需要模型权重。

## 第四步：准备结构预测清单

```bash
python paper_clean_v28/03_prepare_structure_manifest.py \
  --best_csv paper_clean_v28_outputs/generated_fasta_clean/best_designs.csv \
  --native_jsonl 17_complexes_native.jsonl \
  --out_csv paper_clean_v28_outputs/af3_manifest.csv
```

这个表用于整理哪些序列需要拿去预测结构。预测结构回来后，再写结构指标脚本。

## 当前结论边界

- 旧的 38% 氨基酸恢复率不能作为干净主结果。
- 训练阶段是否需要重训，需要用这个文件夹的干净评价结果和后续修正版训练结果对照。
- `frankenstein_v28.pt` 先作为当前最终模型进行完整干净评价。
- 旧结构先不要删，必须先做序列和结构对应关系检查，能对上的才继续使用。

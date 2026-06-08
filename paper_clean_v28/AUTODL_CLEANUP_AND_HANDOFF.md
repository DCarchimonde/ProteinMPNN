# AutoDL 清理、结果整理和师兄交接说明

本文件记录 `frankenstein_v28.pt` 的干净评价工作流。

## 1. 不建议删除旧目录

旧目录先不要删除。原因：

1. 旧代码可以作为审计证据，说明为什么重新评价。
2. 旧生成结果还需要追溯来源。
3. 旧结构文件可能部分能复用，但必须先做序列和链对应检查。

建议做法：不删除旧文件，只把论文后续工作全部集中到下面两个目录。

## 2. GitHub 代码主目录

只维护代码：

```text
paper_clean_v28/
```

这里放干净脚本，不放大数据、不放模型、不放输出结果。

当前核心脚本：

```text
paper_clean_v28/01_eval_clean_model.py
paper_clean_v28/02_score_generated_fastas.py
paper_clean_v28/03_prepare_structure_manifest.py
paper_clean_v28/04_audit_native_chains.py
paper_clean_v28/clean_v28_common.py
```

## 3. AutoDL 输出主目录

只认这个输出目录：

```text
paper_clean_v28_outputs/
```

推荐保留以下子目录和文件：

```text
paper_clean_v28_outputs/monomer_clean/
paper_clean_v28_outputs/complex_native_clean/
paper_clean_v28_outputs/generated_fasta_clean_auto_single/
paper_clean_v28_outputs/native_chain_audit.csv
paper_clean_v28_outputs/af3_manifest.csv
paper_clean_v28_outputs/structure_manifest_warnings.csv
```

下面两个目录只是中间错误口径调试结果，不建议用于论文：

```text
paper_clean_v28_outputs/generated_fasta_clean/
paper_clean_v28_outputs/generated_fasta_clean_masked/
```

可以移动到归档目录，不建议直接删除。

## 4. AutoDL 本地整理命令

在 `~/ProteinMPNN-main` 下运行：

```bash
mkdir -p paper_clean_v28_outputs/_archive_wrong_chain_modes
mv paper_clean_v28_outputs/generated_fasta_clean paper_clean_v28_outputs/_archive_wrong_chain_modes/ 2>/dev/null || true
mv paper_clean_v28_outputs/generated_fasta_clean_masked paper_clean_v28_outputs/_archive_wrong_chain_modes/ 2>/dev/null || true
```

整理后，论文主结果只看：

```text
paper_clean_v28_outputs/monomer_clean/
paper_clean_v28_outputs/complex_native_clean/
paper_clean_v28_outputs/generated_fasta_clean_auto_single/
paper_clean_v28_outputs/af3_manifest.csv
```

## 5. 当前已确认结果

### 5.1 单体测试集

干净评价位点：1505。

推荐论文主口径：`strict_naturalized_input` 下的 `known_sequence_methylation`。

结果：

```text
基础氨基酸恢复率：16.08%
甲基化正样本数：323
已知序列甲基化 F1：80.14%
端到端甲基化 F1：60.91%
```

### 5.2 复合物天然短肽

干净评价位点：251。

结果：

```text
基础氨基酸恢复率：21.12%
甲基化正样本数：0
```

注意：复合物天然短肽没有甲基化正样本，所以不能报告甲基化召回率、F1 或 AUC，只能报告误报率或预测甲基化比例。

### 5.3 生成 FASTA

最终使用目录：

```text
paper_clean_v28_outputs/generated_fasta_clean_auto_single/
```

结果：

```text
天然复合物目标数：17
原始生成序列数：4115
去重后生成序列数：4015
最佳设计条目数：85
警告数：0
```

说明：85 = 17 个目标 × 5 个温度。

## 6. 为什么使用 auto_single 口径

native jsonl 里 `masked_list` 为空，短肽链在 `visible_list` 中。

很多复合物有两条短肽链，例如 X/Y、D/F、G/H，但是两条短肽链序列完全相同。生成 FASTA 里只有一条短肽序列，所以应该按设计序列长度自动匹配单条短肽链，而不是把两条短肽链拼起来。

因此论文序列评价必须使用：

```bash
python paper_clean_v28/02_score_generated_fastas.py \
  --native_jsonl 17_complexes_native.jsonl \
  --fasta_dir all_temperature_results \
  --out_dir paper_clean_v28_outputs/generated_fasta_clean_auto_single \
  --eval_chains auto_single \
  --max_peptide_len 30
```

不要使用：

```text
--eval_chains short
--eval_chains masked
```

## 7. 给师兄的结构预测清单

文件：

```text
paper_clean_v28_outputs/af3_manifest.csv
```

该文件包含 85 个结构预测任务，每个任务对应一个目标和一个温度下的最佳设计短肽。

重要说明：

1. `design_peptide_seq` 保留小写字母，表示 N-甲基化残基。
2. `design_peptide_natural_seq` 是把小写甲基化残基还原成天然氨基酸后的序列。
3. 如果 AlphaFold 3 或其他结构预测平台不支持 N-甲基化残基，需要使用 `design_peptide_natural_seq` 预测结构，同时额外记录甲基化位点。
4. 如果平台支持修饰残基，则优先使用 `design_peptide_seq` 和甲基化位点信息。

## 8. 给师兄的信息模板

```text
师兄，我现在已经把旧代码结果重新整理成一个干净评价流程了。

最终模型先固定为 frankenstein_v28.pt。旧代码里发现主要问题是：训练/评价里填充位点曾被当作真实残基，另外复合物短肽评价时不能把两条相同短肽链拼接起来和单条生成序列比较。

我现在重新建立了 paper_clean_v28 工作区，所有新评价都放在 paper_clean_v28_outputs 下面。

目前已确认：
1. 单体测试集干净评价位点 1505，strict naturalized 输入下，已知序列甲基化预测 F1 为 80.14%。
2. 复合物天然短肽干净评价位点 251，但没有甲基化正样本，所以不能报告复合物甲基化 F1，只能看误报率。
3. 生成 FASTA 已用 auto_single 口径重新对齐，17 个复合物 × 5 个温度，共 85 个最佳设计条目，警告数为 0。

我已经生成结构预测清单：paper_clean_v28_outputs/af3_manifest.csv。
这个表里每一行是一个结构预测任务，包含 target、温度、native peptide、design peptide、天然化后的 design peptide、甲基化数量和恢复率。

想请您帮我确认后续结构预测怎么处理 N-甲基化残基：
- 如果 AF3 输入支持修饰残基，我想保留小写甲基化位点信息来建模；
- 如果不支持，我先用 design_peptide_natural_seq 预测结构，同时单独记录甲基化位点。

结构回来后，我会继续算 RMSD_CA、Backbone RMSD、All-atom RMSD、pLDDT、ipLDDT、pTM、ipTM、PAE、Designability 和结构多样性。
```

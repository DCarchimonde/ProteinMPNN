import argparse
import os
import sys
import torch
import torch.nn.functional as F
import json
import numpy as np
from torch.utils.data import DataLoader

# 确保能导入项目根目录的模块
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    # 从 v12 导入模型和数据处理函数
    from nmethyl.train_final_v12_polishing import DecoupledProteinMPNN, featurize_batch, JSONLDataset, collate_fn
    from nmethyl.utils.nmethyl_config import EXTENDED_AA_ALPHABET, NATURAL_AA_ALPHABET, NMETHYL_TO_NATURAL_MAPPING
except ImportError as e:
    print(f"Error: {e}")
    sys.exit(1)

def sample_sequences():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default="./final_run_v12_soup/best_model_soup.pt")
    parser.add_argument("--test_data", type=str, required=True)
    parser.add_argument("--output_fasta", type=str, default="final_designs.fasta")
    parser.add_argument("--num_seqs", type=int, default=2, help="Number of sequences per backbone")
    parser.add_argument("--temp", type=float, default=0.1, help="Sampling temperature (lower = more confident)")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"Loading model from {args.checkpoint}...")
    # v12 模型初始化需要 augment_eps 参数，推理时设为 0 关闭噪声
    model = DecoupledProteinMPNN(augment_eps=0.0).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device)['model_state_dict'])
    model.eval()

    test_ds = JSONLDataset(args.test_data)
    # 这里的 batch_size 设为 1 是为了方便逐个蛋白生成多条序列
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False, collate_fn=collate_fn)

    print(f"Generating {args.num_seqs} sequences per target...")
    
    with open(args.output_fasta, 'w') as f_out:
        with torch.no_grad():
            for i, batch in enumerate(test_loader):
                # 获取蛋白名称
                item = batch[0]
                name = item.get('name', item.get('pdb_id', f'target_{i}'))
                
                # 复制 batch 以生成多条序列 (Batch Expansion)
                batch_expanded = batch * args.num_seqs
                
                f = featurize_batch(batch_expanded, device)
                if f is None: continue
                
                # [修复点] 正确解包 6 个返回值，与 v12 保持一致
                X, S, mask, chain_M, residue_idx, chain_encoding_all = f[0], f[1], f[2], f[3], f[4], f[5]

                # Forward
                logits_base, logits_methyl = model(X, S, mask, chain_M, residue_idx, chain_encoding_all)
                
                # Sampling
                probs_base = F.softmax(logits_base / args.temp, dim=-1)
                probs_methyl = F.softmax(logits_methyl / args.temp, dim=-1)
                
                # 从分布中采样
                sample_base = torch.multinomial(probs_base.view(-1, probs_base.size(-1)), 1).view(probs_base.size(0), -1)
                sample_methyl = torch.multinomial(probs_methyl.view(-1, probs_methyl.size(-1)), 1).view(probs_methyl.size(0), -1)
                
                # 解码为字符串
                for j in range(args.num_seqs):
                    seq_str = ""
                    L = int(mask[j].sum().item())
                    
                    for k in range(L):
                        base_idx = sample_base[j, k].item()
                        is_me = sample_methyl[j, k].item()
                        
                        # 1. 先获取基础天然氨基酸字符
                        if base_idx < len(NATURAL_AA_ALPHABET):
                             aa_char = NATURAL_AA_ALPHABET[base_idx]
                        else:
                             aa_char = 'X' # 理论上不应该发生
                        
                        # 2. 如果预测为甲基化，尝试转换为对应的小写字符
                        if is_me == 1:
                            # 查找映射：Natural Index -> Methyl Relative Index
                            for m_rel, n_idx in NMETHYL_TO_NATURAL_MAPPING.items():
                                if n_idx == base_idx:
                                    # 找到了！计算在 Extended 表中的绝对位置
                                    # Extended = [Natural(20)] + [Methyl(14)] + [X]
                                    m_abs = len(NATURAL_AA_ALPHABET) + m_rel
                                    if m_abs < len(EXTENDED_AA_ALPHABET):
                                        aa_char = EXTENDED_AA_ALPHABET[m_abs]
                                    break
                        
                        seq_str += aa_char
                    
                    # 写入 FASTA 文件
                    f_out.write(f">{name}_design_{j}\n{seq_str}\n")
                    
    print(f"Done. Sequences saved to {args.output_fasta}")

if __name__ == "__main__":
    sample_sequences()
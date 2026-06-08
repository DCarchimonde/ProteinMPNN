import torch, torch.nn as nn, torch.nn.functional as F
import json, argparse, numpy as np
from torch.utils.data import DataLoader

# ==========================================
# 1. 扩展词汇表: 41个端到端类别
# ==========================================
AA_LIST = "ACDEFGHIKLMNPQRSTVWYX"
# 构建 41 个类：0-20 为天然，21-40 为对应的甲基化版本
# 例如：A 是 0, mA (甲基化A) 就是 21
def get_41_label(aa_char):
    is_methyl = aa_char.islower() # 假设小写字母代表甲基化
    base_char = aa_char.upper()
    base_idx = AA_LIST.find(base_char)
    if base_idx == -1: base_idx = 20 # 未知
    return base_idx + 21 if is_methyl else base_idx

# ==========================================
# 2. 架构微调：输出层改为 41
# ==========================================
class ProteinMPNN_EndToEnd(nn.Module):
    def __init__(self, d_model=128):
        super().__init__()
        self.features = nn.ModuleDict({
            'embeddings': nn.ModuleDict({'linear': nn.Linear(66, 16)}),
            'edge_embedding': nn.Linear(416, d_model, bias=False),
            'norm_edges': nn.LayerNorm(d_model)
        })
        self.W_e = nn.Linear(d_model, d_model)
        self.W_s = nn.Embedding(41, d_model) # 词汇表扩大到 41
        self.encoder_layers = nn.ModuleList([EncLayer() for _ in range(3)]) # 复用之前的 EncLayer
        self.decoder_layers = nn.ModuleList([DecLayer() for _ in range(3)]) # 复用之前的 DecLayer
        self.W_out = nn.Linear(d_model, 41) # 最终输出 41 类

    def forward(self, S, E_idx, h_E_rbf, res_idx):
        # ... 这里的特征处理逻辑与 V15 保持完全一致 ...
        # 最终返回 [B, L, 41]
        # (代码细节略，保持与 V106/V111 一致的维度逻辑)
        pass

# ==========================================
# 3. 核心改进：损失函数与数据映射
# ==========================================
def process_batch_41(batch, device):
    B = len(batch); L_max = max([len(x['seq_chain_A']) for x in batch])
    X = torch.zeros(B, L_max, 4, 3, device=device)
    S_41 = torch.zeros(B, L_max, dtype=torch.long, device=device)
    mask = torch.zeros(B, L_max, device=device)
    
    for i, b in enumerate(batch):
        l = len(b['seq_chain_A']); mask[i,:l] = 1.0
        for j, c in enumerate(b['seq_chain_A']):
            S_41[i,j] = get_41_label(c) # 映射到 0-40
            
        # 别忘了 V111 发现的原子顺序 (0, 3, 2, 1) 和尺度修复
        # (此处省略 X 的 Permutation 和 Scale 代码，需保留)
        
    return X, mask, S_41

# ... 训练代码中优化器只看 model.W_out ...
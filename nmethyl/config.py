"""
Configuration for N-methylation extension
"""

# 扩展的氨基酸字母表
EXTENDED_AA_ALPHABET = 'ACDEFGHIKLMNPQRSTVWYXabcdefghijklmnopqrs'

# N-甲基化氨基酸映射
NMETHYL_MAPPING = {
    # 天然氨基酸
    'ALA': 'A', 'ARG': 'R', 'ASN': 'N', 'ASP': 'D', 'CYS': 'C',
    'GLN': 'Q', 'GLU': 'E', 'GLY': 'G', 'HIS': 'H', 'ILE': 'I',
    'LEU': 'L', 'LYS': 'K', 'MET': 'M', 'PHE': 'F', 'PRO': 'P',
    'SER': 'S', 'THR': 'T', 'TRP': 'W', 'TYR': 'Y', 'VAL': 'V',
    # N-甲基化氨基酸
    'NCY': 'c', 'SOQ': 'd', 'EME': 'e',
    'E9V': 'h', 'IML': 'i', 'MEA': 'f',
    'MLE': 'l', 'MME': 'm', '5JP': 's',
    'GNC': 'q', 'MMO': 'r', 'MVA': 'v',
    'YNM': 'y', 'NZC': 't',
}

# 训练配置
DEFAULT_TRAIN_CONFIG = {
    'num_epochs': 50,
    'batch_size': 32,
    'learning_rate': 1e-4,
    'backbone_lr_multiplier': 0.1,  # 预训练骨干网络的学习率倍数
}
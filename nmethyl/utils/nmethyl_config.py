"""
Configuration for N-methylation model extension.
This file serves as the single source of truth for all amino acid definitions and mappings.
Based on Data Verification: 2025-01-05
"""

# =============================================================================
# 1. 残基定义 (Residue Definitions)
# =============================================================================

# 标准天然氨基酸：三字母 -> 单字母 (大写)
NATURAL_RESIDUE_MAP = {
    'ALA': 'A', 'ARG': 'R', 'ASN': 'N', 'ASP': 'D', 'CYS': 'C',
    'GLN': 'Q', 'GLU': 'E', 'GLY': 'G', 'HIS': 'H', 'ILE': 'I',
    'LEU': 'L', 'LYS': 'K', 'MET': 'M', 'PHE': 'F', 'PRO': 'P',
    'SER': 'S', 'THR': 'T', 'TRP': 'W', 'TYR': 'Y', 'VAL': 'V',
}

# N-甲基化氨基酸：根据 PDB 原子分析结果修正后的 20 种映射
# 键：HETATM 代码 | 值：对应天然氨基酸的单字母 (小写)
NMETHYL_RESIDUE_MAP = {
    # --- 疏水性 / 脂肪族 ---
    'MAA': 'a', # Ala (数据验证: 仅含CB)
    'SAR': 'g', # Gly (数据验证: 无侧链)
    'MLE': 'l', # Leu
    'IML': 'i', # Ile
    'MVA': 'v', # Val
    'MME': 'm', # Met
    'MEA': 'f', # Phe
    'YNM': 'y', # Tyr
    'E9M': 'w', # Trp (数据验证: 含吲哚环)

    # --- 极性 / 电荷 ---
    '5JP': 's', # Ser
    'SER': 's', # Ser (HETATM version)
    'NZC': 't', # Thr
    'NCY': 'c', # Cys
    'ZCA': 'n', # Asn (数据验证: 含OD1, ND2)
    'GNC': 'q', # Gln (数据验证: 含OE1, NE2)
    'SOQ': 'd', # Asp
    'EME': 'e', # Glu
    'NMK': 'k', # Lys
    'MMO': 'r', # Arg
    'E9V': 'h', # His
}

# 将所有映射合并为一个总的映射表
ALL_RESIDUE_MAP = {**NATURAL_RESIDUE_MAP, **NMETHYL_RESIDUE_MAP}


# =============================================================================
# 2. 字母表定义 (Alphabet Definitions)
# =============================================================================

# 20种标准天然氨基酸的字母表
NATURAL_AA_ALPHABET = "ACDEFGHIKLMNPQRSTVWY"

# 自动生成N-甲基化氨基酸的字母表 (去重并排序)
# 修正后应该包含: a,c,d,e,f,g,h,i,k,l,m,n,q,r,s,t,v,w,y (共19种, 因为5JP和SER都映射为s)
METHYL_AA_ALPHABET = "".join(sorted(list(set(NMETHYL_RESIDUE_MAP.values()))))

# 构建完整的扩展字母表
EXTENDED_AA_ALPHABET = NATURAL_AA_ALPHABET + METHYL_AA_ALPHABET + "X"


# =============================================================================
# 3. 索引映射 (Index Mappings)
# =============================================================================

# 创建一个从字母到索引的字典
_alphabet_to_index = {aa: i for i, aa in enumerate(EXTENDED_AA_ALPHABET)}

# N-甲基化到天然氨基酸的索引映射 (用于权重初始化)
NMETHYL_TO_NATURAL_MAPPING = {
    idx_methyl: _alphabet_to_index[methyl_char.upper()]
    for idx_methyl, methyl_char in enumerate(METHYL_AA_ALPHABET)
}

# 扩展字母表到索引的完整映射
EXTENDED_AA_TO_INDEX = _alphabet_to_index

# N-甲基化索引 -> 天然索引 的反向查找字典 (用于Loss计算)
NATURAL_IDX_MAP = {}
for nmethyl_idx_relative, natural_idx in NMETHYL_TO_NATURAL_MAPPING.items():
    nmethyl_actual_idx = len(NATURAL_AA_ALPHABET) + nmethyl_idx_relative
    NATURAL_IDX_MAP[nmethyl_actual_idx] = natural_idx


# =============================================================================
# 4. 训练配置
# =============================================================================

DEFAULT_TRAIN_CONFIG = {
    'num_epochs': 150,
    'batch_size': 8,
    'learning_rate': 1e-4,
    'backbone_lr_multiplier': 0.1,
    'save_interval': 10,
    'early_stopping_patience': 20,
}

if __name__ == '__main__':
    print("--- N-甲基化模型配置核对 ---")
    print(f"检测到 {len(NATURAL_AA_ALPHABET)} 种天然氨基酸。")
    print(f"检测到 {len(METHYL_AA_ALPHABET)} 种N-甲基化类别 (小写): {METHYL_AA_ALPHABET}")
    print(f"完整扩展字母表 (共 {len(EXTENDED_AA_ALPHABET)} 个字符):")
    print(f"  {EXTENDED_AA_ALPHABET}")
    print("\n映射关系验证 (HETATM -> Token -> Parent):")
    for het, token in sorted(NMETHYL_RESIDUE_MAP.items()):
        print(f"  {het} -> '{token}' -> {token.upper()}")
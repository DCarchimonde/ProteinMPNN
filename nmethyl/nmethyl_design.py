import torch
import torch.nn.functional as F
import numpy as np
import os
import sys

# 添加父目录到路径以便导入ProteinMPNN模块
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model_utils import ExtendedProteinMPNN, extended_featurize
from .utils.nmethyl_config import EXTENDED_AA_ALPHABET, NMETHYL_REVERSE_MAPPING

class NmethylDesigner:
    def __init__(self, model_path):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.load_model(model_path)
        self.extended_alphabet = EXTENDED_AA_ALPHABET
        print(f"N-methylation designer loaded on {self.device}")
    
    def load_model(self, model_path):
        """加载训练好的N-甲基化模型"""
        model = ExtendedProteinMPNN(
            num_letters=40,
            node_features=128,
            edge_features=128,
            hidden_dim=128,
            num_encoder_layers=3,
            num_decoder_layers=3,
            vocab=40
        ).to(self.device)
        
        checkpoint = torch.load(model_path, map_location=self.device)
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        
        model.eval()
        return model
    
    def design_sequence(self, backbone_coords, chain_id='A', temperature=0.1, num_samples=5):
        """为给定骨架设计包含N-甲基化氨基酸的序列"""
        
        # 准备输入数据
        batch = [{
            'coords_chain_A': backbone_coords,
            'seq_chain_A': 'X' * len(backbone_coords),  # 占位序列
            'masked_list': ['A'],
            'visible_list': [],
            'num_of_chains': 1
        }]
        
        with torch.no_grad():
            X, S, mask, lengths, chain_M, residue_idx, mask_self, chain_encoding_all = \
                extended_featurize(batch, self.device)
            
            log_probs = self.model(X, S, mask, chain_M, residue_idx, chain_encoding_all)
            
            # 采样序列
            designed_sequences = []
            sequence_scores = []
            
            for sample_idx in range(num_samples):
                # 应用温度采样
                probs = torch.softmax(log_probs / temperature, dim=-1)
                sampled_seq = torch.multinomial(probs.view(-1, 40), 1).view(S.shape)
                
                # 计算序列分数
                seq_probs = torch.gather(probs, -1, sampled_seq.unsqueeze(-1)).squeeze(-1)
                mask_for_scoring = mask * chain_M
                sequence_score = torch.sum(torch.log(seq_probs + 1e-10) * mask_for_scoring) / torch.sum(mask_for_scoring)
                
                # 转换为氨基酸序列
                seq_str = ''.join([self.extended_alphabet[idx] for idx in sampled_seq[0].cpu().numpy()])
                
                designed_sequences.append(seq_str)
                sequence_scores.append(sequence_score.item())
        
        return designed_sequences, sequence_scores
    
    def get_sequence_probabilities(self, backbone_coords, sequence):
        """获取给定序列在骨架上的概率分布"""
        batch = [{
            'coords_chain_A': backbone_coords,
            'seq_chain_A': sequence,
            'masked_list': [],
            'visible_list': ['A'],
            'num_of_chains': 1
        }]
        
        with torch.no_grad():
            X, S, mask, lengths, chain_M, residue_idx, mask_self, chain_encoding_all = \
                extended_featurize(batch, self.device)
            
            log_probs = self.model(X, S, mask, chain_M, residue_idx, chain_encoding_all)
            probs = torch.exp(log_probs)
            
            return probs.cpu().numpy()
    
    def convert_to_pdb_names(self, sequence):
        """将扩展字母表序列转换为PDB残基名称"""
        pdb_sequence = []
        for aa in sequence:
            if aa in NMETHYL_REVERSE_MAPPING:
                pdb_sequence.append(NMETHYL_REVERSE_MAPPING[aa])
            elif aa in self.extended_alphabet[:21]:  # 天然氨基酸
                # 这里需要天然氨基酸到三字母码的映射
                natural_mapping = {
                    'A': 'ALA', 'C': 'CYS', 'D': 'ASP', 'E': 'GLU', 'F': 'PHE',
                    'G': 'GLY', 'H': 'HIS', 'I': 'ILE', 'K': 'LYS', 'L': 'LEU',
                    'M': 'MET', 'N': 'ASN', 'P': 'PRO', 'Q': 'GLN', 'R': 'ARG',
                    'S': 'SER', 'T': 'THR', 'V': 'VAL', 'W': 'TRP', 'Y': 'TYR'
                }
                pdb_sequence.append(natural_mapping.get(aa, 'UNK'))
            else:
                pdb_sequence.append('UNK')
        
        return pdb_sequence
    
    def analyze_sequence(self, sequence):
        """分析序列中的N-甲基化氨基酸含量"""
        total_residues = len(sequence)
        nmethyl_residues = sum(1 for aa in sequence if aa in 'abcdefghijklmnopqrs')
        natural_residues = total_residues - nmethyl_residues
        
        return {
            'total_residues': total_residues,
            'natural_residues': natural_residues,
            'nmethyl_residues': nmethyl_residues,
            'nmethyl_percentage': (nmethyl_residues / total_residues) * 100
        }

# 使用示例
def example_usage():
    """使用示例"""
    # 初始化设计器
    designer = NmethylDesigner("extended_model_weights/nmethyl_mpnn_final.pt")
    
    # 示例骨架坐标（需要实际数据）
    # backbone_coords = [...]  # 实际的骨架坐标
    
    # 设计序列
    # sequences, scores = designer.design_sequence(backbone_coords, num_samples=3)
    
    # 输出结果
    # for i, (seq, score) in enumerate(zip(sequences, scores)):
    #     analysis = designer.analyze_sequence(seq)
    #     print(f"Sample {i+1}: {seq}")
    #     print(f"Score: {score:.4f}, N-methyl residues: {analysis['nmethyl_residues']}")
    #     print(f"PDB names: {designer.convert_to_pdb_names(seq)}")
    #     print()
    
    print("Example usage - replace with actual backbone coordinates")

if __name__ == "__main__":
    example_usage()
import argparse
import os
import sys
import torch
import json
import numpy as np
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import classification_report

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from model_utils import ProteinMPNN
    from nmethyl.utils.nmethyl_config import EXTENDED_AA_ALPHABET, NMETHYL_TO_NATURAL_MAPPING, EXTENDED_AA_TO_INDEX, NATURAL_AA_ALPHABET
    # 复用 v11 的数据处理函数 (注意：v11 的 featurize_batch 现在返回 6 个元素)
    from nmethyl.train_final_v11_distillation import featurize_batch, JSONLDataset, collate_fn
except ImportError as e:
    print(f"Error: {e}")
    sys.exit(1)

def evaluate_teacher():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrained_weights", type=str, required=True)
    parser.add_argument("--test_data", type=str, required=True)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(">>> Loading Teacher (Standard ProteinMPNN)...")
    # Teacher: k=48, vocab=21
    teacher_model = ProteinMPNN(num_letters=21, vocab=21, k_neighbors=48).to(device)
    
    # 加载权重 (Standard)
    ckpt = torch.load(args.pretrained_weights, map_location=device)
    state_dict = ckpt.get('model_state_dict', ckpt)
    clean_state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    teacher_model.load_state_dict(clean_state_dict, strict=True)
    teacher_model.eval()

    print(f">>> Evaluating on {args.test_data}...")
    test_ds = JSONLDataset(args.test_data)
    test_loader = DataLoader(test_ds, batch_size=8, shuffle=False, collate_fn=collate_fn)

    all_preds, all_targets = [], []

    with torch.no_grad():
        for batch in test_loader:
            f = featurize_batch(batch, device)
            if f is None: continue
            
            # [修复] 直接解包 6 个元素，与 v11 保持一致
            # 返回顺序: [X, S, mask, chain_M, residue_idx, chain_encoding_all]
            X, S, mask, chain_M, residue_idx, chain_encoding_all = f

            # 构造 Teacher 的 Input S (将 N-Methyl 映射回 Natural)
            f_teacher_S = S.clone()
            offset = len(NATURAL_AA_ALPHABET)
            for m_rel, n_idx in NMETHYL_TO_NATURAL_MAPPING.items():
                f_teacher_S[f_teacher_S == (m_rel + offset)] = n_idx
            f_teacher_S[f_teacher_S >= 21] = 20 # Others -> X

            # Forward
            logits = teacher_model(X, f_teacher_S, mask, chain_M, residue_idx, chain_encoding_all)
            probs = torch.softmax(logits, dim=-1) # [B, L, 21]
            preds = torch.argmax(probs, dim=-1)   # [B, L]

            # 收集结果
            preds_flat = preds.cpu().numpy().flatten()
            targets_flat = S.cpu().numpy().flatten()
            
            # 映射 Target: N-methyl -> Natural
            targets_natural_mapped = targets_flat.copy()
            for m_rel, n_idx in NMETHYL_TO_NATURAL_MAPPING.items():
                targets_natural_mapped[targets_flat == (m_rel + offset)] = n_idx
            
            mask_flat = mask.cpu().numpy().flatten().astype(bool)
            x_idx = EXTENDED_AA_TO_INDEX['X']
            valid = mask_flat & (targets_flat != x_idx)

            all_preds.extend(preds_flat[valid])
            all_targets.extend(targets_natural_mapped[valid])

    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)
    
    if len(all_targets) == 0:
        print("No valid targets found.")
        return

    acc = np.mean(all_preds == all_targets)
    print("\n" + "="*40)
    print(f"TEACHER BASELINE ACCURACY: {acc:.4f} ({acc*100:.2f}%)")
    print("="*40 + "\n")
    
    print("解读指南:")
    print("1. 如果 Teacher Acc ≈ 30%：说明测试数据本身很难，您的 Student 模型 (30.5%) 已经达到了老师的水平，是完美的。")
    print("2. 如果 Teacher Acc ≈ 50%：说明 Student 模型还有提升空间 (可能是蒸馏温度或权重需要调整)。")

if __name__ == "__main__":
    evaluate_teacher()
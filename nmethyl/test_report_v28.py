import argparse
import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import json
import numpy as np
from torch.utils.data import DataLoader, Dataset
import torch.utils.checkpoint

# --- 系统路径 ---
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from model_utils import ProteinMPNN, gather_nodes, cat_neighbors_nodes
    from nmethyl.utils.nmethyl_config import (
        EXTENDED_AA_ALPHABET, NATURAL_AA_ALPHABET,
        NMETHYL_TO_NATURAL_MAPPING, EXTENDED_AA_TO_INDEX, METHYL_AA_ALPHABET
    )
except ImportError as e:
    print(f"Error: Missing project files. {e}")
    sys.exit(1)

# =============================================================================
# 1. 模型定义 (必须与训练时的 V28 完全一致)
# =============================================================================
class RobustHierarchicalProteinMPNN(ProteinMPNN):
    def __init__(self, hidden_dim=128, augment_eps=0.1, **kwargs):
        super().__init__(
            num_letters=21, 
            hidden_dim=hidden_dim, 
            vocab=21, 
            k_neighbors=48, 
            augment_eps=augment_eps, 
            **kwargs
        )
        self.W_s = nn.Embedding(len(EXTENDED_AA_ALPHABET), hidden_dim)
        
        # V28: MLP Head + LayerNorm
        self.W_out_base = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, len(NATURAL_AA_ALPHABET))
        )
        
        self.experts = nn.ModuleList([
            nn.Linear(hidden_dim, 1) for _ in range(len(NATURAL_AA_ALPHABET))
        ])

    def forward(self, X, S, mask, chain_M, residue_idx, chain_encoding_all):
        E, E_idx = self.features(X, mask, residue_idx, chain_encoding_all)
        h_V = torch.zeros((E.shape[0], E.shape[1], E.shape[-1]), device=E.device)
        h_E = self.W_e(E)

        mask_attend = gather_nodes(mask.unsqueeze(-1),  E_idx).squeeze(-1)
        mask_attend = mask.unsqueeze(-1) * mask_attend
        for layer in self.encoder_layers:
            h_V, h_E = torch.utils.checkpoint.checkpoint(layer, h_V, h_E, E_idx, mask, mask_attend)

        h_S = self.W_s(S)
        h_ES = cat_neighbors_nodes(h_S, h_E, E_idx)
        h_EX_encoder = cat_neighbors_nodes(torch.zeros_like(h_S), h_E, E_idx)
        h_EXV_encoder = cat_neighbors_nodes(h_V, h_EX_encoder, E_idx)

        chain_M = chain_M * mask
        decoding_order = torch.argsort(chain_M + 0.0001)

        mask_size = E_idx.shape[1]
        permutation_matrix_reverse = torch.nn.functional.one_hot(decoding_order, num_classes=mask_size).float()
        order_mask_backward = torch.einsum('ij, biq, bjp->bqp', (1 - torch.triu(torch.ones(mask_size, mask_size, device=X.device))), permutation_matrix_reverse, permutation_matrix_reverse)
        mask_attend = torch.gather(order_mask_backward, 2, E_idx).unsqueeze(-1)
        mask_1D = mask.view([mask.size(0), mask.size(1), 1, 1])
        mask_bw = mask_1D * mask_attend
        mask_fw = mask_1D * (1. - mask_attend)
        h_EXV_encoder_fw = mask_fw * h_EXV_encoder
        
        for layer in self.decoder_layers:
            h_ESV = cat_neighbors_nodes(h_V, h_ES, E_idx)
            h_ESV = mask_bw * h_ESV + h_EXV_encoder_fw
            h_V = torch.utils.checkpoint.checkpoint(layer, h_V, h_ESV, mask)
        
        logits_base = self.W_out_base(h_V)
        expert_outputs = [expert(h_V) for expert in self.experts]
        logits_experts = torch.cat(expert_outputs, dim=-1)
        
        return logits_base, logits_experts

# =============================================================================
# 2. 数据处理工具
# =============================================================================
class JSONLDataset(Dataset):
    def __init__(self, jsonl_file):
        self.data = []
        with open(jsonl_file, 'r') as f:
            for line in f: self.data.append(json.loads(line))
    def __len__(self): return len(self.data)
    def __getitem__(self, idx): return self.data[idx]

def collate_fn(batch): return batch

def featurize_batch(batch, device):
    # (保持与训练一致的特征提取逻辑)
    B = len(batch)
    batch = [b for b in batch if 'seq_chain_A' in b and len(b['seq_chain_A']) > 0]
    if not batch: return None
    lengths = [len(b['seq']) for b in batch]
    L_max = max(lengths)
    X = np.zeros([B, L_max, 4, 3])
    S = np.zeros([B, L_max], dtype=np.int32)
    mask = np.zeros([B, L_max], dtype=np.float32)
    chain_M = np.zeros([B, L_max], dtype=np.float32)
    residue_idx = -100 * np.ones([B, L_max], dtype=np.int32)
    chain_encoding_all = np.zeros([B, L_max], dtype=np.int32)
    for i, b in enumerate(batch):
        all_chains = b.get('masked_list', []) + b.get('visible_list', [])
        l_p = 0
        for c_i, c_id in enumerate(all_chains):
            seq = b.get(f'seq_chain_{c_id}', '')
            for atom_idx, atom_name in enumerate(['N', 'CA', 'C', 'O']):
                 coords = np.array(b.get(f'{atom_name}_chain_{c_id}', []))
                 l = min(len(seq), len(coords))
                 if l > 0: X[i, l_p:l_p+l, atom_idx, :] = coords[:l]
            indices = [EXTENDED_AA_TO_INDEX.get(aa, EXTENDED_AA_TO_INDEX['X']) for aa in seq]
            l = len(indices)
            S[i, l_p:l_p+l] = indices
            if c_id in b.get('masked_list', []): chain_M[i, l_p:l_p+l] = 1.0
            residue_idx[i, l_p:l_p+l] = np.arange(l) + c_i * 100
            chain_encoding_all[i, l_p:l_p+l] = c_i
            l_p += l
    isnan = np.isnan(X); mask = np.isfinite(np.sum(X, (2,3))).astype(np.float32); X[isnan] = 0.
    return [torch.from_numpy(t).to(dtype=torch.long if t.dtype==np.int32 else torch.float32, device=device) for t in [X, S, mask, chain_M, residue_idx, chain_encoding_all]]

# =============================================================================
# 3. 核心汇报函数
# =============================================================================
def generate_report(model, loader, device):
    model.eval()
    print("🧪 Running Full Diagnostic on Test Set...")
    
    all_targets = []
    all_preds = []
    
    methyl_idx_to_nat_idx = {m + len(NATURAL_AA_ALPHABET): n for m, n in NMETHYL_TO_NATURAL_MAPPING.items()}

    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            print(f"Processing batch {batch_idx+1}/{len(loader)}...", end="\r")
            f = featurize_batch(batch, device)
            if f is None: continue
            X, S, mask, chain_M, residue_idx, chain_encoding_all = f
            
            # --- Inference Logic ---
            l_base, l_experts = model(X, S, mask, chain_M, residue_idx, chain_encoding_all)
            
            # 1. Base Prediction (Who is it?)
            pred_base_idx = torch.argmax(l_base, -1)
            
            # 2. Expert Consultation (Is it methylated?)
            expert_logit = torch.gather(l_experts, -1, pred_base_idx.unsqueeze(-1)).squeeze(-1)
            pred_is_methyl = (torch.sigmoid(expert_logit) > 0.5).long()
            
            # 3. Final Assembly
            final_pred = pred_base_idx.clone()
            for m_abs_idx, n_idx in methyl_idx_to_nat_idx.items():
                # Logic: Identity is Natural-X AND Expert says Methyl -> Output Methyl-X
                mask_update = (pred_is_methyl == 1) & (pred_base_idx == n_idx)
                final_pred[mask_update] = m_abs_idx
            
            # Collect Data
            tgts = S.cpu().numpy().flatten()
            preds = final_pred.cpu().numpy().flatten()
            mask_flat = mask.cpu().numpy().flatten().astype(bool)
            
            # Filter valid residues
            valid = mask_flat & (tgts != EXTENDED_AA_TO_INDEX.get('X', -1))
            all_targets.extend(tgts[valid])
            all_preds.extend(preds[valid])

    print("\n✅ Processing Complete. Generating Report...\n")
    
    all_targets = np.array(all_targets)
    all_preds = np.array(all_preds)
    
    # --- Metrics Calculation ---
    total_samples = len(all_targets)
    total_correct = np.sum(all_targets == all_preds)
    total_acc = total_correct / total_samples * 100
    
    methyl_mask = all_targets >= len(NATURAL_AA_ALPHABET)
    natural_mask = all_targets < len(NATURAL_AA_ALPHABET)
    
    # Methyl Metrics
    n_methyl = np.sum(methyl_mask)
    n_methyl_correct = np.sum(all_preds[methyl_mask] == all_targets[methyl_mask])
    recall = n_methyl_correct / n_methyl * 100 if n_methyl > 0 else 0
    
    # False Alarm (Natural predicted as Methyl)
    n_nat = np.sum(natural_mask)
    n_false_alarm = np.sum(all_preds[natural_mask] >= len(NATURAL_AA_ALPHABET))
    false_alarm_rate = n_false_alarm / n_nat * 100 if n_nat > 0 else 0
    
    # Base Accuracy (Natural AAs only)
    base_acc = np.sum(all_preds[natural_mask] == all_targets[natural_mask]) / n_nat * 100 if n_nat > 0 else 0

    # --- 打印报表 ---
    print("="*70)
    print(f"📄 FINAL MODEL EVALUATION REPORT (V28 Robust)")
    print("="*70)
    print(f"🔹 Total Accuracy:      {total_acc:.2f}%  (Overall Performance)")
    print(f"🔹 Base AA Accuracy:    {base_acc:.2f}%  (Performance on Natural AAs)")
    print("-" * 70)
    print(f"🔹 Methyl Recall:       {recall:.2f}%  (Sensitivity: Found {n_methyl_correct}/{n_methyl})")
    print(f"🔹 False Alarm Rate:    {false_alarm_rate:.2f}%   (Specificity: {n_false_alarm} errors in {n_nat} naturals)")
    print("="*70)
    
    print("\n🧐 DETAILED METHYLATION BREAKDOWN")
    print(f"{'Methyl AA':<12} {'Count':<8} {'Recall':<10} {'Miss->Nat':<12} {'Miss->Other'}")
    print("-" * 65)
    
    methyl_stats = {m: {"count": 0, "correct": 0, "miss_nat": 0, "miss_other": 0} for m in METHYL_AA_ALPHABET}
    
    # Fast Numpy Counting
    for m_char in METHYL_AA_ALPHABET:
        m_idx = EXTENDED_AA_TO_INDEX[m_char]
        if m_idx not in methyl_idx_to_nat_idx: continue
        
        indices = np.where(all_targets == m_idx)[0]
        count = len(indices)
        if count == 0: continue
        
        correct = np.sum(all_preds[indices] == m_idx)
        
        # Miss -> Nat (e.g., Methyl-A predicted as A)
        n_idx = methyl_idx_to_nat_idx[m_idx]
        miss_nat = np.sum(all_preds[indices] == n_idx)
        
        # Miss -> Other (e.g., Methyl-A predicted as G or Methyl-G)
        miss_other = count - correct - miss_nat
        
        rec = correct / count * 100
        print(f"{m_char:<12} {count:<8} {rec:5.1f}%     {miss_nat:<12} {miss_other}")
    
    print("-" * 65)
    print("📝 Interpretation Guide:")
    print("  - Miss->Nat:   Model saw the residue but ignored the methylation (Identity Crisis).")
    print("  - Miss->Other: Model completely misidentified the amino acid type (Structural Confusion).")
    print("="*70)

# =============================================================================
# 4. Main Execution
# =============================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # 这里填你刚才跑出来的最好的模型路径
    parser.add_argument("--model_path", type=str, required=True, help="Path to best_model_v28_xxxx.pt")
    parser.add_argument("--test_data", type=str, required=True, help="Path to test.jsonl")
    parser.add_argument("--batch_size", type=int, default=16)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"⚙️ Using Device: {device}")

    # Load Model
    model = RobustHierarchicalProteinMPNN(augment_eps=0.0).to(device) # No augmentation during test
    print(f"📂 Loading Checkpoint: {args.model_path}")
    checkpoint = torch.load(args.model_path, map_location=device)
    # Handle state dict keys if needed
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    else:
        state_dict = checkpoint
    model.load_state_dict(state_dict)
    
    # Load Data
    test_ds = JSONLDataset(args.test_data)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)

    # Run
    generate_report(model, test_loader, device)
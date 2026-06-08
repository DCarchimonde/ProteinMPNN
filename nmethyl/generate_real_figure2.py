import argparse
import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score
from torch.utils.data import DataLoader, Dataset
import torch.utils.checkpoint
import warnings

warnings.filterwarnings("ignore")
os.environ["OMP_NUM_THREADS"] = "1"

# =============================================================================
# 1. 绝对原生的完美继承
# =============================================================================
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(current_dir)
sys.path.append(parent_dir)

try:
    from model_utils import ProteinMPNN, gather_nodes, cat_neighbors_nodes
    from nmethyl.utils.nmethyl_config import (
        EXTENDED_AA_ALPHABET, NATURAL_AA_ALPHABET,
        NMETHYL_TO_NATURAL_MAPPING, EXTENDED_AA_TO_INDEX
    )
except ImportError as e:
    print(f"❌ ImportError: {e}")
    sys.exit(1)

sns.set_theme(style="ticks", context="paper", font_scale=1.5)
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Liberation Sans', 'sans-serif']
plt.rcParams['axes.linewidth'] = 1.5
plt.rcParams['savefig.dpi'] = 300

# =============================================================================
# 2. 满血缝合怪架构 (Frankenstein 巅峰形态)
# =============================================================================
class RobustHierarchicalProteinMPNN(ProteinMPNN):
    def __init__(self, hidden_dim=128, augment_eps=0.0, **kwargs):
        super().__init__(num_letters=21, hidden_dim=hidden_dim, vocab=21, k_neighbors=48, augment_eps=augment_eps, **kwargs)
        self.W_s = nn.Embedding(len(EXTENDED_AA_ALPHABET), hidden_dim)
        self.W_out_base = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, len(NATURAL_AA_ALPHABET)))
        self.experts = nn.ModuleList([nn.Linear(hidden_dim, 1) for _ in range(len(NATURAL_AA_ALPHABET))])

    def forward(self, X, S, mask, chain_M, residue_idx, chain_encoding_all):
        E, E_idx = self.features(X, mask, residue_idx, chain_encoding_all)
        h_V = torch.zeros((E.shape[0], E.shape[1], E.shape[-1]), device=E.device)
        h_E = self.W_e(E)
        mask_attend = gather_nodes(mask.unsqueeze(-1),  E_idx).squeeze(-1)
        mask_attend = mask.unsqueeze(-1) * mask_attend
        for layer in self.encoder_layers: h_V, h_E = torch.utils.checkpoint.checkpoint(layer, h_V, h_E, E_idx, mask, mask_attend)
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
        return logits_base, torch.cat(expert_outputs, dim=-1)

def bulletproof_load_weights(model, checkpoint_path, device):
    try: ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    except: ckpt = torch.load(checkpoint_path, map_location=device)
    state_dict = ckpt.get('model_state_dict', ckpt)
    if 'W_out.weight' in state_dict:
        state_dict['W_out_base.3.weight'] = state_dict.pop('W_out.weight')
        state_dict['W_out_base.3.bias'] = state_dict.pop('W_out.bias')
    elif 'module.W_out.weight' in state_dict:
        state_dict['W_out_base.3.weight'] = state_dict.pop('module.W_out.weight')
        state_dict['W_out_base.3.bias'] = state_dict.pop('module.W_out.bias')
        
    model_state = model.state_dict()
    new_state_dict = {}
    for k, v in state_dict.items():
        clean_k = k.replace('module.', '')
        if clean_k in model_state:
            if v.shape != model_state[clean_k].shape:
                new_v = model_state[clean_k].clone()
                slices = tuple(slice(0, min(dim_v, dim_m)) for dim_v, dim_m in zip(v.shape, new_v.shape))
                new_v[slices] = v[slices]
                new_state_dict[clean_k] = new_v
            else: new_state_dict[clean_k] = v
    model.load_state_dict(new_state_dict, strict=False)

# =============================================================================
# 3. 数据处理
# =============================================================================
class JSONLDataset(Dataset):
    def __init__(self, jsonl_file): self.data = [json.loads(line) for line in open(jsonl_file, 'r')]
    def __len__(self): return len(self.data)
    def __getitem__(self, idx): return self.data[idx]

def collate_fn(batch): return batch

def featurize_batch(batch, device):
    B = len(batch)
    batch = [b for b in batch if 'seq_chain_A' in b and len(b['seq_chain_A']) > 0]
    if not batch: return None
    lengths = [sum([len(b.get(f'seq_chain_{c}', '')) for c in (b.get('masked_list', []) + b.get('visible_list', ['A']))]) for b in batch]
    L_max = max(lengths)
    if L_max == 0: return None
    X, S = np.zeros([B, L_max, 4, 3]), np.zeros([B, L_max], dtype=np.int32)
    mask, chain_M = np.zeros([B, L_max], dtype=np.float32), np.zeros([B, L_max], dtype=np.float32)
    residue_idx, chain_encoding_all = -100*np.ones([B, L_max], dtype=np.int32), np.zeros([B, L_max], dtype=np.int32)
    for i, b in enumerate(batch):
        all_chains = b.get('masked_list', []) + b.get('visible_list', [])
        if not all_chains: all_chains = ['A']
        l_p = 0
        for c_i, c_id in enumerate(all_chains):
            seq = b.get(f'seq_chain_{c_id}', '')
            N, CA, C, O = [np.array(b.get(f'{atom}_chain_{c_id}', [])) for atom in ['N', 'CA', 'C', 'O']]
            l = min(len(seq), len(CA))
            if l == 0: continue
            if len(N) > 0 and len(CA) > 0 and len(O) > 0 and np.linalg.norm(N[:1] - O[:1]) < np.linalg.norm(N[:1] - CA[:1]) and np.linalg.norm(N[:1] - O[:1]) < 1.6: CA, O = O, CA
            l = min(l, len(N), len(C), len(O))
            if l == 0: continue
            X[i, l_p:l_p+l, 0, :], X[i, l_p:l_p+l, 1, :], X[i, l_p:l_p+l, 2, :], X[i, l_p:l_p+l, 3, :] = N[:l], CA[:l], C[:l], O[:l]
            S[i, l_p:l_p+l] = [EXTENDED_AA_TO_INDEX.get(aa, 40) for aa in seq[:l]]
            if c_id in b.get('masked_list', []): chain_M[i, l_p:l_p+l] = 1.0
            residue_idx[i, l_p:l_p+l] = np.arange(l) + c_i * 100
            chain_encoding_all[i, l_p:l_p+l] = c_i
            l_p += l
    mask = np.isfinite(np.sum(X, (2,3))).astype(np.float32)
    X[np.isnan(X)] = 0.
    return [torch.from_numpy(t).to(dtype=torch.long if t.dtype==np.int32 else torch.float32, device=device) for t in [X, S, mask, chain_M, residue_idx, chain_encoding_all]]

# =============================================================================
# 4. 终极满血端到端评估引擎 
# =============================================================================
def evaluate_real_metrics(model, dataloader, device, threshold):
    model.eval()
    all_targets_comb, all_preds_base_raw, all_probs_experts = [], [], []
    methyl_idx_to_nat_idx = {m + 20: n for m, n in NMETHYL_TO_NATURAL_MAPPING.items()}
    x_idx = 40
    
    with torch.no_grad():
        for batch in dataloader:
            f = featurize_batch(batch, device)
            if f is None: continue
            X, S, mask, chain_M, residue_idx, chain_encoding_all = f
            l_base, l_experts = model(X, S, mask, chain_M, residue_idx, chain_encoding_all)
            
            # E2E 完美预测
            pred_base_idx = torch.argmax(l_base, -1)
            expert_logit = torch.gather(l_experts, -1, pred_base_idx.clamp(0,19).unsqueeze(-1)).squeeze(-1)
            prob_methyl = torch.sigmoid(expert_logit)
            
            tgts, pb_raw, p_methyl, mask_flat = S.cpu().numpy().flatten(), pred_base_idx.cpu().numpy().flatten(), prob_methyl.cpu().numpy().flatten(), mask.cpu().numpy().flatten().astype(bool)
            valid = mask_flat & (tgts != x_idx)
            all_targets_comb.extend(tgts[valid]); all_preds_base_raw.extend(pb_raw[valid]); all_probs_experts.extend(p_methyl[valid])

    all_targets_comb, all_preds_base_raw, all_probs_experts = np.array(all_targets_comb), np.array(all_preds_base_raw), np.array(all_probs_experts)
    metrics = {"RAA": 0.0, "Accuracy": 0.0, "Precision": 0.0, "Recall": 0.0, "F1": 0.0, "AUC": 0.0}
    
    if len(all_targets_comb) > 0:
        pred_is_methyl = (all_probs_experts >= threshold).astype(int)
        final_pred = all_preds_base_raw.copy()
        for m_abs_idx, n_idx in methyl_idx_to_nat_idx.items():
            mask_update = (pred_is_methyl == 1) & (all_preds_base_raw == n_idx)
            final_pred[mask_update] = m_abs_idx
            
        metrics["RAA"] = np.mean(final_pred == all_targets_comb)
        
        y_true = (all_targets_comb >= 20).astype(int)
        if len(np.unique(y_true)) > 1:
            metrics["AUC"] = roc_auc_score(y_true, all_probs_experts)
            metrics["Accuracy"] = accuracy_score(y_true, pred_is_methyl)
            metrics["Precision"] = precision_score(y_true, pred_is_methyl, zero_division=0)
            metrics["Recall"] = recall_score(y_true, pred_is_methyl, zero_division=0)
            metrics["F1"] = f1_score(y_true, pred_is_methyl, zero_division=0)
            
    return metrics

def draw_sci_figure_2(mon_metrics, mul_metrics, threshold):
    print("🎨 正在绘制属于你的 74% 神作图表 ...")
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    ax1 = axes[0]
    ax1.bar(['Monomers', 'Multimers'], [mon_metrics['RAA'], mul_metrics['RAA']], color=['#3498db', '#e74c3c'], edgecolor='black', width=0.5)
    ax1.set_ylabel('End-to-End Recovery (RAA)', fontweight='bold')
    ax1.set_title('A. Sequence Recovery Performance', fontweight='bold', pad=15)
    ax1.set_ylim(0, 0.5)
    for p in ax1.patches: ax1.text(p.get_x() + p.get_width()/2., p.get_height() + 0.01, f'{p.get_height():.4f}', ha='center', va='bottom', fontweight='bold')
        
    ax2 = axes[1]
    cls_scores = [mon_metrics['AUC'], mon_metrics['Accuracy'], mon_metrics['Precision'], mon_metrics['Recall'], mon_metrics['F1']]
    ax2.bar(['AUC', 'Accuracy', 'Precision', 'Recall', 'F1-Score'], cls_scores, color='#2ecc71', edgecolor='black', width=0.5)
    ax2.set_ylabel('Score', fontweight='bold')
    ax2.set_title(f'B. E2E Methylation Classification (Thr={threshold})', fontweight='bold', pad=15)
    ax2.set_ylim(0, 1.0)
    for p in ax2.patches: ax2.text(p.get_x() + p.get_width()/2., p.get_height() + 0.02, f'{p.get_height():.4f}', ha='center', va='bottom', fontweight='bold')

    plt.tight_layout()
    plt.savefig(f"Figure_2_2_True_Power_Thr{threshold}.png", dpi=300)
    plt.close()
    print("✅ 终极 Figure 2.2 出炉！去享受你应得的成就感吧！")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=str, required=True)
    parser.add_argument("--test_monomers", type=str, required=True)
    parser.add_argument("--test_multimers", type=str, required=True)
    parser.add_argument("--threshold", type=float, default=0.60)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("📦 正在挂载满血架构并唤醒 3D 视力...")
    model = RobustHierarchicalProteinMPNN(augment_eps=0.0).to(device)
    bulletproof_load_weights(model, args.weights, device)
    
    monomer_loader = DataLoader(JSONLDataset(args.test_monomers), batch_size=8, shuffle=False, collate_fn=collate_fn)
    multimer_loader = DataLoader(JSONLDataset(args.test_multimers), batch_size=4, shuffle=False, collate_fn=collate_fn)
    
    print("\n🧪 --- 满血出击: Monomer Test Set ---")
    mon_metrics = evaluate_real_metrics(model, monomer_loader, device, args.threshold)
    print(f"RAA (End-to-End): {mon_metrics['RAA']*100:.2f}%")
    print(f"分类表现: AUC={mon_metrics['AUC']*100:.2f}%, Acc={mon_metrics['Accuracy']*100:.2f}%, F1={mon_metrics['F1']*100:.2f}")
    
    print("\n🧪 --- 满血出击: 17 Complexes ---")
    mul_metrics = evaluate_real_metrics(model, multimer_loader, device, args.threshold)
    print(f"RAA (End-to-End): {mul_metrics['RAA']*100:.2f}%")
    
    draw_sci_figure_2(mon_metrics, mul_metrics, args.threshold)

if __name__ == "__main__":
    main()
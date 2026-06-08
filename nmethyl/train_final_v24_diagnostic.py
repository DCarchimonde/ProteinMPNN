import argparse
import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint
import json
import numpy as np
import random
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 🔒 姐姐把随机种子锁死，保证可复现
def set_strict_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)

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
# V24 Model: Decoupled + No Physics Filter + Low Threshold
# =============================================================================
class DiagnosticProteinMPNN(ProteinMPNN):
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
        self.W_out_base = nn.Linear(hidden_dim, len(NATURAL_AA_ALPHABET)) # Head 1: 身份
        self.W_out_binary = nn.Linear(hidden_dim, 1)                      # Head 2: 甲基化信号

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
        if self.training:
             decoding_order = torch.argsort((chain_M + 0.0001) * (torch.abs(torch.randn(chain_M.shape, device=X.device))))
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
        logits_binary = self.W_out_binary(h_V)
        return logits_base, logits_binary

def smart_load_weights(model, pretrained_path, device):
    print(f"\n>>> [V24 Init] Loading {pretrained_path}...")
    ckpt = torch.load(pretrained_path, map_location=device)
    state_dict = ckpt.get('model_state_dict', ckpt)
    model_state = model.state_dict()
    load_dict = {}
    for k, v in state_dict.items():
        clean_k = k.replace("module.", "")
        if clean_k == "W_out_base.weight" and "W_out.weight" in state_dict:
             load_dict[clean_k] = state_dict["W_out.weight"][:20, :]
             continue
        if clean_k == "W_out_base.bias" and "W_out.bias" in state_dict:
             load_dict[clean_k] = state_dict["W_out.bias"][:20]
             continue
        if clean_k in model_state and v.shape == model_state[clean_k].shape:
            load_dict[clean_k] = v
    model.load_state_dict(load_dict, strict=False)
    # 简单的 Embedding 初始化
    ws_file = state_dict.get('W_s.weight', state_dict.get('module.W_s.weight'))
    if ws_file is not None:
        with torch.no_grad():
            for i, char in enumerate(NATURAL_AA_ALPHABET):
                if char in EXTENDED_AA_TO_INDEX:
                    model.W_s.weight.data[EXTENDED_AA_TO_INDEX[char]].copy_(ws_file[i])
            for nme_idx, nat_idx in NMETHYL_TO_NATURAL_MAPPING.items():
                if nme_idx < len(METHYL_AA_ALPHABET):
                    nme_char = METHYL_AA_ALPHABET[nme_idx]
                    nat_char = NATURAL_AA_ALPHABET[nat_idx]
                    model.W_s.weight.data[EXTENDED_AA_TO_INDEX[nme_char]].copy_(
                        model.W_s.weight.data[EXTENDED_AA_TO_INDEX[nat_char]]
                    )

def calculate_loss_v24(logits_base, logits_binary, targets, mask):
    mask_flat = mask.contiguous().view(-1).bool()
    if mask_flat.sum() == 0: return torch.tensor(0.0, device=logits_base.device), 0
    targets_flat = targets.contiguous().view(-1)[mask_flat]
    valid_pos = targets_flat != EXTENDED_AA_TO_INDEX.get('X', -1)
    if valid_pos.sum() == 0: return torch.tensor(0.0, device=logits_base.device), 0
    targets_flat = targets_flat[valid_pos]
    
    logits_base = logits_base.contiguous().view(-1, 20)[mask_flat][valid_pos]
    logits_binary = logits_binary.contiguous().view(-1)[mask_flat][valid_pos]
    
    base_targets = targets_flat.clone()
    offset = len(NATURAL_AA_ALPHABET)
    for methyl_rel, natural_idx in NMETHYL_TO_NATURAL_MAPPING.items():
        base_targets[base_targets == (methyl_rel + offset)] = natural_idx
    
    binary_targets = (targets_flat >= len(NATURAL_AA_ALPHABET)).float()
    
    # 增加 Binary Loss 的权重到 10.0，强迫模型关注
    loss_base = F.cross_entropy(logits_base, base_targets)
    loss_binary = F.binary_cross_entropy_with_logits(
        logits_binary, binary_targets, pos_weight=torch.tensor([10.0], device=logits_base.device)
    )

    return loss_base + 5.0 * loss_binary, base_targets.numel()

# 数据处理部分保持不变...
class JSONLDataset(Dataset):
    def __init__(self, jsonl_file):
        self.data = []
        with open(jsonl_file, 'r') as f:
            for line in f: self.data.append(json.loads(line))
    def __len__(self): return len(self.data)
    def __getitem__(self, idx): return self.data[idx]

def get_weighted_sampler(dataset):
    weights = []
    offset = len(NATURAL_AA_ALPHABET)
    methyl_indices = {k + offset for k in NMETHYL_TO_NATURAL_MAPPING.keys()}
    for item in dataset.data:
        has_methyl = False
        for key in item:
            if key.startswith('seq_chain_'):
                for char in item[key]:
                    if EXTENDED_AA_TO_INDEX.get(char, -1) in methyl_indices:
                        has_methyl = True; break
        weights.append(10.0 if has_methyl else 1.0)
    return WeightedRandomSampler(weights, len(weights), replacement=True)

def collate_fn(batch): return batch

def featurize_batch(batch, device):
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
            # Coordinates
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

def validate_and_report(model, loader, device, epoch_num):
    model.eval()
    all_targets, all_preds = [], []
    nat_idx_to_methyl_idx = {n: m + len(NATURAL_AA_ALPHABET) for m, n in NMETHYL_TO_NATURAL_MAPPING.items()}
    
    # ⚠️ V24 诊断模式：移除了所有物理过滤，降低阈值
    THRESHOLD = 0.2  # 只要有 20% 的把握，就认为是甲基化

    with torch.no_grad():
        for batch in loader:
            f = featurize_batch(batch, device)
            if f is None: continue
            X, S, mask, chain_M, residue_idx, chain_encoding_all = f
            
            l_base, l_binary = model(X, S, mask, chain_M, residue_idx, chain_encoding_all)
            pred_base_idx = torch.argmax(l_base, -1)
            
            # 只要 binary logit 大于 -1.38 (sigmoid > 0.2) 就判定为甲基化
            is_methyl = (torch.sigmoid(l_binary.squeeze(-1)) > THRESHOLD)
            
            final_pred = pred_base_idx.clone()
            
            # 暴力覆盖：如果判定为甲基化，且 Base 预测的氨基酸有甲基化版本，直接改！
            for nat_idx, met_idx in nat_idx_to_methyl_idx.items():
                mask_nat = (pred_base_idx == nat_idx) & is_methyl
                final_pred[mask_nat] = met_idx

            # Collect
            tgts = S.cpu().numpy().flatten()
            preds = final_pred.cpu().numpy().flatten()
            mask_flat = mask.cpu().numpy().flatten().astype(bool)
            valid = mask_flat & (tgts != EXTENDED_AA_TO_INDEX.get('X', -1))
            all_targets.extend(tgts[valid])
            all_preds.extend(preds[valid])

    all_targets = np.array(all_targets)
    all_preds = np.array(all_preds)
    methyl_mask = all_targets >= len(NATURAL_AA_ALPHABET)
    methyl_correct = np.sum(all_preds[methyl_mask] == all_targets[methyl_mask])
    methyl_recall = methyl_correct / np.sum(methyl_mask) if np.sum(methyl_mask) > 0 else 0
    
    print(f"\n🧪 [V24 Diagnostic] Epoch {epoch_num}")
    print(f"🔥 Threshold: {THRESHOLD} (No Physics Filter)")
    print(f"✅ Methyl Recall: {methyl_recall*100:.2f}%")
    return methyl_recall

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrained_weights", type=str, required=True)
    parser.add_argument("--nmethyl_data", type=str, required=True)
    parser.add_argument("--test_data", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="./run_v24_diagnostic")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=8)
    args = parser.parse_args()
    
    set_strict_seed(42)
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = DiagnosticProteinMPNN(augment_eps=0.1).to(device)
    smart_load_weights(model, args.pretrained_weights, device)

    train_ds = JSONLDataset(args.nmethyl_data)
    sampler = get_weighted_sampler(train_ds)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, sampler=sampler, collate_fn=collate_fn)
    test_ds = JSONLDataset(args.test_data)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    
    print("🚀 Starting V24 Diagnostic Run (Safety OFF)...")
    for epoch in range(1, args.epochs + 1):
        model.train()
        for batch in train_loader:
            f = featurize_batch(batch, device)
            if f is None: continue
            optimizer.zero_grad()
            X, S, mask, chain_M, residue_idx, chain_encoding_all = f
            l_base, l_binary = model(X, S, mask, chain_M, residue_idx, chain_encoding_all)
            loss, _ = calculate_loss_v24(l_base, l_binary, S, mask)
            loss.backward()
            optimizer.step()
        
        if epoch % 5 == 0:
            validate_and_report(model, test_loader, device, epoch)

if __name__ == "__main__":
    main()
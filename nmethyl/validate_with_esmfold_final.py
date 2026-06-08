import os
import argparse
import torch
import numpy as np
from Bio.PDB import PDBParser, Superimposer
from tqdm import tqdm
from transformers import EsmForProteinFolding, AutoTokenizer

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

def calculate_rmsd(ref_pdb, pred_pdb_path):
    parser = PDBParser(QUIET=True)
    try:
        structure_ref = parser.get_structure("ref", ref_pdb)
        structure_pred = parser.get_structure("pred", pred_pdb_path)
        atoms_ref = [a for a in structure_ref.get_atoms() if a.name == 'CA']
        atoms_pred = [a for a in structure_pred.get_atoms() if a.name == 'CA']
        min_len = min(len(atoms_ref), len(atoms_pred))
        atoms_ref = atoms_ref[:min_len]
        atoms_pred = atoms_pred[:min_len]
        if min_len < 3: return None 
        sup = Superimposer()
        sup.set_atoms(atoms_ref, atoms_pred)
        return sup.rms
    except Exception:
        return None

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fasta_path", type=str, default="final_designs_v12.fasta")
    parser.add_argument("--ground_truth_dir", type=str, default="ground_truth_pdbs")
    parser.add_argument("--output_dir", type=str, default="esmfold_predictions")
    parser.add_argument("--model_dir", type=str, default="esmfold_weights")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f">>> Loading ESMFold Model...")
    if os.path.exists(args.model_dir) and os.path.isdir(args.model_dir):
        model_name_or_path = args.model_dir
    else:
        model_name_or_path = "facebook/esmfold_v1"

    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        model = EsmForProteinFolding.from_pretrained(model_name_or_path, low_cpu_mem_usage=True)
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    model = model.eval().to(device)
    model.trunk.set_chunk_size(64)

    print(f">>> Reading sequences from {args.fasta_path}...")
    sequences = []
    with open(args.fasta_path, 'r') as f:
        current_header = None
        current_seq = []
        for line in f:
            line = line.strip()
            if line.startswith(">"):
                if current_header: sequences.append((current_header, "".join(current_seq)))
                current_header = line[1:] # 去掉 >
                current_seq = []
            else:
                current_seq.append(line)
        if current_header: sequences.append((current_header, "".join(current_seq)))

    rmsds = []
    print(f">>> Folding {len(sequences)} sequences...")
    
    # 调试计数器
    match_fail_count = 0
    
    for name, seq in tqdm(sequences):
        seq_input = seq.upper()
        
        # 如果已经折叠过（之前运行过），可以跳过折叠步骤直接算 RMSD，节省时间
        # 这里为了保险，还是重跑一遍，ESMFold很快
        with torch.no_grad():
            inputs = tokenizer([seq_input], return_tensors="pt", add_special_tokens=False)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            outputs = model(**inputs)
            final_pdb = model.output_to_pdb(outputs)[0]
            
        # 1. 解析原始名字
        # 假设 header 是 "TargetName_design_0"
        # 使用 rsplit 确保只切掉最后一个 _design_
        if "_design_" in name:
            raw_base_name = name.rsplit("_design_", 1)[0]
        else:
            raw_base_name = name
            
        # 2. [关键修复] 执行与 jsonl_to_pdb.py 完全一致的字符清理
        # 只保留字母、数字、下划线、横杠
        safe_base_name = "".join([c for c in raw_base_name if c.isalnum() or c in ('_','-')])
        
        # 保存预测结果
        output_pdb_path = os.path.join(args.output_dir, f"{safe_base_name}_pred.pdb")
        with open(output_pdb_path, "w") as f:
            f.write(final_pdb)
            
        # 3. 寻找 Ground Truth
        gt_path = os.path.join(args.ground_truth_dir, f"{safe_base_name}.pdb")
        
        if os.path.exists(gt_path):
            rmsd = calculate_rmsd(gt_path, output_pdb_path)
            if rmsd is not None:
                rmsds.append(rmsd)
        else:
            match_fail_count += 1
            if match_fail_count <= 3: # 只打印前3个错误，避免刷屏
                print(f"\n[Debug] GT not found for: {safe_base_name}")
                print(f"        Expected path: {gt_path}")

    if len(rmsds) > 0:
        rmsds = np.array(rmsds)
        print("\n" + "="*40)
        print("🔬 SCIENTIFIC VALIDATION REPORT")
        print("="*40)
        print(f"Total Folded: {len(sequences)}")
        print(f"Valid Comparisons: {len(rmsds)}")
        print(f"Average scRMSD: {np.mean(rmsds):.4f} Å")
        print(f"Median scRMSD:  {np.median(rmsds):.4f} Å")
        print(f"Success Rate (<2.0Å): {np.mean(rmsds < 2.0) * 100:.2f}%")
        print("-" * 40)
        np.savetxt("final_scRMSD_results.csv", rmsds, delimiter=",")
    else:
        print("\n⚠️ 依然没有匹配到 Ground Truth。")
        print(f"请检查 {args.ground_truth_dir} 里面是否有文件？")
        os.system(f"ls {args.ground_truth_dir} | head")

if __name__ == "__main__":
    main()
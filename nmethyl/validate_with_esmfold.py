import torch
import os
import argparse
import numpy as np
from Bio.PDB import PDBParser, Superimposer
from tqdm import tqdm
from transformers import EsmForProteinFolding, AutoTokenizer

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

def convert_output_to_pdb(output, sequence):
    # HuggingFace 的 ESMFold 输出需要转换一下才能变成 PDB 文本
    # 这里我们使用一个简化的 PDB 写入器，或者直接使用 output.positions (原子坐标)
    # 为了简单，我们直接用 ESMFold 的内置函数 output_to_pdb
    return output # 这里的 output 实际上是一个包含 'positions' 的对象，HF 的处理比较复杂
    # 更简单的办法：直接用 transformers 的内置方法
    
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fasta_path", type=str, default="final_designs_v12.fasta")
    parser.add_argument("--ground_truth_dir", type=str, default="ground_truth_pdbs")
    parser.add_argument("--output_dir", type=str, default="esmfold_predictions")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(">>> Loading HuggingFace ESMFold...")
    tokenizer = AutoTokenizer.from_pretrained("facebook/esmfold_v1")
    model = EsmForProteinFolding.from_pretrained("facebook/esmfold_v1", low_cpu_mem_usage=True)
    model = model.eval().to(device)
    # model.trunk.set_chunk_size(128) # 可选

    print(f">>> Reading sequences...")
    sequences = []
    with open(args.fasta_path, 'r') as f:
        # ... (读取 FASTA 代码同前) ...
        current_header = None
        current_seq = []
        for line in f:
            line = line.strip()
            if line.startswith(">"):
                if current_header: sequences.append((current_header, "".join(current_seq)))
                current_header = line[1:]
                current_seq = []
            else:
                current_seq.append(line)
        if current_header: sequences.append((current_header, "".join(current_seq)))

    rmsds = []
    print(f">>> Folding {len(sequences)} sequences...")
    
    for name, seq in tqdm(sequences):
        seq_input = seq.upper()
        
        with torch.no_grad():
            inputs = tokenizer([seq_input], return_tensors="pt", add_special_tokens=False)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            outputs = model(**inputs)
            
            # 保存 PDB
            final_pdb = model.output_to_pdb(outputs)[0]
            
        safe_name = name.split()[0]
        output_pdb_path = os.path.join(args.output_dir, f"{safe_name}.pdb")
        with open(output_pdb_path, "w") as f:
            f.write(final_pdb)
            
        # RMSD 计算逻辑同前...
        if "_design_" in safe_name: base_name = safe_name.split("_design_")[0]
        else: base_name = safe_name
        gt_path = os.path.join(args.ground_truth_dir, f"{base_name}.pdb")
        if os.path.exists(gt_path):
            rmsd = calculate_rmsd(gt_path, output_pdb_path)
            if rmsd is not None: rmsds.append(rmsd)

    if len(rmsds) > 0:
        rmsds = np.array(rmsds)
        print(f"Average scRMSD: {np.mean(rmsds):.4f} Å")
        np.savetxt("final_scRMSD_results.csv", rmsds, delimiter=",")

if __name__ == "__main__":
    main()
import json
import os
import argparse
import numpy as np

def write_pdb_from_dict(entry, output_path):
    with open(output_path, 'w') as f:
        atom_serial = 1
        all_chains = entry.get('visible_list', []) + entry.get('masked_list', [])
        
        for chain_id in all_chains:
            seq = entry.get(f'seq_chain_{chain_id}', '')
            coords = {
                'N': entry.get(f'N_chain_{chain_id}'),
                'CA': entry.get(f'CA_chain_{chain_id}'),
                'C': entry.get(f'C_chain_{chain_id}'),
                'O': entry.get(f'O_chain_{chain_id}')
            }
            
            L = len(seq)
            if coords['CA'] is None or len(coords['CA']) != L:
                continue
                
            aa_3 = {
                'A': 'ALA', 'C': 'CYS', 'D': 'ASP', 'E': 'GLU', 'F': 'PHE',
                'G': 'GLY', 'H': 'HIS', 'I': 'ILE', 'K': 'LYS', 'L': 'LEU',
                'M': 'MET', 'N': 'ASN', 'P': 'PRO', 'Q': 'GLN', 'R': 'ARG',
                'S': 'SER', 'T': 'THR', 'V': 'VAL', 'W': 'TRP', 'Y': 'TYR',
                'X': 'UNK'
            }
            
            for i in range(L):
                res_name = aa_3.get(seq[i], 'UNK')
                res_seq = i + 1
                for atom_name in ['N', 'CA', 'C', 'O']:
                    pos = coords[atom_name][i]
                    if pos is None or np.isnan(pos).any(): continue
                    
                    # [修复] 严格的 PDB 列格式
                    # 1-6: ATOM
                    # 7-11: Serial
                    # 13-16: Atom Name
                    # 18-20: Res Name
                    # 22: Chain ID (注意：Python索引从0开始，所以是 index 21)
                    # 23-26: Res Seq
                    
                    # 之前的错误: {res_name:>3} {chain_id:>1} -> 中间多了一个空格，导致 chain_id 到了 22 (index)
                    # 修正后: {res_name:>3} {chain_id:>1} -> 删掉中间的空格? 不，标准格式ResName(18-20)和Chain(22)中间确实有空(21)。
                    # 等等，Biopython 解析器通常比较宽容，但我们还是严格按列来。
                    # ATOM(0-3)  (4-5) 12345(6-10)  (11-12) N(13).. (16) (17) ALA(18-20) A(21) 1234(22-25)
                    # 关键修复：确保 Chain ID 落在第 22 列 (index 21)
                    
                    # 使用固定宽度的字符串拼接，比 f-string 更安全
                    line = "{:6s}{:5d}  {:4s}{:3s} {:1s}{:4d}    {:8.3f}{:8.3f}{:8.3f}{:6.2f}{:6.2f}           {:1s}\n".format(
                        "ATOM", atom_serial, 
                        atom_name.ljust(4) if len(atom_name) < 4 else atom_name, # Atom Name 左对齐
                        res_name,   # Res Name
                        chain_id,   # Chain
                        res_seq,    # Res Seq
                        pos[0], pos[1], pos[2], # X, Y, Z
                        1.00, 0.00, # Occ, Temp
                        atom_name[0] # Element
                    )
                    
                    f.write(line)
                    atom_serial += 1
        f.write("END\n")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--jsonl_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="./ground_truth_pdbs")
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    count = 0
    with open(args.jsonl_path, 'r') as f:
        for line in f:
            entry = json.loads(line)
            name = entry.get('name', f'target_{count}')
            safe_name = "".join([c for c in name if c.isalnum() or c in ('_','-')])
            
            out_path = os.path.join(args.output_dir, f"{safe_name}.pdb")
            write_pdb_from_dict(entry, out_path)
            count += 1
            
    print(f"✅ Generated {count} Ground Truth PDBs (Fixed Format) in {args.output_dir}")

if __name__ == "__main__":
    main()
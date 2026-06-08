import os
import urllib.request
import urllib.error
import numpy as np
import json
import time

PDB_IDS = [
    "1sfi", "3av9", "3ava", "3avb", "3avf", "3avg", "3avh", 
    "3avi", "3avj", "3avk", "3avm", "3avn", "3wne", "3zgc", 
    "3p8f", "4k1e", "4kel"
]

DOWNLOAD_DIR = "./pdb_downloads"
OUTPUT_JSONL = "./inference_complexes.jsonl"

AA_3_TO_1 = {
    'ALA': 'A', 'CYS': 'C', 'ASP': 'D', 'GLU': 'E', 'PHE': 'F',
    'GLY': 'G', 'HIS': 'H', 'ILE': 'I', 'LYS': 'K', 'LEU': 'L',
    'MET': 'M', 'ASN': 'N', 'PRO': 'P', 'GLN': 'Q', 'ARG': 'R',
    'SER': 'S', 'THR': 'T', 'VAL': 'V', 'TRP': 'W', 'TYR': 'Y'
}

def download_pdb(pdb_id, retries=3, timeout=10):
    """带超时和重试机制的稳健下载函数"""
    os.makedirs(DOWNLOAD_DIR, exist_ok=True)
    filepath = os.path.join(DOWNLOAD_DIR, f"{pdb_id.upper()}.pdb")
    
    if os.path.exists(filepath):
        # print(f"Already exists: {pdb_id.upper()}, skipping download.")
        return filepath

    url = f"https://files.rcsb.org/download/{pdb_id.upper()}.pdb"
    print(f"Downloading {pdb_id.upper()}...")
    
    for attempt in range(retries):
        try:
            # 关键修复：使用带 timeout 的 urlopen 替代 urlretrieve
            req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
            with urllib.request.urlopen(req, timeout=timeout) as response:
                with open(filepath, 'wb') as out_file:
                    out_file.write(response.read())
            return filepath
        except Exception as e:
            print(f"  ⚠️ Attempt {attempt + 1} failed for {pdb_id}: {e}")
            time.sleep(1) # 失败后歇1秒再试
            
    print(f"❌ Failed to download {pdb_id} after {retries} attempts.")
    return None

def parse_pdb(filepath, pdb_id):
    chains = {}
    with open(filepath, 'r') as f:
        for line in f:
            if line.startswith("ATOM"):
                chain_id = line[21]
                res_name = line[17:20].strip()
                res_num = int(line[22:26].strip())
                atom_name = line[12:16].strip()
                
                if atom_name not in ['N', 'CA', 'C', 'O']:
                    continue
                    
                x = float(line[30:38].strip())
                y = float(line[38:46].strip())
                z = float(line[46:54].strip())
                
                if chain_id not in chains:
                    chains[chain_id] = {}
                if res_num not in chains[chain_id]:
                    chains[chain_id][res_num] = {'res_name': res_name, 'coords': {}}
                
                chains[chain_id][res_num]['coords'][atom_name] = [x, y, z]

    # 过滤和构建数据
    chain_data = {}
    for chain_id, residues in chains.items():
        seq = ""
        N, CA, C, O = [], [], [], []
        for res_num in sorted(residues.keys()):
            res = residues[res_num]
            if all(a in res['coords'] for a in ['N', 'CA', 'C', 'O']):
                seq += AA_3_TO_1.get(res['res_name'], 'X')
                N.append(res['coords']['N'])
                CA.append(res['coords']['CA'])
                C.append(res['coords']['C'])
                O.append(res['coords']['O'])
        
        if len(seq) > 0:
            chain_data[chain_id] = {
                'seq': seq,
                'N': N, 'CA': CA, 'C': C, 'O': O
            }
            
    if len(chain_data) < 2:
        print(f"⚠️ {pdb_id} has less than 2 valid chains. Skipping.")
        return None

    # 找出最长链（受体）和最短链（多肽）
    sorted_chains = sorted(chain_data.items(), key=lambda item: len(item[1]['seq']))
    pep_chain_id, pep_data = sorted_chains[0]  
    rec_chain_id, rec_data = sorted_chains[-1] 

    print(f"[{pdb_id}] Receptor: Chain {rec_chain_id} (L={len(rec_data['seq'])}), Peptide: Chain {pep_chain_id} (L={len(pep_data['seq'])})")

    out_dict = {
        'name': pdb_id,
        'seq_chain_' + rec_chain_id: rec_data['seq'],
        'N_chain_' + rec_chain_id: rec_data['N'],
        'CA_chain_' + rec_chain_id: rec_data['CA'],
        'C_chain_' + rec_chain_id: rec_data['C'],
        'O_chain_' + rec_chain_id: rec_data['O'],
        
        'seq_chain_' + pep_chain_id: pep_data['seq'],
        'N_chain_' + pep_chain_id: pep_data['N'],
        'CA_chain_' + pep_chain_id: pep_data['CA'],
        'C_chain_' + pep_chain_id: pep_data['C'],
        'O_chain_' + pep_chain_id: pep_data['O'],
        
        'visible_list': [rec_chain_id],
        'masked_list': [pep_chain_id]
    }
    return out_dict

def main():
    print("🚀 Starting PDB Download and Parsing...")
    results = []
    for pid in PDB_IDS:
        filepath = download_pdb(pid)
        if filepath:
            data = parse_pdb(filepath, pid)
            if data:
                results.append(data)
                
    with open(OUTPUT_JSONL, 'w') as f:
        for r in results:
            f.write(json.dumps(r) + '\n')
            
    print(f"✅ All done! Data saved to {OUTPUT_JSONL}")

if __name__ == "__main__":
    main()
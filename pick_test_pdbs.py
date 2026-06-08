import os
import json
import shutil

# ================= 配置路径 =================
TEST_JSONL = "./nmethyl_data/test_set/test.jsonl"             # 👈 你分好的测试集 JSONL 文件路径
ORIGINAL_PDB_DIR = "./nmethyl_data/raw_pdb"       # 👈 你最初存放所有 PDB 的老文件夹
OUTPUT_DIR = "./monomer_native_pdbs"  # 👈 准备导出的天然单体测试集文件夹
# ===========================================

def main():
    if not os.path.exists(ORIGINAL_PDB_DIR):
        print(f"❌ 找不到原始 PDB 文件夹: {ORIGINAL_PDB_DIR}，请检查路径！")
        return
        
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    success_count = 0
    missing_count = 0

    with open(TEST_JSONL, 'r') as f:
        for line in f:
            if not line.strip():
                continue
            data = json.loads(line)
            name = data.get('name')
            
            if not name:
                continue
            
            # 拼接原始路径和目标路径
            src_pdb = os.path.join(ORIGINAL_PDB_DIR, f"{name}.pdb")
            dst_pdb = os.path.join(OUTPUT_DIR, f"{name}.pdb")
            
            if os.path.exists(src_pdb):
                shutil.copy(src_pdb, dst_pdb)
                success_count += 1
            else:
                print(f"⚠️ 警告: 在原始目录中没找到 {name}.pdb")
                missing_count += 1

    print(f"\n🎉 大功告成！")
    print(f"✅ 成功从原始库中提取并复制了 {success_count} 个测试集天然 PDB 到 {OUTPUT_DIR}")
    if missing_count > 0:
        print(f"❌ 有 {missing_count} 个测试集对应的 PDB 没找到，请检查原始文件夹是否完整。")

if __name__ == "__main__":
    main()
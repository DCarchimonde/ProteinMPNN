import os

# 随便挑一个 4212 字节的设计文件
target_file = "monomer_design_pdbs/Me_1002AAAsresult_proc0002_0072.pdb"

if os.path.exists(target_file):
    print(f"📄 开始读取 {target_file} 的前 20 行内容：\n")
    with open(target_file, "r", encoding="utf-8") as f:
        lines = f.readlines()
        for line in lines[:20]:
            print(line.strip())
    print(f"\n总行数：{len(lines)} 行")
else:
    print("🚨 找不到该测试文件，请检查文件名！")
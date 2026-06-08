import shutil
import os

def main():
    # 你要打包的文件夹（我们刚刚生成的序列都在这里）
    source_dir = "./generated_peptides"
    
    # 压缩包的名字（随便你改）
    output_filename = "peptides_designs_results"
    
    print("📦 姐姐正在为你施展打包魔法...")
    
    # 检查文件夹存不存在
    if not os.path.exists(source_dir):
        print(f"❌ 哎呀，没找到 '{source_dir}' 文件夹，你确定刚才的生成脚本跑完了吗？")
        return

    try:
        # make_archive(压缩包名不带后缀, 格式, 要压缩的文件夹路径)
        shutil.make_archive(output_filename, 'zip', source_dir)
        print(f"✅ 打包大功告成！")
        print(f"🎯 文件已保存为: {output_filename}.zip")
        print("💻 快去 AutoDL 左侧的文件目录里右键下载它吧！")
    except Exception as e:
        print(f"❌ 打包过程中出了点小意外：{e}")

if __name__ == "__main__":
    main()
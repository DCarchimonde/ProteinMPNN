import os
# 强制设置镜像站环境变量
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

from huggingface_hub import snapshot_download

def download_model():
    print(">>> 开始下载 ESMFold 权重 (约 10GB)...")
    print(">>> 正在使用镜像: https://hf-mirror.com")
    
    try:
        snapshot_download(
            repo_id="facebook/esmfold_v1",
            local_dir="esmfold_weights",
            local_dir_use_symlinks=False, # 不使用软链接，直接下载文件
            resume_download=True,         # 支持断点续传
            max_workers=4                 # 多线程下载
        )
        print("\n>>> 下载完成！")
        
        # 检查文件大小
        print(">>> 正在检查文件大小...")
        os.system("du -sh esmfold_weights")
        
        # 检查关键文件是否存在
        bin_path = "esmfold_weights/pytorch_model.bin"
        if os.path.exists(bin_path):
            size = os.path.getsize(bin_path) / (1024 * 1024 * 1024) # GB
            print(f">>> 核心权重文件 pytorch_model.bin 大小: {size:.2f} GB")
            if size < 9:
                print("⚠️ 警告：权重文件似乎太小了，可能未下载完整！")
        else:
            print("❌ 错误：找不到 pytorch_model.bin！")
            
    except Exception as e:
        print(f"\n❌ 下载出错: {e}")
        print("建议：重新运行此脚本，它会从断开的地方继续下载。")

if __name__ == "__main__":
    download_model()
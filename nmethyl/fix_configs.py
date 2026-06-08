import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
from huggingface_hub import hf_hub_download

def fix():
    # 强制重新下载 tokenizer.json，因为上次它可能是0字节
    print(">>> 正在重新下载 tokenizer.json ...")
    hf_hub_download(
        repo_id="facebook/esmfold_v1",
        filename="tokenizer.json",
        local_dir="esmfold_weights",
        force_download=True
    )
    print(">>> 修复完成。")
    os.system("ls -lh esmfold_weights/tokenizer.json")

if __name__ == "__main__":
    fix()
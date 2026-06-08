# 快速评估脚本
import torch
from train_focal_boost import DecoupledProteinMPNN, JSONLDataset, collate_fn, final_evaluation
from torch.utils.data import DataLoader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = DecoupledProteinMPNN().to(device)

# 1. 扩充维度（必须执行，否则报错）
from nmethyl.utils.nmethyl_config import EXTENDED_AA_ALPHABET
model.extend_embedding_and_heads(len(EXTENDED_AA_ALPHABET))

# 2. 加载你刚跑完的权重
ckpt = torch.load("./run_focal_boost/best_model_focal.pt", map_location=device)
model.load_state_dict(ckpt['model_state_dict'])

# 3. 加载测试集
test_loader = DataLoader(JSONLDataset("nmethyl_data/test_set/test.jsonl"), batch_size=8, collate_fn=collate_fn)

# 4. 跑评估
final_evaluation(model, test_loader, device)
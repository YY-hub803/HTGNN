import os
import glob
import json
import torch
import random
import shutil
import argparse
import numpy as np
import pandas as pd
import sys
from pathlib import Path
# 获取当前脚本所在目录
current_dir = Path(__file__).resolve().parent
# 往上退两级，到项目根目录（src/evaluate → src → 项目根目录）
project_root = current_dir.parent.parent
# 把项目根目录加入 Python 搜索路径
sys.path.append(str(project_root))
from src.models import model
from torch.nn.parameter import Parameter
from src.utils.utils import HeteroDataset
from src.train import test
from src.utils import crit
from torch_geometric.loader import DataLoader
from data.load_data import load_water_data,load_se_data,build_edge_index_dict
from data.process import create_sliding_windows

parser = argparse.ArgumentParser()
parser.add_argument('--train',type=bool,default=True,help='Whether to train model')             # 是否训练
parser.add_argument('--seed', type=int, default=42, help='Random seed.')                        # 随机种子
parser.add_argument('--freq',type=str,default='7D',help='Frequency.')                           # 时间频率
parser.add_argument('--model', type=str, default="GnnModel", help='GruHANModel/GnnModel/MeteoModel/SocioEcoModel')    # 模型
parser.add_argument('--epochs', type=int, default=400, help='Number of epochs to train.')       # 训练次数
parser.add_argument('--hidden', type=int, default=16, help='Number of hidden units.')           # 隐藏层
parser.add_argument('--batch', type=int, default=32, help='Batch size.')                        # 批量大小
parser.add_argument('--history', type=int, default=4, help='History len.')                     # 历史序列长度
parser.add_argument('--pred', type=int, default=1, help='Pred len.')                            # 预测长度
parser.add_argument('--num_heads', type=int, default=4, help='Number of head attentions.')      # 多头注意力
parser.add_argument('--num_layers',type=int, default=2, help='Number of layers.')               # 模块层数
parser.add_argument('--dropout', type=float, default=0.1, help='Dropout rate.')                 # 丢弃率
parser.add_argument('--lossFun',type=str,default='RMSE',help='Loss function')                   # 损失函数
parser.add_argument('--lr', type=float, default=1e-3, help='Initial learning rate.')            # 学习率
parser.add_argument('--weights',type=bool,default=False,help='Whether to return attn_weights.')  # 是否返回语义权重
args = parser.parse_args()


def set_seeds(seed_value):
    """Set seeds for reproducibility."""
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed(seed_value)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
# set seeds
set_seeds(args.seed)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MODEL_FACTORY = {
    "GruHANModel": model.GruHANModel,
    'GruModel': model.GruModel,
    "MeteoModel":model.MeteoModel,
    "GnnModel":model.GnnModel,
    "SocioEcoModel":model.SocioEcoModel,
}
Loss_FACTORY = {
    "MSE": crit.MSELoss,
    "MAE": crit.MAELoss,
    "RMSE": crit.RMSELoss,
    "Huber": crit.HuberLoss,
    "MixLoss": crit.MixLoss,
}

dir_model = "%s_B%d_H%d_L%d_NL%d_NH%d_lr%.4f" % (
    args.model,
    args.batch,
    args.hidden,
    args.history,
    args.num_layers,
    args.num_heads,
    args.lr,
)

dir_WQ = r"..\..\data\WQ_data"
dir_SE = r"..\..\data\SE_data"
dir_info = r"..\..\data\info_data"
freq = args.freq
full_data_dir = f'output'
os.makedirs(full_data_dir, exist_ok=True)
output_dir = f"..\..\Random{args.seed}OutPut"
os.makedirs(output_dir, exist_ok=True)

dir_output = os.path.join(output_dir,dir_model)
vis_folder = os.path.join(dir_output, 'visualization')
if not os.path.exists(vis_folder):
    os.makedirs(vis_folder, exist_ok=True)
    print(f"成功创建模型输出文件夹: {vis_folder}")
else:
    print(f"模型输出文件夹已存在: {vis_folder}")
    shutil.rmtree(vis_folder, ignore_errors=True)
    os.makedirs(vis_folder, exist_ok=True)

dir_wq_x = {
    "x_tp": os.path.join(dir_WQ, 'input_yobs_TP.csv'),
}
dir_wq_y = {
    "TP": os.path.join(dir_WQ, 'input_yobs_TP.csv'),
}

dir_se_x = {
    "x_pre": os.path.join(dir_SE, 'input_xforce_prcp.csv'),
    "x_pet": os.path.join(dir_SE, 'input_xforce_pet.csv'),
    "x_temp": os.path.join(dir_SE, 'input_xforce_TEMP.csv'),
    "x_NDVI": os.path.join(dir_SE,'input_xforce_NDVI.csv'),
    "x_Light": os.path.join(dir_SE,'input_xforce_Light.csv'),
}
dir_se_c = {
    "2021": os.path.join(dir_SE, 'input_c_2021.csv'),
    "2022": os.path.join(dir_SE, 'input_c_2022.csv'),
    "2023": os.path.join(dir_SE, 'input_c_2023.csv'),
    "2024": os.path.join(dir_SE, 'input_c_2024.csv'),
}

dir_info = {
    'city_to_water': os.path.join(dir_info, 'city_to_water.csv'),
    'water_to_water': os.path.join(dir_info, 'water_to_water.csv'),
    'city_to_city': os.path.join(dir_info, 'city_to_city.csv'),
    'water_points': os.path.join(dir_info, 'water_points.csv'),
    'city_points': os.path.join(dir_info, 'city_points.csv'),
    'Date_Range': os.path.join(dir_info, 'D_R.csv'),
}
json_path = r'D:\Program\HTGNN\data\dataset\train_stats.json'
# ---------------------- 站点和日期 ----------------------
city_points = pd.read_csv(dir_info['city_points'],encoding='gbk')
water_points = pd.read_csv(dir_info['water_points'])
Date_Range = pd.read_csv(dir_info['Date_Range'])

city_nm = city_points['P_nm'].values
water_nm = water_points['P_nm'].values
num_cities = len(city_nm)
num_water_nodes = len(water_nm)

start_date = Date_Range['start'].min()
end_date = Date_Range['end'].max()

full_date_range = pd.date_range(start=start_date, end=end_date, freq=freq)
date_length = len(full_date_range)
# ---------------------- 加载数据 --------------------------
X, Y = load_water_data(dir_wq_x,dir_wq_y,num_water_nodes,date_length)
X_city,X_city_static = load_se_data(dir_se_x,dir_se_c,num_cities,full_date_range)
edge_index_dict = build_edge_index_dict(dir_info)
with open(json_path, "r", encoding="utf-8") as f:
    train_stats = json.load(f)
stats_tensor = {}
for key, val in train_stats.items():
    stats_tensor[key] = torch.tensor(val, dtype=torch.float32)
# ---------------------------------------------------------
X_norm = (X-stats_tensor['x_mean'])/stats_tensor['x_std']
Y_norm = (Y-stats_tensor['y_mean'])/stats_tensor['y_std']
X_city_norm = (X_city-stats_tensor['x_city_mean'])/stats_tensor['x_city_std']
X_city_static_norm = (X_city_static-stats_tensor['x_static_mean'])/stats_tensor['x_static_std']


X_seq, Y_seq,X_city_seq,X_static_seq = create_sliding_windows(X_norm,Y_norm,X_city_norm,X_city_static_norm,args.history,args.pred)





dataset = HeteroDataset(
    X_seq,
    Y_seq,
    X_city_seq,
    X_static_seq,
    edge_index_dict=edge_index_dict)



sample_data = dataset[0]
metadata = sample_data.metadata()
water_dyn_feat = sample_data['water'].x.shape[-1]
city_dyn_feat = sample_data['city'].x_dyn.shape[-1]
city_static_feat = sample_data['city'].x_static.shape[-1]
print(f"水质动态特征数: {water_dyn_feat}")
print(f"城市动态特征数: {city_dyn_feat}")
print(f"城市静态特征数: {city_static_feat}")
data_loader= DataLoader(dataset, batch_size=args.batch, shuffle=False)

model = MODEL_FACTORY[args.model](water_dyn_feat,city_dyn_feat,city_static_feat,
                    args.num_heads,args.hidden, len(dir_wq_y),
                    args.num_layers,
                    args.dropout,metadata)

# ---------------------- 加载模型用于评估 ----------------------------
model_files = glob.glob(os.path.join(dir_output, "*.pt"))
if not model_files:
    raise FileNotFoundError("未能找到训练保存的模型文件，请检查 train_G 是否成功保存。")
latest_model_path = max(model_files, key=os.path.getmtime)
print(f">>> 加载原始模型进行插补: {latest_model_path}")
model_raw = torch.load(latest_model_path,weights_only=False)
Target_Name = list(dir_wq_y.keys())

y_out, y_true = test.evaluate(
    model_raw, data_loader,
    stats_tensor['y_mean'], stats_tensor['y_std'],
    water_nm, num_water_nodes, Target_Name,
    args.pred, full_data_dir, DEVICE)
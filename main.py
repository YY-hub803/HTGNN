import os
import glob
import torch
import random
import shutil
import numpy as np
import pandas as pd
from src.models import model
from src.utils import crit,vis
from src.train import train,test,infer
from src.utils.utils import HeteroDataset,get_loader
from data.load_data import load_water_data,load_se_data,build_edge_index_dict
from data.process import get_windows


def set_seeds(seed_value):
    """Set seeds for reproducibility."""
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed(seed_value)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
# set seeds
random_seed = 40
set_seeds(random_seed)


hyper_params = {
    "epoch_run": 400,
    "epoch_save": 10,
    "hidden_size": 64,
    "num_heads": 4,
    'history_len': 32,
    'pred_len':1,
    "batch_size":32,
    "num_layers" : 2,
    "drop_rate": 0.5,
    "warmup_epochs":10,
    "base_lr":1e-4,
    "BACKEND":"GruHANModel", # select models    GcnLstmModel/PhysicsSTNNModel
    "lossFun":'RMSE'
}

BACKEND= hyper_params["BACKEND"]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MODEL_FACTORY = {
    "GruHANModel": model.GruHANModel,
}
Loss_FACTORY = {
    "MSE": crit.MSELoss,
    "MAE": crit.MAELoss,
    "RMSE": crit.RMSELoss,
}

dir_model = "%s_B%d_H%d_L%d_P%d_dr%.2f_lr%.4f" % (
    hyper_params['BACKEND'],
    hyper_params['batch_size'],
    hyper_params['hidden_size'],
    hyper_params['history_len'],
    hyper_params['pred_len'],
    hyper_params['drop_rate'],
    hyper_params['base_lr'],
)

dir_WQ = r"data\WQ_data"
dir_SE = r"data\SE_data"
dir_info = r"data\info_data"
freq = '4h'
output_dir = f"Random_OutPut_{freq}"
os.makedirs(output_dir, exist_ok=True)

dir_output = os.path.join(output_dir,dir_model)
vis_folder = os.path.join(dir_output, 'visualization')
if not os.path.exists(vis_folder):
    # 创建文件夹，如果有必要会创建中间目录
    os.makedirs(vis_folder, exist_ok=True)
    print(f"成功创建模型输出文件夹: {vis_folder}")
else:
    print(f"模型输出文件夹已存在: {vis_folder}")
    shutil.rmtree(vis_folder, ignore_errors=True)
    os.makedirs(vis_folder, exist_ok=True)

dir_wq_x = {
    "x_pet": os.path.join(dir_WQ, 'input_xforce_pet.csv'),
    "x_temp": os.path.join(dir_WQ, 'input_xforce_temp.csv'),
    "x_tp": os.path.join(dir_WQ, 'input_yobs_TP.csv'),
    "x_tn": os.path.join(dir_WQ, 'input_yobs_TN.csv'),
    "x_do": os.path.join(dir_WQ, 'input_yobs_DO.csv'),
    "x_TEMP": os.path.join(dir_WQ, 'input_yobs_temp.csv'),
    "x_cod": os.path.join(dir_WQ, 'input_yobs_CODMn.csv'),
}
dir_wq_y = {
    "TP": os.path.join(dir_WQ, 'input_yobs_TP.csv'),
    "TN": os.path.join(dir_WQ, 'input_yobs_TN.csv'),
}

dir_se_x = {
    "x_pre": os.path.join(dir_SE, 'input_xforce_prcp.csv'),
}
dir_se_c = {
    "c_all": os.path.join(dir_SE, 'input_c_all.csv'),
}

dir_info = {
    'city_to_water': os.path.join(dir_info, 'city_to_water.csv'),
    'water_to_water': os.path.join(dir_info, 'water_to_water.csv'),
    'water_points': os.path.join(dir_info, 'water_points.csv'),
    'city_points': os.path.join(dir_info, 'city_points.csv'),
    'Date_Range': os.path.join(dir_info, 'D_R.csv'),
}

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

# ---------------------------------------------------------


# ---------------------- 加载数据 --------------------------
X, Y = load_water_data(dir_wq_x,dir_wq_y,num_water_nodes,date_length)
X_city,X_city_static = load_se_data(dir_se_x,dir_se_c,num_cities,date_length)
edge_index_dict = build_edge_index_dict(dir_info)
# ---------------------------------------------------------

# ---------------------- 创建数据集 -------------------------
train_ratio = 0.6
val_ratio = 0.2
train_end = int(date_length * train_ratio)
val_end = int(date_length * val_ratio)
test_date_range = full_date_range[train_end + val_end+32:,]
Sample_data,data_splits, train_stats=get_windows(X,Y,X_city,X_city_static,
                                                train_ratio,val_ratio,
                                                hyper_params['history_len'],
                                                hyper_params['pred_len'])

Train = HeteroDataset(
    Sample_data['train_x'],
    Sample_data['train_y'],
    Sample_data['train_x_city'],
    Sample_data['X_city_static'],
    edge_index_dict=edge_index_dict)
Val = HeteroDataset(
    Sample_data['val_x'],
    Sample_data['val_y'],
    Sample_data['val_x_city'],
    Sample_data['X_city_static'],
    edge_index_dict=edge_index_dict)
Test = HeteroDataset(
    Sample_data['test_x'],
    Sample_data['test_y'],
    Sample_data['test_x_city'],
    Sample_data['X_city_static'],
    edge_index_dict=edge_index_dict)
# ---------------------------------------------------------

# ---------------------- 实例化模型和损失函数 -------------------------
# 所有数据的输入维度与图结构不变
sample_data = Train[0]
metadata = sample_data.metadata()
water_dyn_feat = sample_data['water'].x.shape[-1]
city_dyn_feat = sample_data['city'].x_dyn.shape[-1]
city_static_feat = sample_data['city'].x_static.shape[-1]
print(f"水质动态特征数: {water_dyn_feat}")
print(f"城市动态特征数: {city_dyn_feat}")
print(f"城市静态特征数: {city_static_feat}")
model = MODEL_FACTORY[BACKEND](water_dyn_feat,city_dyn_feat,city_static_feat,
                    hyper_params['num_heads'],hyper_params['hidden_size'], len(dir_wq_y),
                    hyper_params['num_layers'],hyper_params['pred_len'],
                    hyper_params['drop_rate'],metadata)
print(f"模型参数: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
lossFun = Loss_FACTORY[hyper_params['lossFun']]()
# ------------------------------------------------------------------

# ---------------------- 创建Loder并训练模型 -------------------------
train_loader ,val_loader,test_loader= get_loader(Train,Val,Test,hyper_params['batch_size'])
best_model = train.train(
    model,train_loader, val_loader,lossFun,
    hyper_params['epoch_run'],
    hyper_params['base_lr'],
    dir_output,device)

# ------------------------------------------------------------------

# ---------------------- 加载模型用于评估 ----------------------------
model_files = glob.glob(os.path.join(dir_output, "*.pt"))
if not model_files:
    raise FileNotFoundError("未能找到训练保存的模型文件，请检查 train_G 是否成功保存。")
latest_model_path = max(model_files, key=os.path.getmtime)
print(f">>> 加载原始模型进行插补: {latest_model_path}")
model_raw = torch.load(latest_model_path,weights_only=False)
Target_Name = list(dir_wq_y.keys())
y_out, y_true = test.evaluate(
    model_raw, test_loader,
    train_stats['y_mean'], train_stats['y_std'],
    water_nm,num_water_nodes,Target_Name,
    hyper_params['pred_len'],dir_output,device)

# ------------------------------------------------------------------

# ---------------------- 可视化测试集的效果 ---------------------------

if 'y_out' in locals():
    print("------------------------ 生成可视化图表 ------------------------------")
    vis_mapping = {
        "DO": lambda: vis.vis_filled(y_true['DO'], y_out['DO'], test_date_range, vis_folder, "DO"),
        "TP": lambda: vis.vis_filled(y_true['TP'], y_out['TP'], test_date_range, vis_folder, "TP"),
        "NTU": lambda: vis.vis_filled(y_true['NTU'], y_out['NTU'], test_date_range, vis_folder, "NTU"),
        "TN": lambda: vis.vis_filled(y_true['TN'], y_out['TN'], test_date_range, vis_folder, "TN"),
        "EC": lambda: vis.vis_filled(y_true['EC'], y_out['EC'], test_date_range, vis_folder, "EC")
    }
    for var_name, vis_func in vis_mapping.items():
        if var_name in Target_Name:
            vis_func()  # 执行对应变量的可视化函数
            print(f"已执行 {var_name} 的可视化，保存至 {vis_folder}")


# ------------------------------------------------------------------
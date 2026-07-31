import os
import glob
import torch
from src.utils import vis
from src.train import train,test
from src.utils.get_set_loader import get_loader,get_set
from src.utils.creat_window import get_windows
from src.utils.utils import set_seeds,set_all
from configs import config as cfg
from src.utils.load_data import load_data,load_siteInfo,load_timeSeries


TRAIN_RATIO = 0.7
VAL_RATIO = 0.15

set_seeds(cfg.get("train_config")["seed"])
trainModel,Loss,DEVICE,DIR_MODEL,DIR_OUTPUT,VIS_FOLDER = set_all(cfg)
# ---------------------Load Data------------------------
water_nm,city_nm, num_cities,num_water_nodes = load_siteInfo(cfg)
full_date_range, DATE_LENGTH = load_timeSeries(cfg)
X, Y, X_city, X_city_static, edge_attr, edge_index_dict = load_data(cfg,full_date_range,DATE_LENGTH,num_cities)

# ---------------------- 创建数据集 -------------------------
TRAIN_END = int(DATE_LENGTH * TRAIN_RATIO)
VAL_END = int(DATE_LENGTH * VAL_RATIO)
test_date_range = full_date_range[TRAIN_END + VAL_END + cfg.get("train_config")['history']:,]

window_input = (X,Y,X_city,X_city_static,edge_attr,TRAIN_RATIO,VAL_RATIO,)

Sample_data,data_splits, train_stats=get_windows(window_input,cfg)


Train,Val,Test = get_set(Sample_data,edge_index_dict)
torch.save(Test,r'data\dataset\Test_dataset.pt')        # 保存Test数据集用于explain
# ---------------------------------------------------------

# ---------------------- 实例化模型和损失函数 -------------------------
# 所有数据的输入维度与图结构不变
sample_data = Train[0]
metadata = sample_data.metadata()
water_dyn_feat = sample_data['water'].x.shape[-1]
city_dyn_feat = sample_data['city'].x_dyn.shape[-1]
city_static_feat = sample_data['city'].x_static.shape[-1]
edge_feat_dims = {}
for edge_type in sample_data.edge_types:
    if 'edge_attr' in sample_data[edge_type]:
        edge_feat_dims[edge_type] = sample_data[edge_type].edge_attr.shape[-1]

model = trainModel(
    water_dyn_feat=water_dyn_feat,
    city_dyn_feat=city_dyn_feat,
    city_static_feat=city_static_feat,
    num_heads=cfg.get("train_config")['num_heads'],
    hidden_size=cfg.get("train_config")['hidden'],
    output_size=len(cfg.get("wq_y")),
    num_layers=cfg.get("train_config")['num_layers'],
    drop_rate=cfg.get("train_config")['dropout'],
    metadata=metadata,
    edge_attr_dim=edge_feat_dims,
    edge_index_dict=edge_index_dict)
print(f"模型参数: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
lossFun = Loss()
# ------------------------------------------------------------------

# ---------------------- 创建Loder并训练模型 -------------------------
train_loader ,val_loader,test_loader= get_loader(Train,Val,Test,cfg.get("train_config")['batch'])
if cfg.get("train_config")['train']:
    best_model = train.train(
        model,train_loader, val_loader,lossFun,
        cfg.get("train_config")['epochs'],
        cfg.get("train_config")['lr'],
        DIR_OUTPUT,DEVICE)

# ------------------------------------------------------------------

# ---------------------- 加载模型用于评估 ----------------------------
model_raw = test.load_latest_model(DIR_OUTPUT)
Target_Name = list(cfg.get("wq_y").keys())

if cfg.get("train_config")['weights']:
    y_out,y_true,weights = test.evaluate(
        model_raw, test_loader,
        train_stats['y_mean'], train_stats['y_std'],
        water_nm,num_water_nodes,Target_Name,
        cfg.get("train_config")['pred'],DIR_OUTPUT,DEVICE,cfg.get("train_config")["weights"])
else:
    y_out, y_true = test.evaluate(
        model_raw, test_loader,
        train_stats['y_mean'], train_stats['y_std'],
        water_nm,num_water_nodes,Target_Name,
        cfg.get("train_config")['pred'],DIR_OUTPUT,DEVICE)

# ------------------------------------------------------------------

# ---------------------- 可视化测试集的效果 ---------------------------

if 'y_out' in locals():
    print("------------------------ 生成可视化图表 ------------------------------")
    vis_mapping = {
        "TP": lambda: vis.vis_filled(y_true['TP'], y_out['TP'], test_date_range, VIS_FOLDER, "TP"),
        "TN": lambda: vis.vis_filled(y_true['TN'], y_out['TN'], test_date_range, VIS_FOLDER, "TN"),
    }
    for var_name, vis_func in vis_mapping.items():
        if var_name in Target_Name:
            vis_func()  # 执行对应变量的可视化函数
            print(f"已执行 {var_name} 的可视化，保存至 {VIS_FOLDER}")


# ------------------------------------------------------------------
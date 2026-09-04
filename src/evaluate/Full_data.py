import os
import glob
import json
import torch
from src.utils.get_set_loader import HeteroDataset
from src.train import test
from torch_geometric.loader import DataLoader
from configs import config as cfg
from src.utils.utils import set_seeds,set_all,check_folder
from src.utils.load_data import load_data,load_siteInfo,load_timeSeries


json_path = r'D:\Program\HTGNN\data\dataset\train_stats.json'
# ---------------------- 站点和日期 ----------------------

set_seeds(cfg.get("train_config")["seed"])
trainModel,Loss,DEVICE,DIR_MODEL,MODEL_PATH,_ = set_all(cfg)
OUTPUT_DIR = os.path.join(os.getcwd(),"output18")
check_folder(OUTPUT_DIR)
DIR_OUTPUT = os.path.join(OUTPUT_DIR,DIR_MODEL)
check_folder(DIR_OUTPUT)
VIS_FOLDER = os.path.join(DIR_OUTPUT, 'visualization')
check_folder(VIS_FOLDER)
# ---------------------Load Data------------------------
water_nm,city_nm, num_cities,num_water_nodes = load_siteInfo(cfg)
full_date_range, DATE_LENGTH = load_timeSeries(cfg)
X, Y, X_city, X_city_static, edge_attr_dict, edge_index_dict = load_data(cfg,full_date_range,DATE_LENGTH,num_cities)

# ---------------------- 加载数据 --------------------------

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
edge_attr_norm = (edge_attr_dict[('water', 'flows_to', 'water')]-stats_tensor['edge_attr_mean'])/stats_tensor['edge_attr_std']
edge_static_norm = (edge_attr_dict[('city', 'impact', 'water')]-stats_tensor['edge_staticAttr_mean'])/stats_tensor['edge_staticAttr_std']

window_input = (X_norm,Y_norm,X_city_norm,X_city_static_norm,edge_attr_norm)
window_size, pred_len = cfg.get("train_config")['history'],cfg.get("train_config")['pred']

def create_sliding_windows(data_tple,window_size,pred_len):
    X = data_tple[0]
    Y = data_tple[1]
    x_city = data_tple[2]
    x_static = data_tple[3]
    edge_attr = data_tple[4]
    xs, ys, xs_city, xs_static, edge_attr_seq = [], [], [], [], []
    T = X.size(1)
    for t in range(T - window_size):
        xs.append(X[:,t:t+window_size,:])
        ys.append(Y[:,t+window_size,:])
        xs_city.append(x_city[:,t:t+window_size,:])
        xs_static.append(x_static[:,t:t+window_size,:])
        edge_attr_seq.append(edge_attr[:,t:t+window_size,:])
    X_seq = torch.stack(xs)
    Y_seq = torch.stack(ys)
    X_city_seq = torch.stack(xs_city)
    X_static_seq = torch.stack(xs_static)
    edge_attr_seq = torch.stack(edge_attr_seq)
    return X_seq, Y_seq,X_city_seq,X_static_seq,edge_attr_seq
X_seq, Y_seq,X_city_seq,X_static_seq,edge_attr_seq = create_sliding_windows(window_input,window_size,pred_len)



dataset = HeteroDataset(
    X_seq,
    Y_seq,
    X_city_seq,
    X_static_seq,
    edge_attr_seq,
    edge_static_norm,
    edge_index_dict=edge_index_dict,
    )
torch.save(dataset,r'D:\Program\HTGNN\data\dataset\ALL_dataset.pt')


sample_data = dataset[0]
metadata = sample_data.metadata()
water_dyn_feat = sample_data['water'].x.shape[-1]
city_dyn_feat = sample_data['city'].x_dyn.shape[-1]
city_static_feat = sample_data['city'].x_static.shape[-1]
edge_feat_dims = {}
for edge_type in sample_data.edge_types:
    if 'edge_attr' in sample_data[edge_type]:
        edge_feat_dims[edge_type] = sample_data[edge_type].edge_attr.shape[-1]

data_loader= DataLoader(dataset, batch_size=cfg.get("train_config")['batch'], shuffle=False,drop_last=True)

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

# ---------------------- 加载模型用于评估 ----------------------------
model_raw = test.load_latest_model(MODEL_PATH)
Target_Name = list(cfg.get("wq_y").keys())

y_out, y_true,weights = test.evaluate(
        model_raw, data_loader,
        stats_tensor['y_mean'], stats_tensor['y_std'],
        water_nm,num_water_nodes,Target_Name,
        cfg.get("train_config")['pred'],DIR_OUTPUT,DEVICE,cfg.get("train_config")["weights"])
import os
import torch
import random
import numpy as np
import pandas as pd
from src.models.model import GruHANModel
from src.utils.utils import HeteroDataset
from data.load_data import load_water_data,build_edge_index_dict
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
random_seed = 250
set_seeds(random_seed)


hyper_params = {
    "epoch_run": 400,
    "epoch_save": 10,
    "hidden_size": 32,
    'history_len': 32,
    'pred_len':1,
    "batch_size":32,
    "num_layers" : 2,
    "drop_rate": 0.3,
    "warmup_epochs":10,
    "base_lr":1e-3,
    "BACKEND":"GcnLstmModel", # select models    GcnLstmModel/PhysicsSTNNModel
    "lossFun":'MAE'
}

dir_WQ = r"data\WQ_data"
dir_SE = r"data\SE_data"
dir_info = r"data\info_data"
freq = '4h'
# output_dir = f"Random_OutPut_{freq}"
# os.makedirs(output_dir, exist_ok=True)
# if hyper_params['pred_len'] == 1:
#     dir_output = os.path.join(output_dir,dir_model)
# else:
#     dir_output = os.path.join(f"Multi{hyper_params['pred_len']}_{output_dir}", dir_model)


dir_wq_x = {

    "x_pet": os.path.join(dir_WQ, 'input_xforce_pet.csv'),
    "x_temp": os.path.join(dir_WQ, 'input_xforce_temp.csv'),
    "x_vp": os.path.join(dir_WQ, 'input_xforce_vp.csv'),
    "x_pre": os.path.join(dir_WQ, 'input_xforce_prcp.csv'),
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
    "c_all": os.path.join(dir_SE, 'input_c_all.csv'),
}

dir_info = {
    'city_to_water': os.path.join(dir_info, 'city_to_water.csv'),
    'water_to_water': os.path.join(dir_info, 'water_to_water.csv'),
    'water_points': os.path.join(dir_info, 'water_points.csv'),
    'city_points': os.path.join(dir_info, 'city_points.csv'),
    'Date_Range': os.path.join(dir_info, 'D_R.csv'),
}

num_cities = 28
num_water_nodes = 14
D_R = pd.read_csv(dir_info['Date_Range'])
start_date = D_R['start'].min()
end_date = D_R['end'].max()
full_date_range = pd.date_range(start=start_date, end=end_date, freq=freq)
date_length = len(full_date_range)

X, Y = load_water_data(dir_wq_x,dir_wq_y,num_water_nodes,date_length)

edge_index_dict = build_edge_index_dict(dir_info)

train_ratio = 0.6
val_ratio = 0.2
Sample_data,data_splits, masks, train_stats=get_windows(X,Y,train_ratio,val_ratio,
                                                        hyper_params['history_len'],
                                                        hyper_params['pred_len'])
Train = HeteroDataset(
    Sample_data['train_x'],
    Sample_data['train_y'],
    train_x_city_seq,
    x_city_static,
    edge_index_dict=edge_index_dict # 图的拓扑字典
)
Val = HeteroDataset(
    Sample_data['val_x'],
    Sample_data['val_y'],
    train_x_city_seq,
    x_city_static,
    edge_index_dict=edge_index_dict
)
Test = HeteroDataset(
    Sample_data['test_x'],
    Sample_data['test_y'],
    train_x_city_seq,
    x_city_static,
    edge_index_dict=edge_index_dict
)

# 初始化模型: 隐藏层维度 64, 输出维度 1 (如回归预测某项污染指标)
# 提取模型所需的参数
in_channels_dict = {
    'city': data['city'].x.size(1),
    'water': data['water'].x.size(1)
}
metadata = data.metadata()

model = GruHANModel(water_dyn_feat,city_dyn_feat,city_static_feat,
                    hidden_size, output_size, num_layers,pred_len,
                    drop_rate,metadata)

# 优化器与损失函数 (以回归任务为例)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
criterion = torch.nn.MSELoss()

# 简单训练循环
model.train()
for epoch in range(100):
    optimizer.zero_grad()

    # 前向传播 (传入字典形式的节点特征和边索引)
    out = model(data.x_dict, data.edge_index_dict)

    # 计算损失 (这里仅对水质节点计算)
    loss = criterion(out, data['water'].y)

    # 反向传播与优化
    loss.backward()
    optimizer.step()

    if epoch % 20 == 0:
        print(f'Epoch: {epoch:>3}, Loss: {loss.item():.4f}')


model.eval()
with torch.no_grad():
    _, semantic_attention = model(data.x_dict, data.edge_index_dict, return_attention=True)

print("\n--- 语义层级注意力权重分析 ---")
# 提取指向 water 节点的关系
edge_types_to_water = [edge for edge in metadata[1] if edge[2] == 'water']
water_semantic_attn = semantic_attention['water']

for i, edge_type in enumerate(edge_types_to_water):
    # 包含多头注意力的均值
    avg_weight = water_semantic_attn[i].mean().item()
    print(f"关系 {edge_type}: 贡献权重 = {avg_weight:.4f}")
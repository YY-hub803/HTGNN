import pandas as pd
import numpy as np
import torch
from torch_geometric.utils import to_undirected
from src.utils.utils import load_timeseries, load_attribute





def load_water_data(dir_x, dir_y, date_length):
    """
    加载并对齐水质节点的特征 (X) 和标签 (Y)
    """
    print("开始加载水质节点数据...")
    x = load_timeseries(dir_x, date_length)
    y = load_timeseries(dir_y, date_length)
    # 3. 转换为 PyTorch Tensors
    X_water = torch.tensor(x, dtype=torch.float32)
    Y_water = torch.tensor(y, dtype=torch.float32)
    return X_water, Y_water


def load_se_data(dir_x,dir_c,num_sites, full_date_range,num_static_features=6):
    date_length = len(full_date_range)
    c_dyn = load_timeseries(dir_x, date_length)

    X_city_static_annual = np.zeros((num_sites,date_length,num_static_features))
    c_static = load_attribute(dir_c)
    for t in range(date_length):
        current_year = str(full_date_range[t].year)
        X_city_static_annual[:,t,:] =  c_static[current_year]

    c_dyn = torch.tensor(c_dyn, dtype=torch.float32)
    c_annual_static = torch.tensor(X_city_static_annual, dtype=torch.float32)
    return c_dyn ,c_annual_static

def load_ea_data(dir,full_date_range):
    date_length = len(full_date_range)
    new_dir = dict(list(dir.items())[:2])

    edge_attr_dict = {}
    e_dyn = torch.from_numpy(load_timeseries(new_dir, date_length)).float()
    e_static = torch.from_numpy(pd.read_csv(dir["e_static"], delimiter=",").to_numpy())[: ,None,:].expand(-1, e_dyn.shape[1], -1).float()
    edge_attr_water = torch.concatenate((e_dyn, e_static), dim=2)
    edge_attr_dict[('water', 'flows_to', 'water')] = edge_attr_water

    edge_city_water = torch.from_numpy(pd.read_csv(dir["e_cityW"], delimiter=",").to_numpy()).float()
    edge_attr_dict[('city', 'impact', 'water')] = edge_city_water


    return edge_attr_dict

def load_edge_index(path,is_undirected=False):
    df = pd.read_csv(path)
    edges_np = df.values[:, :2].T
    edge_index = torch.tensor(edges_np, dtype=torch.long)
    if is_undirected:
        edge_index = to_undirected(edge_index)
    return edge_index

def build_edge_index_dict(dir_edges):
    edge_index_dict = {}
    print("开始加载图的边信息 (拓扑结构)...")
    edge_index_dict[('water', 'flows_to', 'water')] = load_edge_index(
        dir_edges["water_to_water"],
        is_undirected=False)
    edge_city_water = load_edge_index(dir_edges["city_to_water"], is_undirected=False)
    edge_index_dict[('city', 'impact', 'water')] = edge_city_water

    edge_city_city = load_edge_index(dir_edges["city_to_city"], is_undirected=True)
    edge_index_dict[('city', 'impact', 'city')] = edge_city_city

    for edge_type, tensor in edge_index_dict.items():
        print(f"关系 {edge_type} 加载完成: 边数量 = {tensor.shape[1]}")

    return edge_index_dict

#---------------------------------------------------------------------------#
def load_timeSeries(cfg):
    Date_Range = pd.read_csv(cfg.get("info_config")['Date_Range'])
    start_date = Date_Range['start'].min()
    end_date = Date_Range['end'].max()
    full_date_range = pd.date_range(start=start_date, end=end_date, freq=cfg.get("train_config")["freq"])
    date_length = len(full_date_range)
    return full_date_range, date_length

def load_siteInfo(cfg):
    city_points = pd.read_csv(cfg.get("info_config")['city_points'], encoding='gbk')
    water_points = pd.read_csv(cfg.get("info_config")['water_points'])
    city_nm = city_points['P_nm'].values
    water_nm = water_points['P_nm'].values
    num_cities = len(city_nm)
    num_water_nodes = len(water_nm)

    return water_nm,city_nm, num_cities,num_water_nodes

def load_data(cfg,full_date_range,date_length,num_cities):
    """
    加载数据
    """

    X, Y = load_water_data(cfg.get("wq_x"),cfg.get("wq_y"),date_length)

    X_city,X_city_static = load_se_data(cfg.get("se_x"),cfg.get("se_c"),num_cities,full_date_range)

    edge_attr_dict = load_ea_data(cfg.get("ea_config"),full_date_range)

    edge_index_dict = build_edge_index_dict(cfg.get("info_config"))

    return X, Y, X_city, X_city_static, edge_attr_dict, edge_index_dict

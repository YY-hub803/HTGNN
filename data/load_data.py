import pandas as pd
import numpy as np
import torch
from torch_geometric.utils import to_undirected


def load_timeseries(dict_data, num_sites, date_length):
    """Load data_1D from time-series inputs"""
    data_list = []
    for path in dict_data.values():
        loaded_data = pd.read_csv(path, delimiter=",").to_numpy()
        reshaped_data = np.reshape(np.ravel(loaded_data.T), (num_sites, date_length, 1))
        data_list.append(reshaped_data)
    return np.concatenate(data_list, axis=2)

def load_attribute(dict_data):
    """Load data from constant attributes"""
    data_list = [np.loadtxt(path, delimiter=",", skiprows=1) for path in dict_data.values()]
    return np.concatenate(data_list, axis=1)

def load_water_data(dir_x, dir_y,num_sites, date_length):
    """
    加载并对齐水质节点的特征 (X) 和标签 (Y)
    """
    print("开始加载水质节点数据...")
    x = load_timeseries(dir_x, num_sites, date_length)
    y = load_timeseries(dir_y, num_sites, date_length)
    # 3. 转换为 PyTorch Tensors

    X_water = torch.tensor(x, dtype=torch.float32)
    Y_water = torch.tensor(y, dtype=torch.float32)
    print(f"数据加载完成！")
    print(f"水质节点特征 X 形状: {X_water.shape}")
    print(f"水质节点标签 Y 形状: {Y_water.shape}")
    return X_water, Y_water


def load_se_data(dir_x,dir_c,num_sites, date_length):
    c_dyn = load_timeseries(dir_x, num_sites, date_length)
    c_static = load_attribute(dir_c)
    c_dyn = torch.tensor(c_dyn, dtype=torch.float32)
    c_static = torch.tensor(c_static, dtype=torch.float32)
    return c_dyn ,c_static




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
        is_undirected=False
    )

    edge_index_dict[('city', 'impact', 'water')] = load_edge_index(
        dir_edges["city_to_water"],
        is_undirected=False
    )

    for edge_type, tensor in edge_index_dict.items():
        print(f"关系 {edge_type} 加载完成: 边数量 = {tensor.shape[1]}")

    return edge_index_dict


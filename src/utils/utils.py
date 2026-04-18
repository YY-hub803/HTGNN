import torch
from torch_geometric.data import HeteroData


class HeteroDataset(torch.utils.data.Dataset):
    def __init__(self, x_water_seq, y_water_seq, x_city_seq, edge_index_dict):
        self.x_water = x_water_seq
        self.y_water = y_water_seq
        self.x_city = x_city_seq
        self.edge_index_dict = edge_index_dict

    def __len__(self):
        return self.x_water.size(0)

    def __getitem__(self, idx):
        data = HeteroData()
        # data['water'].x -> [num_water_nodes, seq_len, features]
        data['water'].x = self.x_water[idx]
        data['water'].y = self.y_water[idx]

        # data['city'].x -> [num_city_nodes, seq_len, features]
        data['city'].x = self.x_city[idx]

        for edge_type, edge_index in self.edge_index_dict.items():
            data[edge_type].edge_index = edge_index

        return data

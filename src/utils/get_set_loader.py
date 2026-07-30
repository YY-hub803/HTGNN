import torch
from torch_geometric.data import HeteroData
from torch.utils.data import Dataset
from torch_geometric.loader import DataLoader

class HeteroDataset(Dataset):
    def __init__(self, x_water_seq, y_water_seq, x_city_dyn_seq,x_city_static,edge_attr, edge_index_dict):
        # [Samples, num_sites, seq_len, dyn_features]
        self.x_water = x_water_seq
        self.y_water = y_water_seq
        # [Samples, num_city, seq_len, dyn_features]
        self.x_city = x_city_dyn_seq
        # [num_city, static_features]
        self.x_city_static = x_city_static
        self.edge_attr = edge_attr

        self.edge_index_dict = edge_index_dict
        # 提前提取节点数量
        self.num_water_nodes = x_water_seq.size(1)
        self.num_city_nodes = x_city_dyn_seq.size(1)
        self.num_water_edge = len(self.edge_index_dict)

    def __len__(self):
        return self.x_water.size(0)

    def get_edge_attr(self,data,idx):
        for edge_type, edge_index in self.edge_index_dict.items():
            data[edge_type].edge_index = edge_index.clone()
            if edge_type == ('water', 'flows_to', 'water') :
                data[edge_type].edge_attr = self.edge_attr[idx]
            else:
                num_edges = edge_index.size(1)
                data[edge_type].edge_attr = torch.ones(num_edges, 1,
                                                    device=self.x_water.device)
        return data

    def __getitem__(self, idx):
        data = HeteroData()
        data['water'].num_nodes = self.num_water_nodes
        data['city'].num_nodes = self.num_city_nodes

        # data['water'].x -> [num_water, seq_len, features]
        data['water'].x = self.x_water[idx]
        data['water'].y = self.y_water[idx]

        # data['city'].x_dyn -> [num_city, seq_len, features]
        # data['city'].x_static -> [num_city, features]
        data['city'].x_dyn = self.x_city[idx]
        data['city'].x_static = self.x_city_static[idx]
        return self.get_edge_attr(data,idx)


def get_set(Sample_data,edge_index_dict):
    dataset_dict = {}
    for split in ['train','val','test']:
        dataset_dict[split] = HeteroDataset(
            Sample_data[f'{split}_x'],
            Sample_data[f'{split}_y'],
            Sample_data[f'{split}_x_city'],
            Sample_data[f'{split}_x_static'],
            Sample_data[f'{split}_edge_attr'],
            edge_index_dict=edge_index_dict)
    Train,Val,Test = dataset_dict['train'],dataset_dict['val'],dataset_dict['test']
    return Train,Val,Test

def get_loader(Train,Val,Test,batch_size):
    train_loader = DataLoader(Train, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(Val, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(Test, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader
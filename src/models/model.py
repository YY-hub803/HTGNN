import torch
import torch.nn as nn
from GRU import GRULayer
from GNN import HANLayer

class GruHANModel(nn.Module):
    def __init__(self,water_dyn_feat, city_dyn_feat, city_static_feat, hidden_size, out_channels, metadata,drop_rate):
        super(GruHANModel, self).__init__()
        # 1. 动态特征的时序编码器
        self.gru_water = GRULayer(input_size=water_dyn_feat,
                                hidden_size=hidden_size,
                                num_layers=num_layers,
                                pred_len=pred_len,
                                drop_rate=drop_rate)
        self.gru_city = GRULayer(input_size=city_dyn_feat,
                                hidden_size=hidden_size,
                                num_layers=num_layers,
                                pred_len=pred_len,
                                drop_rate=drop_rate)

        self.city_fusion_layer = nn.Linear(hidden_size + city_static_feat, hidden_size)
        self.han = HANLayer(
            in_channels_dict=hidden_size, 
            hidden_size=hidden_size,
            out_size = hidden_size,
            metadata=metadata,
            heads=4
        )


    def forward(self, water_dyn_feat, city_dyn_feat, city_static_feat):




        return self.gru_water()
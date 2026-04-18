import torch
import torch.nn as nn
from GRU import GRULayer
from GNN import HANLayer

class GruHANModel(nn.Module):
    def __init__(self,water_dyn_feat, city_dyn_feat, city_static_feat,
                 hidden_size, output_size, num_layers,pred_len,drop_rate,metadata):
        super(GruHANModel, self).__init__()
        self.pred_len = pred_len
        # 1. 动态特征的时序编码器
        self.gru_water = GRULayer(input_size=water_dyn_feat,
                                hidden_size=hidden_size,
                                num_layers=num_layers,
                                drop_rate=drop_rate)
        self.gru_city = GRULayer(input_size=city_dyn_feat,
                                hidden_size=hidden_size,
                                num_layers=num_layers,
                                drop_rate=drop_rate)
        # 城市节点静态特征处理层------将GRU的输出与原始静态特征拼接
        self.city_fusion_layer = nn.Linear(hidden_size + city_static_feat, hidden_size)

        self.han = HANLayer(
            input_size_dict=hidden_size,
            hidden_size=hidden_size,
            out_size = hidden_size,
            metadata=metadata,
            heads=4
        )
        self.dense = nn.Linear(hidden_size, self.pred_len*output_size)

    def forward(self, batch_data, return_attention=False):

        # 提取水质节点时序特征
        h_water = self.gru_water(batch_data['water'].x)         # [Nodes, hidden_size]
        # 提取城市降雨/气候的时序特征
        h_city_dyn = self.gru_city(batch_data['city'].x_dyn)    # [Nodes, hidden_size]


        # 将动态时序状态与静态 GDP 等拼接
        # batch_data['city'].x_static 形状: [Nodes, static_features]
        city_combined = torch.cat([h_city_dyn, batch_data['city'].x_static], dim=-1)
        h_city_final = torch.relu(self.city_fusion_layer(city_combined))

        # ---构建用于异构图的特征字典 ---
        x_dict_encoded = {
            'water': h_water,
            'city': h_city_final
        }

        # --- 图卷积与预测 ---
        if return_attention:
            out_dict, semantic_attn = self.han(x_dict_encoded, batch_data.edge_index_dict,
                                               return_semantic_attention_weights=True)
        else:
            out_dict = self.han(x_dict_encoded, batch_data.edge_index_dict)
            semantic_attn = None
        prediction = self.dense(torch.relu(out_dict['water']))

        if return_attention:
            return prediction, semantic_attn
        return prediction
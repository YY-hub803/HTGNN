import torch
import torch.nn as nn
import torch.nn.functional as F

from .Prediction_Head import Attention
from .gru_model import GRULayer
from .gnn_model import EAHGTLayer
from torch_geometric.nn import GATConv

class GruEAHGTModel(nn.Module):
    def __init__(self, **kwargs):
        super(GruEAHGTModel, self).__init__()
        # 从 kwargs 提取参数，使用描述性变量名
        water_dyn_feat = kwargs.get('water_dyn_feat')
        city_dyn_feat = kwargs.get('city_dyn_feat')
        city_static_feat = kwargs.get('city_static_feat')
        drop_rate = kwargs.get('drop_rate', 0.3)
        metadata = kwargs.get('metadata')
        num_heads = kwargs.get('num_heads', 4)
        edge_dims = kwargs.get('edge_attr_dim')

        self.edge_index_dict = kwargs.get('edge_index_dict')
        self.output_size = kwargs.get('output_size', 1)  
        self.hidden_size = kwargs.get('hidden_size', 32)

        # ==============================
        # 1. feature encoder
        # ==============================
        self.water_proj = nn.Linear(water_dyn_feat, self.hidden_size)
        self.city_proj = nn.Linear(city_dyn_feat, self.hidden_size)
        # city static encoder
        self.city_map = nn.Sequential(
            nn.Linear(city_static_feat, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.hidden_size))

        # ==============================
        # 2. temporal encoder
        # ==============================
        # city temporal GRU
        self.city_gru_cell = nn.GRUCell(input_size=self.hidden_size, hidden_size=self.hidden_size)
        # water temporal GRU
        self.water_gru_cell = nn.GRUCell(input_size=self.hidden_size, hidden_size=self.hidden_size)

        # ==============================
        # 3. HGT spatial propagation
        # ==============================
        self.conv_layers = nn.ModuleList([  
            EAHGTLayer(
                self.hidden_size,
                metadata,
                edge_dims=edge_dims,
                heads=num_heads)
        ])
        self.gat_conv = GATConv(self.hidden_size, self.hidden_size)

        # ==============================
        # 5. output
        # ==============================
        self.layer_norm = nn.LayerNorm(self.hidden_size)  
        self.predictor = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.ReLU(),
            nn.Dropout(drop_rate),
            nn.Linear(self.hidden_size, self.output_size))

    def get_edge_attr_dict(self, batch_data, time_step):
        edge_attr_dict = {}
        edge_index_dict = {}
        for edge_type, edge_index in self.edge_index_dict.items():
            edge_index_dict[edge_type] = batch_data[edge_type].edge_index
            if edge_type == ('water', 'flows_to', 'water'):
                edge_attr_dict[edge_type] = batch_data[edge_type].edge_attr[:, time_step, :]
            else:
                edge_attr_dict[edge_type] = batch_data[edge_type].edge_attr

        return edge_attr_dict, edge_index_dict

    def forward(self, batch_data, return_attention=False):
        # ==================================
        # input
        # ==================================
        x_water = batch_data['water'].x
        x_city = batch_data['city'].x_dyn
        num_water_nodes, seq_len, _ = x_water.shape  
        num_city_nodes, _, _ = x_city.shape  

        # hidden state
        h_water = torch.zeros(num_water_nodes, self.hidden_size, device=x_water.device)
        h_city = torch.zeros(num_city_nodes, self.hidden_size, device=x_water.device)

        # feature embedding
        x_water_emb = self.water_proj(x_water)
        x_city_emb = self.city_proj(x_city)
        city_static_emb = self.city_map(batch_data['city'].x_static)


        # ==================================
        # temporal loop
        # ==================================
        Attention_list = []
        for time_step in range(seq_len):
            # ------------------------------
            # 0. edge_attr_dict
            # ------------------------------
            edge_attr_dict, edge_index_dict = self.get_edge_attr_dict(batch_data, time_step)

            # ------------------------------
            # 1. city temporal encoding
            # ------------------------------
            city_input = x_city_emb[:, time_step, :]+city_static_emb[:, time_step, :]
            h_city = self.city_gru_cell(city_input, h_city)

            # ------------------------------
            # 2. water current feature
            # ------------------------------
            water_input = x_water_emb[:, time_step, :]
            # previous water state
            water_state = h_water + water_input

            # ------------------------------
            # 3. HGT spatial propagation
            # ------------------------------
            node_feat_dict = {'water': water_state, 'city': h_city}
            hgt_out = node_feat_dict
            for conv_layer in self.conv_layers:
                if return_attention:
                    # 记录边上的注意力权重
                    # tuple([],[]),reshape注意顺序
                    #('water', 'flows_to', 'water')------>water_message
                    #('city', 'impact', 'water')------>city_water_message
                    #('city', 'impact', 'city')------>city_message

                    hgt_out, attention_dict = conv_layer(hgt_out, edge_index_dict, edge_attr_dict, return_attention=True)
                    attn_dict = {}
                    for k, (edge_idx, alpha) in attention_dict.items():
                        attn_dict[k] = (
                            edge_idx.detach().cpu().numpy(),  # [2, E]
                            alpha.detach().cpu().numpy()  # [E, heads]
                        )
                    Attention_list.append(attn_dict)
                else:
                    hgt_out = conv_layer(hgt_out, edge_index_dict, edge_attr_dict)


            # HGT后的water message
            city_water_message = hgt_out['water']
            water_message = self.gat_conv(city_water_message,edge_index_dict[('water', 'flows_to', 'water')],edge_attr_dict[('water', 'flows_to', 'water')])
            h_water = self.water_gru_cell(water_message, h_water)

        # ==================================
        # prediction
        # ==================================
        h_water = self.layer_norm(h_water)
        pred = self.predictor(h_water)
        if return_attention:
            return pred.view(-1, self.output_size), Attention_list
        else:
            return pred.view(-1, self.output_size)

import torch
import torch.nn as nn
import torch.nn.functional as F
from .gru_model import GRULayer
from .gnn_model import HGTLayer
from torch_geometric.nn import GCNConv, GATConv
from torch_geometric_temporal.nn.recurrent import DCRNN
from torch_geometric.utils import add_self_loops

# class GruHANModel(nn.Module):
#     def __init__(self,water_dyn_feat, city_dyn_feat, city_static_feat,num_heads,
#                 hidden_size, output_size, num_layers,drop_rate,metadata,max_time_steps=32):
#         super(GruHANModel, self).__init__()
#         self.ny = output_size
#         self.hidden_size = hidden_size
#         self.water_proj = nn.Linear(
#             water_dyn_feat,
#             hidden_size)
#         self.city_proj = nn.Linear(
#             city_dyn_feat,
#             hidden_size)
#         self.cell_water = nn.GRUCell(input_size=hidden_size,
#                                 hidden_size=hidden_size)
#         self.cell_city = nn.GRUCell(input_size=hidden_size,
#                                 hidden_size=hidden_size)
#         self.city_map = nn.Linear(city_static_feat, hidden_size)
#         self.conv = nn.ModuleList([
#             HGTLayer(in_channels=hidden_size,
#                     out_channels=hidden_size,
#                     metadata=metadata,
#                     heads=num_heads),
#         ])
#         self.gat = GATConv(hidden_size, hidden_size,num_heads,False)
#         self.norm = nn.LayerNorm(hidden_size)
#
#         self.predictor = nn.Sequential(
#             nn.Linear(hidden_size, hidden_size),
#             nn.ReLU(),
#             nn.Dropout(drop_rate),
#             nn.Linear(hidden_size, output_size)
#         )
#
#     def forward(self, batch_data, return_attention=False):
#         # 提取水质节点时序特征
#         x_water = batch_data['water'].x        # [Nodes, hidden_size]
#         x_city = batch_data['city'].x_dyn
#         Nw, T, nF = x_water.shape
#         Nc,T,nF = x_city.shape
#         h = torch.zeros(Nw, self.hidden_size, device=x_water.device)
#         h_c = torch.zeros(Nc, self.hidden_size, device=x_water.device)
#         x_w = self.water_proj(x_water)
#         x_c = self.city_proj(x_city)
#         city_s = self.city_map(batch_data['city'].x_static)
#         attention = list()
#         for t in range(T):
#             # batch_data['city'].x_static 形状: [Nodes, static_features]
#             x_t = x_w[:,t,:]
#             city_t = city_s+x_c[:,t,:]
#             z_w = h+x_t
#             z_c = h_c+city_t
#             x_dict = {
#                 'water': z_w,
#                 'city': z_c}
#             for conv in self.conv:
#                 m = conv(x_dict, batch_data.edge_index_dict)
#                 z_w = m['water']+z_w  # [Nw, H]
#                 z_c = m['city']+z_c
#             if return_attention:
#                 h,(edge_index_with_loops, alpha)= self.gat(z_w,batch_data.edge_index_dict[('water', 'flows_to', 'water')],return_attention_weights=True)
#                 attention.append(alpha)
#             else:
#                 h = self.gat(z_w, batch_data.edge_index_dict[('water', 'flows_to', 'water')])
#             h =self.cell_water(z_w,h)
#             h_c = self.cell_city(z_c,h_c)
#         h = self.norm(h)
#         pred = self.predictor(h)
#
#         if return_attention:
#             return pred.view(-1,self.ny),edge_index_with_loops,torch.stack(attention,dim=1)
#         else:
#             return pred.view(-1,self.ny)

class GruHANModel(nn.Module):
    def __init__(
            self,**kwargs):
        super(GruHANModel, self).__init__()
        water_dyn_feat = kwargs.get('water_dyn_feat')
        city_dyn_feat = kwargs.get('city_dyn_feat')
        city_static_feat = kwargs.get('city_static_feat')
        drop_rate = kwargs.get('drop_rate', 0.3)
        metadata = kwargs.get('metadata')
        num_heads = kwargs.get('num_heads', 4)
        self.ny = kwargs.get('output_size', 1)
        self.hidden_size = kwargs.get('hidden_size', 64)
        # ==============================
        # 1. feature encoder
        # ==============================
        self.water_proj = nn.Linear(water_dyn_feat,self.hidden_size)
        self.city_proj = nn.Linear(city_dyn_feat,self.hidden_size)
        # city static encoder
        self.city_map = nn.Sequential(
            nn.Linear(city_static_feat, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.hidden_size))
        # ==============================
        # 2. temporal encoder
        # ==============================
        # city temporal GRU
        self.cell_city = nn.GRUCell(input_size=self.hidden_size,hidden_size=self.hidden_size)
        # water temporal GRU
        self.cell_water = nn.GRUCell(input_size=self.hidden_size,hidden_size=self.hidden_size)
        # ==============================
        # 3. HGT spatial propagation
        # ==============================
        self.conv = nn.ModuleList([
            HGTLayer(
                in_channels=self.hidden_size,
                out_channels=self.hidden_size,
                metadata=metadata,
                heads=num_heads)])
        # ==============================
        # 4. city-water influence gate
        # ==============================
        self.city_gate = nn.Sequential(
            nn.Linear(self.hidden_size*2, self.hidden_size),
            nn.Sigmoid())
        # ==============================
        # 5. output
        # ==============================
        self.norm = nn.LayerNorm(self.hidden_size)
        self.predictor = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.ReLU(),
            nn.Dropout(drop_rate),
            nn.Linear(self.hidden_size, self.ny))

    def forward(self,batch_data,return_attention=False):
        # ==================================
        # input
        # ==================================
        x_water = batch_data['water'].x
        x_city = batch_data['city'].x_dyn
        Nw,T,_ = x_water.shape
        Nc,T,_ = x_city.shape
        # hidden state
        h_w = torch.zeros(Nw,self.hidden_size,device=x_water.device)
        h_c = torch.zeros(Nc,self.hidden_size,device=x_water.device)
        # feature embedding
        x_w = self.water_proj(x_water)
        x_c = self.city_proj(x_city)
        city_static = self.city_map(batch_data['city'].x_static)
        # ==================================
        # temporal loop
        # ==================================
        for t in range(T):
            # ------------------------------
            # 1. city temporal encoding
            # ------------------------------
            city_input = (x_c[:,t,:]+city_static)
            h_c = self.cell_city(city_input,h_c)
            # ------------------------------
            # 2. water current feature
            # ------------------------------
            water_input = x_w[:,t,:]
            # previous water state
            water_state = h_w + water_input
            # ------------------------------
            # 3. HGT spatial propagation
            # ------------------------------
            x_dict = {'water': water_state,'city': h_c}
            hgt_out = x_dict
            for conv in self.conv:
                hgt_out = conv(hgt_out,batch_data.edge_index_dict)
            # HGT后的water message
            city_water_msg = (hgt_out['water'])
            # ------------------------------
            # 4. city-water gate
            # ------------------------------
            gate = self.city_gate(torch.cat([water_state,city_water_msg],dim=-1))
            # city influence injection
            water_state = (water_state + gate * city_water_msg)
            # ------------------------------
            # 5. water temporal GRU
            # ------------------------------
            h_w = self.cell_water(water_state,h_w)
        # ==================================
        # prediction
        # ==================================
        h_w = self.norm(h_w)
        pred = self.predictor(h_w)
        return pred.view(-1,self.ny)


class GruModel(nn.Module):
    def __init__(self,water_dyn_feat, city_dyn_feat, city_static_feat,num_heads,
                hidden_size, output_size, num_layers,drop_rate,metadata,):
        super(GruModel, self).__init__()
        self.ny = output_size
        # 1. 动态特征的时序编码器
        self.gru_water = GRULayer(input_size=water_dyn_feat,
                                hidden_size=hidden_size,
                                num_layers=num_layers,
                                drop_rate=drop_rate)

        self.predictor = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(drop_rate),
            nn.Linear(hidden_size, output_size)
        )

    def forward(self, batch_data, return_attention=False):
        # 提取水质节点时序特征
        h_water = self.gru_water(batch_data['water'].x)
        prediction = self.predictor(torch.relu(h_water[:,-1,:]))
        return prediction

class SocioEcoModel(nn.Module):
    def __init__(self,water_dyn_feat, city_dyn_feat, city_static_feat,num_heads,
                hidden_size, output_size, num_layers,drop_rate,metadata,max_time_steps=32):
        super(SocioEcoModel, self).__init__()
        self.ny = output_size
        self.hidden_size = hidden_size
        self.water_proj = nn.Linear(
            water_dyn_feat,
            hidden_size)
        self.city_proj = nn.Linear(
            2,
            hidden_size)
        self.cell_water = nn.GRUCell(input_size=hidden_size,
                                hidden_size=hidden_size)
        self.cell_city = nn.GRUCell(input_size=hidden_size,
                                hidden_size=hidden_size)
        self.city_map = nn.Linear(city_static_feat, hidden_size)
        self.conv = nn.ModuleList([
            HGTLayer(in_channels=hidden_size,
                    out_channels=hidden_size,
                    metadata=metadata,
                    heads=num_heads),
        ])
        self.norm = nn.LayerNorm(hidden_size)
        self.predictor = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(drop_rate),
            nn.Linear(hidden_size, output_size)
        )

    def forward(self, batch_data, return_attention=False):
        # 提取水质节点时序特征
        x_water = batch_data['water'].x        # [Nodes, hidden_size]
        x_city = batch_data['city'].x_dyn[:,:,3:]
        Nw, T, nF = x_water.shape
        Nc,nF = batch_data['city'].x_static.shape
        h = torch.zeros(Nw, self.hidden_size, device=x_water.device)
        h_c = torch.zeros(Nc, self.hidden_size, device=x_water.device)
        x_w = self.water_proj(x_water)
        x_c = self.city_proj(x_city)
        city_s = self.city_map(batch_data['city'].x_static)

        for t in range(T):
            # batch_data['city'].x_static 形状: [Nodes, static_features]
            x_t = x_w[:,t,:]
            city_t = city_s + x_c[:, t, :]
            z_w = h+x_t
            z_c = city_t+h_c
            x_dict = {
                'water': z_w,
                'city': z_c}
            for conv in self.conv:
                m = conv(x_dict, batch_data.edge_index_dict)
                z_w = m['water']+z_w  # [Nw, H]
                h_c = m['city']+z_c
            h =self.cell_water(z_w,h)
        h = self.norm(h)
        pred = self.predictor(h)
        return pred.view(-1,self.ny)

class MeteoModel(nn.Module):
    def __init__(self,water_dyn_feat, city_dyn_feat, city_static_feat,num_heads,
                hidden_size, output_size, num_layers,drop_rate,metadata,max_time_steps=32):
        super(MeteoModel, self).__init__()
        self.ny = output_size
        self.hidden_size = hidden_size
        self.water_proj = nn.Linear(
            water_dyn_feat,
            hidden_size)
        self.city_proj = nn.Linear(
            3,
            hidden_size)
        self.cell_water = nn.GRUCell(input_size=hidden_size,
                                hidden_size=hidden_size)
        self.conv = nn.ModuleList([
            HGTLayer(in_channels=hidden_size,
                    out_channels=hidden_size,
                    metadata=metadata,
                    heads=num_heads),
        ])
        self.norm = nn.LayerNorm(hidden_size)
        self.predictor = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(drop_rate),
            nn.Linear(hidden_size, output_size)
        )

    def forward(self, batch_data, return_attention=False):
        # 提取水质节点时序特征
        x_water = batch_data['water'].x        # [Nodes, hidden_size]
        x_city = batch_data['city'].x_dyn[:,:,0:3]
        Nw, T, nF = x_water.shape
        Nc,T,nF = x_city.shape
        h = torch.zeros(Nw, self.hidden_size, device=x_water.device)
        h_c = torch.zeros(Nc, self.hidden_size, device=x_water.device)
        x_w = self.water_proj(x_water)
        x_c = self.city_proj(x_city)
        for t in range(T):
            # batch_data['city'].x_static 形状: [Nodes, static_features]
            x_t = x_w[:,t,:]
            city_t = x_c[:,t,:]
            z_w = h+x_t
            z_c = h_c+city_t
            x_dict = {
                'water': z_w,
                'city': z_c}
            for conv in self.conv:
                m = conv(x_dict, batch_data.edge_index_dict)
                z_w = m['water']+z_w  # [Nw, H]
                z_c = m['city']+z_c
            h =self.cell_water(z_w,h)
        h = self.norm(h)
        pred = self.predictor(h)
        return pred.view(-1,self.ny)

class GnnModel(nn.Module):
    def __init__(self,water_dyn_feat, city_dyn_feat, city_static_feat,num_heads,
                hidden_size, output_size, num_layers,drop_rate,metadata,max_time_steps=32):
        super(GnnModel, self).__init__()
        self.ny = output_size
        self.hidden_size = hidden_size
        self.water_proj = nn.Linear(
            water_dyn_feat,
            hidden_size)
        self.cell_water = nn.GRUCell(input_size=hidden_size,
                                hidden_size=hidden_size)
        self.conv = nn.ModuleList([
            HGTLayer(in_channels=hidden_size,
                    out_channels=hidden_size,
                    metadata=metadata,
                    heads=num_heads),
        ])
        self.norm = nn.LayerNorm(hidden_size)
        self.predictor = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(drop_rate),
            nn.Linear(hidden_size, output_size)
        )

    def forward(self, batch_data, return_attention=False):
        # 提取水质节点时序特征
        x_water = batch_data['water'].x        # [Nodes, hidden_size]
        Nw, T, nF = x_water.shape
        h = torch.zeros(Nw, self.hidden_size, device=x_water.device)
        x_w = self.water_proj(x_water)
        edge_index_dict = {
            ('water', 'flows_to', 'water'):
                batch_data.edge_index_dict[
                    ('water', 'flows_to', 'water')
                ]
        }
        for t in range(T):
            # batch_data['city'].x_static 形状: [Nodes, static_features]
            x_t = x_w[:,t,:]
            z_w = h+x_t
            x_dict = {
                'water': z_w}
            for conv in self.conv:
                m = conv(x_dict, edge_index_dict)
                z_w = m['water']+z_w  # [Nw, H]
            h =self.cell_water(z_w,h)
        h = self.norm(h)
        pred = self.predictor(h)
        return pred.view(-1,self.ny)

import torch
import torch.nn as nn
import torch.nn.functional as F
from .gru_model import GRULayer
from .gnn_model import HANLayer,HGTLayer
from .Prediction_Head import Attention


class GruHANModel(nn.Module):
    def __init__(self,water_dyn_feat, city_dyn_feat, city_static_feat,num_heads,
                hidden_size, output_size, num_layers,drop_rate,metadata,max_time_steps=32):
        super(GruHANModel, self).__init__()
        self.ny = output_size
        self.hidden_size = hidden_size
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
        self.city_fusion = nn.Linear(hidden_size + city_static_feat, hidden_size)
        # ===== 3. HGT=====
        self.hgt = HGTLayer(
            in_channels=hidden_size,
            out_channels=hidden_size,
            metadata=metadata,
            heads=1
        )
        # ===== 4. 时间编码 =====
        self.time_emb = nn.Embedding(max_time_steps, hidden_size)
        # ===== 5. Temporal Attention =====
        self.temporal_attn = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=num_heads,
            batch_first=True
        )
        # ===== 6. Cross-node Attention（关键）=====
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=num_heads,
            batch_first=True
        )
        # ===== 7. FFN（Transformer 标准块）=====
        self.ffn = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 2),
            nn.ReLU(),
            nn.Dropout(drop_rate),
            nn.Linear(hidden_size * 2, hidden_size)
        )

        # ===== 8. Gate（稳定融合）=====
        self.gate = nn.Linear(hidden_size * 2, hidden_size)
        # ===== 9. Norm =====
        self.norm1 = nn.LayerNorm(hidden_size)
        self.norm2 = nn.LayerNorm(hidden_size)
        self.norm3 = nn.LayerNorm(hidden_size)

        # ===== 10. Predictor =====
        self.predictor = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(drop_rate),
            nn.Linear(hidden_size, output_size)
        )

    def causal_mask(self, T, device):
        return torch.triu(
            torch.ones(T, T, device=device) * -1e9,
            diagonal=1
        )

    def forward(self, batch_data, return_attention=False):

        # 提取水质节点时序特征
        h_water = self.gru_water(batch_data['water'].x)         # [Nodes, hidden_size]
        # 提取城市降雨/气候的时序特征
        h_city_dyn= self.gru_city(batch_data['city'].x_dyn)    # [Nodes, hidden_size]
        Nw, T, H = h_water.shape
        Nc = h_city_dyn.shape[0]

        # 将动态时序状态与静态 GDP 等拼接
        water_time_outputs = []
        city_time_outputs = []

        for t in range(T):
            # batch_data['city'].x_static 形状: [Nodes, static_features]
            city_t = torch.cat(
                [h_city_dyn[:, t, :], batch_data['city'].x_static],
                dim=-1
            )
            city_t = F.relu(self.city_fusion(city_t))
            # ---构建用于异构图的特征字典 ---
            x_dict = {
                'water': h_water[:, t, :],
                'city': city_t
            }

            if return_attention:
                out_dict, semantic_attn = self.hgt(x_dict, batch_data.edge_index_dict,
                                                return_attention=True)
            else:
                out_dict = self.hgt(x_dict, batch_data.edge_index_dict)
                semantic_attn = None
            water_time_outputs.append(out_dict['water'])  # [Nw, H]
            city_time_outputs.append(out_dict['city'])    # [Nc, H]
        # =========================
        # Step 3: 拼接时间维
        # =========================
        h_water_time = torch.stack(water_time_outputs, dim=1)  # [Nw, T, H]
        h_city_time = torch.stack(city_time_outputs, dim=1)  # [Nc, T, H]
        # =========================
        # Step 4: 时间编码
        # =========================
        time_ids = torch.arange(T, device=h_water_time.device)
        time_emb = self.time_emb(time_ids)  # [T, H]
        h_water_time = h_water_time + time_emb.unsqueeze(0)
        h_city_time  = h_city_time  + time_emb.unsqueeze(0)

        # =========================
        # Step 5: Temporal Attention
        # =========================
        mask = self.causal_mask(T, h_water_time.device)

        h_water_temp, _ = self.temporal_attn(
            h_water_time, h_water_time, h_water_time, attn_mask=mask
        )
        h_water_temp = self.norm1(h_water_temp + h_water_time)

        h_city_temp, _ = self.temporal_attn(
            h_city_time, h_city_time, h_city_time, attn_mask=mask
        )
        h_city_temp = self.norm1(h_city_temp + h_city_time)

        # 取最后时刻（或 mean）
        h_last = h_water_temp[:, -1, :]
        h_mean = h_water_temp.mean(dim=1)
        h_max = torch.logsumexp(h_water_temp, dim=1)
        h_water_final = h_last + h_mean + h_max # [Nw, H]
        h_city_final  = h_city_temp[:, -1, :]   # [Nc, H]

        # =========================
        # Step 6: Cross-node Attention（支持 Nw ≠ Nc）
        # =========================
        mask_cross = torch.full((Nw, Nc), -1e9, device=h_last.device)
        # 屏蔽掉不符合实际的边
        if ('water', 'impacted_by', 'city') in batch_data.edge_index_dict:
            edge = batch_data.edge_index_dict[('water', 'impacted_by', 'city')]
            mask_cross[edge[0], edge[1]] = 0
        row_has_edge = (mask_cross == 0).any(dim=1)
        mask_cross[~row_has_edge] = 0  # fallback 到全连接
        # water queries, city keys
        q = h_water_final.unsqueeze(0)  # [1, Nw, H]
        k = h_city_final.unsqueeze(0)   # [1, Nc, H]
        v = h_city_final.unsqueeze(0)

        h_cross, _ = self.cross_attn(q, k, v,attn_mask=mask_cross)  # [1, Nw, H]
        h_cross = h_cross.squeeze(0)                                # [Nw, H]

        # ===== Gate 融合 =====
        gate = torch.sigmoid(
            self.gate(torch.cat([h_water_final, h_cross], dim=-1))
        )
        h_fused = gate * h_cross + (1 - gate) * h_water_final

        h = self.norm2(h_fused)

        # =========================
        # Step 7: FFN
        # =========================
        h = h + self.ffn(h)
        h = self.norm3(h)

        # =========================
        # Step 8: 预测
        # =========================
        pred = self.predictor(h)

        if return_attention:
            return pred.view(-1,self.ny), semantic_attn
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

        self.dense = nn.Linear(hidden_size, output_size)

    def forward(self, batch_data, return_attention=False):
        # 提取水质节点时序特征
        h_water = self.gru_water(batch_data['water'].x)         # [Nodes, hidden_size]

        prediction = self.dense(torch.relu(h_water[:,-1,:]))
        return prediction
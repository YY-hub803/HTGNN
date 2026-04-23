import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class Attention(nn.Module):
    def __init__(self, hidden_size, num_heads=4, dropout=0.1):
        super().__init__()

        self.mha = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True  # 这样输入形状就是 [Nodes, Seq_Len, Hidden]
        )
        self.query_vector = nn.Parameter(torch.randn(1, 1, hidden_size))
        self.norm = nn.LayerNorm(hidden_size)

    def forward(self, x):
        """
        x: [Nodes, Seq_Len, Hidden_Size] (GRU 的输出)
        """
        nodes = x.size(0)

        query = self.query_vector.expand(nodes, -1, -1)

        # attn_output 形状: [Nodes, 1, Hidden]
        attn_output, attn_weights = self.mha(query, x, x)

        out = attn_output.squeeze(1)

        return out, attn_weights











class GLU(nn.Module):
    def __init__(self, input_size):
        super(GLU, self).__init__()
        self.fc = nn.Linear(input_size, input_size * 2)

    def forward(self, x):
        x = self.fc(x)
        x1, x2 = x.chunk(2, dim=-1)
        return x1 * torch.sigmoid(x2)


class GRN(nn.Module):
    def __init__(self, input_size, hidden_size, dropout=0.1):
        super(GRN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.elu = nn.ELU()
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.glu = GLU(hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.skip_proj = nn.Linear(input_size, hidden_size) if input_size != hidden_size else nn.Identity()

    def forward(self, x):
        residual = self.skip_proj(x)
        x = self.fc1(x)
        x = self.elu(x)
        x = self.fc2(x)
        x = self.dropout(x)
        x = self.glu(x)
        return self.layer_norm(x + residual)


class TFTDecoderHead(nn.Module):
    def __init__(self, hidden_size, pred_len, out_dim, num_heads=4, dropout=0.1):
        super(TFTDecoderHead, self).__init__()
        self.pred_len = pred_len
        self.out_dim = out_dim

        # 1. 历史序列的特征提纯
        self.historical_grn = GRN(hidden_size, hidden_size, dropout)

        # 2. 多头自注意力机制 (提取全局时序依赖)
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=num_heads,
            batch_first=True,
            dropout=dropout)
        self.attn_layer_norm = nn.LayerNorm(hidden_size)

        self.future_queries = nn.Parameter(torch.randn(1, pred_len, hidden_size))

        self.output_grn = GRN(hidden_size, hidden_size, dropout)

        self.final_proj = nn.Linear(hidden_size, out_dim)

    def forward(self, st_features):
        # st_features: [BN, T, H]
        BN, T, H = st_features.shape
        #  [B*N, T, H]
        v = self.historical_grn(st_features)  # Value & Key

        q = self.future_queries.expand(BN, -1, -1).reshape(BN, self.pred_len, H)
        # Multi-Head Attention: Q=未来预测步, K=V=历史序列
        attn_out, attn_weights = self.attention(q, v, v)

        attn_out = self.attn_layer_norm(attn_out + q)

        out = self.output_grn(attn_out)

        out = self.final_proj(out)
        return out.reshape(BN, self.pred_len, self.out_dim)


import torch.nn as nn
from torch_geometric.nn import HANConv,HGTConv


class HANLayer(nn.Module):
    def __init__(self, input_size_dict, hidden_size, out_size, metadata,heads=4):
        """
        in_channels_dict: 字典，指定每种节点的输入特征维度，如 {'city': 5, 'water': 8}
        metadata: 图的元信息，格式为 (node_types, edge_types)
        """
        super().__init__()

        self.han = HANConv(
            in_channels=input_size_dict,
            out_channels=hidden_size,
            metadata=metadata,
            heads=4             # 注意力头数
        )

    def forward(self, x_dict, edge_index_dict, return_attention=False):

        if return_attention:
            out_dict, semantic_attn = self.han(
                x_dict,
                edge_index_dict,
                return_semantic_attention_weights=True
            )
        else:
            out_dict = self.han(x_dict, edge_index_dict)
            semantic_attn = None

        if return_attention:
            return out_dict, semantic_attn
        return out_dict

class HGTLayer(nn.Module):
    def __init__(self, in_channels, out_channels, metadata, heads=4, num_layers=1, dropout=0.1):
        """
        in_channels: int 或 dict（推荐 int，你当前是统一 hidden_size）
        metadata: (node_types, edge_types)
        """

        super().__init__()

        self.node_types, self.edge_types = metadata

        self.convs = nn.ModuleList()
        for _ in range(num_layers):
            conv = HGTConv(
                in_channels=in_channels,
                out_channels=out_channels,
                metadata=metadata,
                heads=heads,

            )
            self.convs.append(conv)

        self.norms = nn.ModuleDict({
            node_type: nn.LayerNorm(out_channels)
            for node_type in self.node_types
        })

        self.dropout = nn.Dropout(dropout)

    def forward(self, x_dict, edge_index_dict, return_attention=False):
        """
        x_dict: {'water': [Nw, H], 'city': [Nc, H]}
        """

        for conv in self.convs:
            out_dict = conv(x_dict, edge_index_dict)

            # ===== residual + norm（关键）=====
            new_dict = {}
            for node_type in x_dict.keys():
                h = out_dict[node_type]

                # residual
                h = h + x_dict[node_type]

                # norm
                h = self.norms[node_type](h)

                # dropout
                h = self.dropout(h)

                new_dict[node_type] = h

            x_dict = new_dict

        if return_attention:

            return x_dict, None

        return x_dict
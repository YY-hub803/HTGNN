import torch.nn as nn
from torch_geometric.nn import HANConv



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
        # 开启 return_semantic_attention_weights 即可获取语义级的注意力权重
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

import torch
import torch.nn as nn
from torch_geometric.nn import HGTConv
from torch_geometric.utils import softmax

class HGTLayer(nn.Module):
    def __init__(self, in_channels, out_channels, metadata, heads=4, num_layers=2, dropout=0.1):

        super().__init__()
        self.node_types, self.edge_types = metadata
        self.conv = HGTConv(
                in_channels=in_channels,
                out_channels=out_channels,
                metadata=metadata,
                heads=heads,
                )
        self.norms = nn.ModuleDict({
            node_type: nn.LayerNorm(out_channels)
            for node_type in self.node_types
        })
        self.dropout = nn.Dropout(dropout)
    def forward(self, x_dict, edge_index_dict, return_attention=False):
        """
        x_dict: {'water': [Nw, H], 'city': [Nc, H]}
        """
        if return_attention:
            out_dict, semantic_attn = self.conv(
                x_dict,
                edge_index_dict,
            )
        else:
            out_dict = self.conv(x_dict, edge_index_dict)
            semantic_attn = None
        new_dict = {}
        for node_type in out_dict.keys():
            h = out_dict[node_type]
            # residual
            h = h + x_dict[node_type]
            # norm
            h = self.norms[node_type](h)
            # dropout
            h = self.dropout(h)
            new_dict[node_type] = h
        if return_attention:
            return new_dict, semantic_attn
        return new_dict

class EAHGTLayer(nn.Module):
    """
    Edge-Aware Heterogeneous Graph Transformer Layer (EA-HGT)

    这是一个考虑边属性的异构图注意力层，核心特点：
    1. 显式编码边属性并将其融入注意力计算
    2. 每种边类型有独立的关系变换参数
    3. 支持多头注意力机制
    4. 包含残差连接和层归一化

    """
    def __init__(self,hidden_dim,metadata,edge_dims,heads=4,dropout=0.1):
        """
        初始化边感知异构图注意力层

        Args:
            hidden_dim (int): 隐藏层维度
            metadata (tuple): 图元数据，包含(node_types, edge_types)
            edge_dims (dict): 每种边类型的特征维度，格式: {edge_type: dim}
            heads (int): 多头注意力的头数，default=4
            dropout (float): Dropout概率，default=0.1
        """

        super().__init__()
        self.node_types, self.edge_types = metadata
        self.hidden_dim = hidden_dim
        self.heads = heads
        self.d_k = hidden_dim // heads
        assert hidden_dim % heads == 0
        # ==========================
        # node type projection
        # ==========================
        self.q_linears = nn.ModuleDict()
        self.k_linears = nn.ModuleDict()
        self.v_linears = nn.ModuleDict()
        for ntype in self.node_types:
            self.q_linears[ntype] = nn.Linear(hidden_dim,hidden_dim)
            self.k_linears[ntype] = nn.Linear(hidden_dim,hidden_dim)
            self.v_linears[ntype] = nn.Linear(hidden_dim,hidden_dim)
        # ==========================
        # 边属性编码器 (Edge Attribute Encoder)
        # 将边特征编码为隐藏维度向量
        # ==========================
        self.edge_encoder = nn.ModuleDict()
        for etype in self.edge_types:
            key=self.edge_key(etype)
            self.edge_encoder[key]=nn.Sequential(
                nn.Linear(edge_dims[etype],hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim,hidden_dim)
            )
        # ==========================
        # 关系变换参数 (Relation-Specific Transformations)
        # 每种边类型有独立的注意力和消息变换矩阵
        # ==========================
        self.relation_att = nn.ParameterDict()
        self.relation_msg = nn.ParameterDict()
        for etype in self.edge_types:
            key=self.edge_key(etype)
            self.relation_att[key]=nn.Parameter(torch.Tensor(heads,self.d_k,self.d_k))
            self.relation_msg[key]=nn.Parameter(torch.Tensor(heads,self.d_k,self.d_k))
            nn.init.xavier_uniform_(self.relation_att[key])
            nn.init.xavier_uniform_(self.relation_msg[key])
        # ==========================
        # output
        # ==========================
        self.out_linear=nn.Linear(hidden_dim,hidden_dim)
        self.norm=nn.ModuleDict({n:nn.LayerNorm(hidden_dim)for n in self.node_types})
        self.dropout=nn.Dropout(dropout)
    def edge_key(self,etype):
        """
        将边类型元组转换为字符串键

        Args:
            etype (tuple): 边类型，格式如 ('water', 'flows_to', 'water')

        Returns:
            str: 拼接后的键名，如 "water__flows_to__water"
        """

        return "__".join(etype)
    def forward(self,x_dict,edge_index_dict,edge_attr_dict,return_attention=False):
        """
        Args:
            x_dict (dict): 节点特征字典，格式: {node_type: [N, hidden_dim]}
            edge_index_dict (dict): 边索引字典，格式: {edge_type: [2, E]}
            edge_attr_dict (dict): 边属性字典，格式: {edge_type: [E, edge_dim]}
            return_attention (bool): 是否返回注意力权重，default=False

        Returns:
            dict: 更新后的节点特征字典
            dict (optional): 注意力权重字典（仅当return_attention=True时返回）
        """

        device=list(x_dict.values())[0].device
        out_dict={}
        att_dict={}
        # 初始化
        for ntype,x in x_dict.items():
            out_dict[ntype]=torch.zeros_like(x)
        # ==================================
        # 遍历每种边类型进行消息传递
        # ==================================
        for etype,edge_index in edge_index_dict.items():
            src_type,_,dst_type=etype           # 解析边类型：源类型、关系、目标类型
            key=self.edge_key(etype)            # 获取边类型的字符串键
            src=edge_index[0]                   # 源节点索引
            dst=edge_index[1]                   # 目标节点索引

            # 获取源节点和目标节点的特征
            x_src=x_dict[src_type]
            x_dst=x_dict[dst_type]
            # ------------------------------
            # Q/K/V 投影
            # 目标节点生成查询，源节点生成键和值
            # ------------------------------

            Q=self.q_linears[dst_type](x_dst)
            K=self.k_linears[src_type](x_src)
            V=self.v_linears[src_type](x_src)
            # ------------------------------
            # 边属性编码
            # 将边特征转换为隐藏空间表示
            # ------------------------------

            edge_emb=self.edge_encoder[key](edge_attr_dict[etype])

            # ------------------------------
            # 3. 多头注意力重塑
            # 将特征张量重塑为多头形式
            # ------------------------------
            Q=Q.view(-1,self.heads,self.d_k)
            K=K.view(-1,self.heads,self.d_k)
            V=V.view(-1,self.heads,self.d_k)


            edge_emb=edge_emb.view(-1,self.heads,self.d_k)

            # ------------------------------
            # 4. 注意力分数计算（核心）
            # 双重注意力：节点-节点 + 节点-边
            # ------------------------------
            q=Q[dst]                # [E, heads, d_k]  目标节点的查询向量
            k=K[src]                # [E, heads, d_k]  源节点的键向量
            e=edge_emb              # [E, heads, d_k]  边属性嵌入
            # 节点-节点注意力分数
            score=(q*k).sum(-1)
            # 节点-边注意力分数
            edge_score=(q*e).sum(-1)
            # 合并并归一化
            score=(score + edge_score)/self.d_k**0.5    # [E, heads]
            alpha=torch.zeros_like(score)
            # 按目标节点分组进行softmax
            alpha=softmax(score,dst)
            # ------------------------------
            # 消息传递与聚合
            # ------------------------------
            msg=V[src]+e                                        # [E, heads, d_k]  源节点消息 + 边属性
            msg=msg*alpha.unsqueeze(-1)                         # [E, heads, d_k]  注意力权重加权
            msg=msg.reshape(msg.shape[0],self.hidden_dim)       # [E, hidden_dim]  合并多头
            msg=self.out_linear(msg)                            # [E, hidden_dim]  输出投影

            # 按目标节点聚合消息
            out_dict[dst_type].index_add_(0,dst,msg)
            if return_attention:
                att_dict[etype]=(edge_index,alpha.detach())

        # ==================================
        # 残差连接与正则化
        # ==================================
        result={}
        for ntype,x in x_dict.items():
            h=out_dict[ntype]               # 聚合后的特征
            h=h+x                           # 残差连接
            h=self.norm[ntype](h)           # 层归一化
            h=self.dropout(h)
            result[ntype]=h
        if return_attention:
            return result,att_dict
        return result
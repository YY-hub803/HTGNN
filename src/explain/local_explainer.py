import torch
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from torch_geometric.data import HeteroData
from captum.attr import IntegratedGradients

class LocalExplanation:
    def __init__(self, model,dataset,edge_index_dict,target_var_idx,device):
        self.model = model.to(device)
        self.data = dataset
        self.edge_index_dict = edge_index_dict
        self.target_var_idx = target_var_idx
        self.device = device
        self.num_samples = len(self.data)
        self.ig = IntegratedGradients(self.global_system_forward)

    def global_system_forward(self,w_x, c_dyn, c_static):
        data = HeteroData()
        data['water'].x = w_x.squeeze(0)
        data['city'].x_dyn = c_dyn.squeeze(0)
        data['city'].x_static = c_static.squeeze(0)

        data['water'].num_nodes = 14
        data['city'].num_nodes = 28
        for edge_type, edge_index in self.edge_index_dict.items():
            data[edge_type].edge_index = edge_index.to(self.device)

        data['water'].batch = torch.zeros(14, dtype=torch.long).to(self.device)
        data['city'].batch = torch.zeros(28, dtype=torch.long).to(self.device)

        out = self.model(data)  # 输出形状: [14,2]

        regional_total_pollution = out[:,self.target_var_idx].sum()

        return regional_total_pollution.unsqueeze(0)

    def explain(self,target_node_idx,n_steps=20, save_path=None):


        global_water_attr = 0.0             # 水质特征
        global_city_dyn_attr = 0.0          # 城市气象
        global_city_static_attr = 0.0       # 城市静态

        for sample in tqdm(self.data, desc=f"站点 {target_node_idx} 全局分析"):
            w_x = sample['water'].x.to(self.device).requires_grad_()
            c_dyn = sample['city'].x_dyn.to(self.device).requires_grad_()
            c_static = sample['city'].x_static.to(self.device).requires_grad_()

            # baseline
            baselines = (
                torch.zeros_like(w_x),
                torch.zeros_like(c_dyn),
                torch.zeros_like(c_static)
            )
            self.model.train()
            # tuple(0,1,2)对应三类特征(w_x, c_dyn, c_static)
            attributions = self.ig.attribute(
                inputs=(w_x, c_dyn, c_static),
                baselines=baselines,
                n_steps=n_steps
            )
            self.model.eval()

            attr_w = np.abs(attributions[0].squeeze(0).cpu().detach().numpy())
            attr_c_dyn = np.abs(attributions[1].squeeze(0).cpu().detach().numpy())
            attr_c_static = np.abs(attributions[2].squeeze(0).cpu().detach().numpy())
            global_water_attr += attr_w
            global_city_dyn_attr += attr_c_dyn
            global_city_static_attr += attr_c_static
        # 1. 求时间序列上的全局平均
        global_water_attr /= self.num_samples
        global_city_dyn_attr /= self.num_samples
        global_city_static_attr /= self.num_samples

        # 2. 空间聚合：把所有特征和时间步的影响力加和，得到每个【节点】的单一分数
        water_node_importance = np.sum(global_water_attr, axis=(1, 2))
        city_dyn_importance = np.sum(global_city_dyn_attr, axis=(1, 2))
        city_static_importance = np.sum(global_city_static_attr, axis=1)
        city_total_importance = city_dyn_importance + city_static_importance
        results = {
            'water_node_importance': water_node_importance,
            'city_node_importance': city_total_importance,
            'water_attr': global_water_attr,
            'city_dyn_attr': global_city_dyn_attr,
            'city_static_attr':global_city_static_attr,
        }
        return results

    def plot_node_seq(self, results, target_node_idx, node_names=None):
        data = np.sum(results['water_attr'][target_node_idx], axis=1)
        n_timesteps = len(data)
        x_labels = [f"t-{i}" if i != 0 else "t" for i in range(n_timesteps - 1, -1, -1)]
        x = np.arange(n_timesteps)
        width = 0.6
        fig, ax = plt.subplots(figsize=(4, 3))
        bar_color = '#425066'
        ax.bar(x, data, width, color=bar_color, label='Contribution')
        ax.set_ylabel("Contribution", fontsize=11, fontweight='bold')
        ax.set_ylim(0, 0.5)
        ax.set_yticks([0, 0.25, 0.5])  # 根据参考图设置特定刻度
        ax.tick_params(axis='y', labelsize=10)  # 调整刻度字体大小
        # 设置 X 轴
        ax.set_xticks(x)
        ax.set_xticklabels(x_labels, fontsize=10)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        node_label = node_names[target_node_idx] if node_names else f"Node {target_node_idx}"
        # ax.set_title(f"{node_label}: Temporal Sensitivity", fontsize=12)
        plt.tight_layout()
        plt.show()

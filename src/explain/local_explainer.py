import os
import torch
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from torch_geometric.data import HeteroData
from captum.attr import IntegratedGradients

plt.style.use('seaborn-v0_8-white')
plt.rcParams['font.family'] = ['Times New Roman',"SimSun",'SimHei']
plt.rcParams['axes.unicode_minus'] = False

class LocalExplanation:
    def __init__(self, model,dataset,edge_index_dict,target_water_idx,target_var_idx,device):
        self.model = model.to(device)
        self.data = dataset
        self.edge_index_dict = edge_index_dict
        self.target_water_idx = target_water_idx
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

        regional_total_pollution = out[self.target_water_idx,self.target_var_idx].sum()

        return regional_total_pollution.unsqueeze(0)

    def explain(self,n_steps=20, save_path=None):


        global_water_attr = 0.0             # 水质特征
        global_city_dyn_attr = 0.0          # 城市气象
        global_city_static_attr = 0.0       # 城市静态

        for sample in tqdm(self.data, desc=f"站点 {self.target_water_idx} 全局分析"):
            sample: HeteroData
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
        results = {
            'water': global_water_attr,
            'city_dyn': global_city_dyn_attr,
            'city_static':global_city_static_attr,
        }
        return results

    def plot_node_pie(self,results,target_idx,saveFolder):
        total_importance = np.sum(results['water'])+np.sum(results['city_dyn'])+np.sum(results['city_static'])
        water_pct = np.sum(results['water'],axis=(1,2))/total_importance
        city_dyn_pct = np.sum(results['city_dyn'],axis=(1,2))/total_importance
        city_static_pct = np.sum(results['city_static'],axis=1)/total_importance
        self_pct = water_pct[target_idx]
        other_water_pct = np.sum(water_pct)-self_pct
        other_city_pct = np.sum(city_static_pct)+np.sum(city_dyn_pct)

        data = [self_pct, other_water_pct, other_city_pct]
        colors = ['#4A5A6A', '#A8A8A8', '#B00000']
        labels = ['Self', 'Other Water Nodes', 'City Nodes']

        fig, ax = plt.subplots(figsize=(3, 3))

        wedges, texts,autotexts = ax.pie(
            data,
            colors=colors,
            startangle=90,
            counterclock=False,
            wedgeprops={'edgecolor': 'white', 'linewidth': 1.5},
            autopct='%1.1f%%',
            pctdistance=0.65,
            textprops={'color': 'white', 'fontweight': 'bold', 'fontsize': 10}  # 默认全局文字样式
        )
        if len(autotexts) >= 3:
            autotexts[2].set_color('black')
        plt.tight_layout()
        plt.savefig(os.path.join(saveFolder,f'{target_idx}Contribution pct.png'))
        plt.show()

    def plot_node_seq(self, results, target_node_idx,saveFolder, node_names=None):
        # 计算占比
        data = np.sum(results['water'][target_node_idx],axis=1)/np.sum(results['water'][target_node_idx])

        data = data[16:]
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
        ax.set_xticklabels(x_labels, fontsize=10,rotation=45, ha='right')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        node_label = node_names[target_node_idx] if node_names else f"Node {target_node_idx}"
        ax.set_title(f"{node_label}: Contribution from 16 time step", fontsize=12)

        plt.tight_layout()
        plt.savefig(os.path.join(saveFolder,f'{node_label}Contribution of 16 time step.png'))
        plt.show()

    def plot_features_importance(self, results, target_node_idx,feature_names_dict,saveFolder, top_k=10):
        all_names = []
        all_values = []
        colors = []
        result_importance = {
            'water': np.sum(results['water'],axis=(0,1)),
            'city_dyn':np.sum(results['city_dyn'],axis=(0,1)),
            'city_static':np.sum(results['city_static'],axis=0),
        }
        categories = ['Water_x', 'City_dyn', 'City_se']
        category_colors = ['#3498db', '#e74c3c', '#2ecc71']

        # 合并所有特征及其类别颜色
        for category, color in zip(['water', 'city_dyn', 'city_static'], category_colors):
            all_names.extend(feature_names_dict[category])
            all_values.extend(result_importance[category])
            colors.extend([color] * len(feature_names_dict[category]))

        # 排序
        all_values = np.array(all_values)
        all_names = np.array(all_names)
        colors = np.array(colors)

        idx = np.argsort(all_values)[::-1][:top_k]  # 取前 K 个

        plt.figure(figsize=(12, 6))
        bars = plt.barh(all_names[idx][::-1], all_values[idx][::-1], color=colors[idx][::-1])
        plt.xlabel('Feature Importance')
        plt.title(f'Feature Importance (Top {top_k})')
        plt.grid(axis='x', linestyle='--', alpha=0.7)

        legend_patches = [mpatches.Patch(color=color, label=category)
                        for category, color in zip(categories, category_colors)]
        plt.legend(handles=legend_patches, title='Feature Types', loc='lower right')

        # 添加标注
        for bar in bars:
            width = bar.get_width()
            plt.text(width + 0.01, bar.get_y() + bar.get_height() / 2, f'{width:.2f}', va='center')
        plt.tight_layout()
        plt.savefig(os.path.join(saveFolder,f'station{target_node_idx}FeatureImportance.png'))
        plt.show()
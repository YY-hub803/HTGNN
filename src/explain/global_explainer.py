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

class GlobalExplanation:
    def __init__(self, model,dataset,edge_index_dict,target_var_idx,device):
        self.model = model.to(device)
        self.data = dataset
        self.edge_index_dict = edge_index_dict
        self.target_var_idx = target_var_idx
        self.device = device
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

        out = self.model(data)  # 输出形状: [14, pred,2]
        # 全局目标：把所有 14 个站点的目标污染物浓度加起来！
        regional_total_pollution = out[:,self.target_var_idx].sum()

        return regional_total_pollution.unsqueeze(0)

    def explain(self,n_steps=20, save_path=None):
        global_w_imp = 0.0
        global_c_dyn_imp = 0.0
        global_c_static_imp = 0.0

        for idx,sample in enumerate(tqdm(self.data, desc="计算全流域 IG")):
            sample: HeteroData          # 类型注释
            water_x = sample['water'].x.to(self.device).requires_grad_()
            city_x_dyn = sample['city'].x_dyn.to(self.device).requires_grad_()
            city_x_static = sample['city'].x_static.to(self.device).requires_grad_()

            # baseline
            baselines = (
                torch.zeros_like(water_x),
                torch.zeros_like(city_x_dyn),
                torch.zeros_like(city_x_static)
            )

            self.model.train()
            # tuple(0,1,2)对应三类特征(w_x, c_dyn, c_static)
            attributions = self.ig.attribute(
                inputs=(water_x, city_x_dyn, city_x_static),
                baselines=baselines,
                n_steps=20
            )
            self.model.eval()
            # 取绝对值
            attr_w = np.abs(attributions[0].squeeze(0).cpu().detach().numpy())  # [14, 32, 7]
            attr_c_dyn = np.abs(attributions[1].squeeze(0).cpu().detach().numpy())  # [28, 32, 1]
            attr_c_static = np.abs(attributions[2].squeeze(0).cpu().detach().numpy())  # [28, F_static]
            global_w_imp += attr_w
            global_c_dyn_imp += attr_c_dyn
            global_c_static_imp += attr_c_static
        results = {
            'water': global_w_imp,
            'city_dyn': global_c_dyn_imp,
            'city_static': global_c_static_imp,

        }
        return results

    def plot_importance(self,results,feature_names_dict,saveFolder,top_k=10):
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
        plt.title(f'Global Feature Importance (Top {top_k})')
        plt.grid(axis='x', linestyle='--', alpha=0.7)

        legend_patches = [mpatches.Patch(color=color, label=category)
                        for category, color in zip(categories, category_colors)]
        plt.legend(handles=legend_patches, title='Feature Types', loc='lower right')

        # 添加标注
        for bar in bars:
            width = bar.get_width()
            plt.text(width + 0.3, bar.get_y() + bar.get_height() / 2, f'{width:.2f}', va='center')
        plt.tight_layout()
        plt.savefig(os.path.join(saveFolder,'GlobalFeatureImportance.png'))
        plt.show()


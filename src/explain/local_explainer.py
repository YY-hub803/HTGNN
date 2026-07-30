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
        regional_total_pollution = (
                regional_total_pollution
                + 0 * w_x.sum()
                + 0 * c_dyn.sum()
                + 0 * c_static.sum()
        )
        return regional_total_pollution.unsqueeze(0)

    def explain(self,n_steps=20, save_path=None):


        global_water_attr = 0.0             # 水质特征
        global_city_dyn_attr = 0.0          # 城市气象
        global_city_static_attr = 0.0       # 城市静态

        all_value = list()
        all_importance = list()

        for sample in tqdm(self.data, desc=f"全局分析"):
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
                n_steps=n_steps,
            )
            self.model.eval()

            water_score = attributions[0].squeeze(0).cpu().detach().numpy()
            city_dyn_score = attributions[1].squeeze(0).cpu().detach().numpy()
            city_static_score = attributions[2].squeeze(0).cpu().detach().numpy()

            attr_w = np.abs(water_score)
            attr_c_dyn = np.abs(city_dyn_score)
            attr_c_static = np.abs(city_static_score)

            sample_importance = np.concatenate([np.mean(water_score, axis=(0, 1)), np.mean(city_dyn_score, axis=(0, 1)),
                            np.mean(city_static_score, axis=0)])
            sample_value = np.concatenate([np.mean(w_x.cpu().detach().numpy(),axis=(0, 1)), np.mean(c_dyn.cpu().detach().numpy(), axis=(0, 1)),
                            np.mean(c_static.cpu().detach().numpy(), axis=0)])
            all_value.append(sample_value)
            all_importance.append(sample_importance)

            global_water_attr += attr_w
            global_city_dyn_attr += attr_c_dyn
            global_city_static_attr += attr_c_static
            torch.cuda.empty_cache()

        results = {
            'water': global_water_attr,
            'city_dyn': global_city_dyn_attr,
            'city_static':global_city_static_attr,
        }
        Sample = {
            "value": all_value,
            "importance": all_importance,
        }
        return results,Sample

class LocalExplanation:
    def __init__(self, model,dataset,edge_index_dict,target_water_idx,target_var_idx,device):
        self.model = model.to(device)
        self.data = dataset
        self.edge_index_dict = edge_index_dict
        self.target_water_idx = target_water_idx
        self.target_var_idx = target_var_idx
        self.device = device
        self.num_samples = len(self.data)
        self.ig = IntegratedGradients(self.warped)

    def warped(self,w_x, c_dyn, c_static):
        data = HeteroData()
        data['water'].x = w_x.squeeze(0)
        data['city'].x_dyn = c_dyn.squeeze(0)
        data['city'].x_static = c_static.squeeze(0)

        num_water_nodes = w_x.shape[0]
        num_city_nodes = c_dyn.shape[0]
        data['water'].num_nodes = num_water_nodes
        data['city'].num_nodes = num_city_nodes


        for edge_type, edge_index in self.edge_index_dict.items():
            data[edge_type].edge_index = edge_index.to(self.device)
            # 边属性从存储中读取，不参与梯度计算
            data[edge_type].edge_attr = self.stored_edge_attrs[edge_type].to(self.device)

        out = self.model(data)  # 输出形状: [14,2]

        regional_total_pollution = out[self.target_water_idx,self.target_var_idx]
        regional_total_pollution = (
                regional_total_pollution
                + 0 * w_x.sum()
                + 0 * c_dyn.sum()
                + 0 * c_static.sum()
        )
        return regional_total_pollution.unsqueeze(0)

    def explain(self,n_steps=5, save_path=None):

        # 计算特征的全局重要性
        global_water_attr = 0.0             # 水质特征
        global_city_dyn_attr = 0.0          # 城市气象
        global_city_static_attr = 0.0       # 城市静态
        # IG分数以及对应样本的输入特征值
        all_value = list()
        all_importance = list()

        threshold = 1e-6
        num_sample = len(self.data)
        for sample in tqdm(self.data, desc=f"站点 {self.target_water_idx} 全局分析"):
            sample: HeteroData

            self.stored_edge_attrs = {}
            for edge_type in self.edge_index_dict:
                if 'edge_attr' in sample[edge_type]:
                    self.stored_edge_attrs[edge_type] = sample[edge_type].edge_attr.clone()
                else:
                    num_edges = sample[edge_type].edge_index.size(1)
                    self.stored_edge_attrs[edge_type] = torch.ones(num_edges, 1)


            w_x = sample['water'].x.unsqueeze(0).to(self.device).requires_grad_()
            c_dyn = sample['city'].x_dyn.unsqueeze(0).to(self.device).requires_grad_()
            c_static = sample['city'].x_static.unsqueeze(0).to(self.device).requires_grad_()

            # baseline
            baselines = (
                torch.zeros_like(w_x),
                torch.zeros_like(c_dyn),
                torch.zeros_like(c_static)
            )
            self.model.eval()
            # tuple(0,1,2)对应三类特征(w_x, c_dyn, c_static)
            with torch.backends.cudnn.flags(enabled=False):
                attributions = self.ig.attribute(
                    inputs=(w_x, c_dyn, c_static),
                    baselines=baselines,
                    n_steps=n_steps,
                    internal_batch_size=1
                )
            # self.model.eval()

            # IG分数
            water_score = attributions[0].squeeze(0).cpu().detach().numpy()  # 【N,T,F】
            city_dyn_score = attributions[1].squeeze(0).cpu().detach().numpy()  # [N,T,F]
            city_static_score = attributions[2].squeeze(0).cpu().detach().numpy()  # [N,F]
            n_w_x = w_x.cpu().squeeze(0).detach().numpy()
            n_c_dyn = c_dyn.cpu().squeeze(0).detach().numpy()
            n_c_static = c_static.cpu().squeeze(0).detach().numpy()
            # 对分数求绝对值，用于计算全局重要性
            attr_w = np.abs(water_score)
            attr_c_dyn = np.abs(city_dyn_score)
            attr_c_static = np.abs(city_static_score)

            # 取出目标节点自身特征的分数，，上游水质节点特征的分数，，city节点特征的分数
            # 取出目标节点自身特征的分数，，上游水质节点特征的分数,对时间序列求平均
            wq_self_score = np.sum(water_score[self.target_water_idx],axis=0)
            wq_self_val = np.mean(n_w_x[self.target_water_idx],axis=0)
            wq_idx = np.where(np.sum(attr_w, axis=(1, 2)) > threshold)[0]
            selected_wq_idx = wq_idx[wq_idx != self.target_water_idx]
            if len(selected_wq_idx) > 0:
                wq_other_score = np.sum(water_score[selected_wq_idx], axis=(0, 1))
                wq_other_val = np.mean(n_w_x[selected_wq_idx], axis=(0, 1))
            else:
                F_dim = n_w_x.shape[-1]
                wq_other_score = np.zeros(F_dim)
                wq_other_val = np.zeros(F_dim)


            # 取出city节点的分数，先根据IG分数绝对值进行筛选，对筛选后的绝对值做平均
            node_imp_dyn = np.sum(attr_c_dyn, axis=(1, 2))
            node_imp_static = np.sum(attr_c_static, axis=1)
            # 筛选出IG分数大于1e-6的节点，然后取并集，
            selected_dyn_nodes = np.where(node_imp_dyn > threshold)[0]
            selected_static_nodes = np.where(node_imp_static > threshold)[0]
            selected_all_cities = np.union1d(selected_dyn_nodes, selected_static_nodes)
            # 取出筛选后的节点的IG分数
            filtered_c_dyn_ig = city_dyn_score[selected_all_cities]
            filtered_c_dyn_val = n_c_dyn[selected_all_cities]
            filtered_c_static_ig = city_static_score[selected_all_cities]
            filtered_c_static_val = n_c_static[selected_all_cities]
            sample_importance = np.concatenate([wq_self_score,wq_other_score, np.sum(filtered_c_dyn_ig, axis=(0, 1)),
                            np.sum(filtered_c_static_ig, axis=0)])
            sample_value = np.concatenate([wq_self_val,wq_other_val, np.mean(filtered_c_dyn_val, axis=(0, 1)),
                            np.mean(filtered_c_static_val, axis=0)])
            all_value.append(sample_value)
            all_importance.append(sample_importance)

            global_water_attr += attr_w
            global_city_dyn_attr += attr_c_dyn
            global_city_static_attr += attr_c_static


        results = {
            'water': global_water_attr/num_sample,
            'city_dyn': global_city_dyn_attr/num_sample,
            'city_static':global_city_static_attr/num_sample}
        sample_importance = {
            "value": np.stack(all_value),
            "importance": np.stack(all_importance),}

        return results,sample_importance

    def plot_node_pie(self,results,target_idx,saveFolder):
        total_importance = np.sum(results['water'])+np.sum(results['city_dyn'])+np.sum(results['city_static'])
        water_pct = np.sum(results['water'],axis=(1,2))/total_importance
        meteo_dyn_pct = np.sum(results['city_dyn'][:, :, 0:3], axis=(1, 2)) / total_importance
        city_static_pct = np.sum(np.sum(results['city_static'],axis=1)+np.sum(results['city_dyn'][:,:,3:5],axis=(1,2)))/total_importance
        self_pct = water_pct[target_idx]
        other_water_pct = np.sum(water_pct)-self_pct
        meteo_pct = np.sum(meteo_dyn_pct)
        other_city_pct = np.sum(city_static_pct)

        data = [self_pct, other_water_pct, other_city_pct,meteo_pct]
        colors = ['#4A5A6A', '#A8A8A8', '#B00000','#2E86AB']
        labels = ['Self', 'Other Water Nodes', 'City Nodes','Meteo']

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
            'city_dyn':np.sum(results['city_dyn'][:,:,0:3],axis=(0,1)),
            'city_static':np.hstack([np.sum(results['city_static'],axis=0),np.sum(results['city_dyn'][:,:,3:],axis=(0,1))]),
        }
        categories = ['Water', 'Meteo', 'Socioeconomic']
        category_colors = ['#4A5A6A', '#2E86AB', '#B00000']

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
            plt.text(width, bar.get_y() + bar.get_height() / 2, f'{width:.2f}', va='center')
        plt.tight_layout()
        plt.savefig(os.path.join(saveFolder,f'station{target_node_idx}FeatureImportance.png'))
        plt.show()
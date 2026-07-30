import os
import json
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from torch_geometric.data import HeteroData
from tqdm import tqdm

# 设置画图样式，确保论文级别的清晰度
plt.style.use('seaborn-v0_8-white')
plt.rcParams['font.family'] = ['Times New Roman', 'SimSun', 'SimHei']  # 支持中英文
plt.rcParams['axes.unicode_minus'] = False


class FullSpaceCounterfactualOptimizer:
    def __init__(self, model, dataset, edge_index_dict, train_stats_path, target_water_idx, target_var_idx,
                 device='cuda'):
        """
        基于优化的反事实解释器 (支持真实物理空间映射与高级可视化)
        """
        self.model = model.to(device)
        self.device = device

        # 冻结模型参数
        for param in self.model.parameters():
            param.requires_grad = False
        self.model.eval()

        self.dataset = dataset
        self.edge_index_dict = edge_index_dict
        self.target_water_idx = target_water_idx
        self.target_var_idx = target_var_idx

        # ================= 核心：加载 Z-Score 统计信息以支持物理量还原 =================
        print("加载 train_stats.json 统计特征...")
        with open(train_stats_path, 'r') as f:
            stats = json.load(f)

        # 提取目标变量(Y)的均值和方差 (将其展平后取对应污染物的索引)
        self.y_mean = torch.tensor(stats['y_mean']).view(-1)[target_var_idx].to(device)
        self.y_std = torch.tensor(stats['y_std']).view(-1)[target_var_idx].to(device)

        # 提取特征(X)的方差 (展平为 1D 向量，方便后续自动广播乘法)
        self.x_static_std = torch.tensor(stats['x_static_std']).view(-1).to(device)
        self.x_city_std = torch.tensor(stats['x_city_std']).view(-1).to(device)
        # ==============================================================================

    def _forward_pass(self, w_x, c_dyn, c_static):
        data = HeteroData()
        data['water'].x = w_x.squeeze(0) if w_x.dim() > 2 else w_x
        data['city'].x_dyn = c_dyn.squeeze(0) if c_dyn.dim() > 3 else c_dyn
        data['city'].x_static = c_static.squeeze(0) if c_static.dim() > 2 else c_static

        data['water'].num_nodes = 14
        data['city'].num_nodes = 28
        for edge_type, edge_index in self.edge_index_dict.items():
            data[edge_type].edge_index = edge_index.to(self.device)

        data['water'].batch = torch.zeros(14, dtype=torch.long).to(self.device)
        data['city'].batch = torch.zeros(28, dtype=torch.long).to(self.device)

        out = self.model(data)
        # 输出为 Z-score 空间的预测值
        return out[self.target_water_idx, self.target_var_idx]

    def generate_counterfactual(self, sample_idx, target_val_real, feature_type='city_static', max_iter=300, lr=0.01,
                                lambda_l1=0.1, verbose=True):
        """
        寻找最小扰动 Delta (目标输入为真实浓度，输出 Delta 也是真实物理改变量)
        """
        sample = self.dataset[sample_idx]
        w_x = sample['water'].x.to(self.device).clone()
        c_dyn = sample['city'].x_dyn.to(self.device).clone()
        c_static = sample['city'].x_static.to(self.device).clone()

        # 1. 计算 Baseline (Z-score 还原为 真实空间)
        with torch.no_grad():
            baseline_pred_z = self._forward_pass(w_x, c_dyn, c_static)
            baseline_pred_real = (baseline_pred_z * self.y_std + self.y_mean).item()

        # 2. 目标转换 (真实空间 映射为 Z-score 空间)
        target_val_z = (target_val_real - self.y_mean.item()) / self.y_std.item()

        if verbose:
            print(f"\n[{feature_type}] 优化启动...")
            print(f" -> 原始预测 (真实浓度): {baseline_pred_real:.4f}  |  期望目标 (真实浓度): {target_val_real:.4f}")
            print(f" -> 对应模型 (Z-Score): {baseline_pred_z.item():.4f}  |  对应目标 (Z-Score): {target_val_z:.4f}")

        # 3. 初始化 Z-score 空间的扰动项
        # 好处：在Z空间中惩罚 L1，相当于对所有特征施加“平权”的干预成本，不偏袒方差大的特征
        if feature_type == 'city_static':
            delta_z = torch.zeros_like(c_static, requires_grad=True)
            optimizer = torch.optim.Adam([delta_z], lr=lr)
        else:
            delta_z = torch.zeros_like(c_dyn, requires_grad=True)
            optimizer = torch.optim.Adam([delta_z], lr=lr)

        criterion = nn.MSELoss()
        target_tensor = torch.tensor([target_val_z], dtype=torch.float32, device=self.device)

        loss_history = []
        pred_history_real = []  # 记录真实空间的收敛曲线供画图

        iterator = range(max_iter) if not verbose else tqdm(range(max_iter), desc="Counterfactual Optimization")
        for i in iterator:
            optimizer.zero_grad()

            if feature_type == 'city_static':
                pred_y_z = self._forward_pass(w_x, c_dyn, c_static + delta_z)
            else:
                pred_y_z = self._forward_pass(w_x, c_dyn + delta_z, c_static)

            # 损失计算在 Z-score 空间进行
            loss_mse = criterion(pred_y_z.unsqueeze(0), target_tensor)
            loss_l1 = torch.norm(delta_z, p=1)

            total_loss = loss_mse + lambda_l1 * loss_l1
            total_loss.backward()
            optimizer.step()

            # 记录真实空间的预测值
            pred_y_real = (pred_y_z * self.y_std + self.y_mean).item()
            pred_history_real.append(pred_y_real)
            loss_history.append(total_loss.item())

            if loss_mse.item() < 1e-6:
                if verbose: print(f"提前收敛于第 {i} 轮，最终预测浓度: {pred_y_real:.4f}")
                break

        # 4. Delta 解码 (Z-score 扰动量 还原为 真实物理改变量)
        if feature_type == 'city_static':
            delta_real = delta_z.detach() * self.x_static_std
        else:
            delta_real = delta_z.detach() * self.x_city_std

        final_delta_real = delta_real.cpu().numpy()
        return final_delta_real, baseline_pred_real, pred_history_real

    # ================= 高级可视化 1：图3 空间干预策略地图 =================
    def plot_spatial_intervention_map(self, delta_real, city_points_path, water_points_path, save_path=None):
        print("\nDrawing Spatial Intervention Map...")
        try:
            city_df = pd.read_csv(city_points_path,encoding="gbk")
            water_df = pd.read_csv(water_points_path)
            lon_col = 'Lon' if 'Lon' in city_df.columns else city_df.columns[1]
            lat_col = 'Lat' if 'Lat' in city_df.columns else city_df.columns[2]
        except Exception as e:
            print(f"Error loading CSV files: {e}. Cannot draw spatial map.")
            return

        # 计算每个城市的真实干预总强度 (各物理特征改变绝对值之和)
        if delta_real.ndim == 3:
            delta_real = np.mean(delta_real, axis=1)
        city_intensity = np.sum(np.abs(delta_real), axis=1)

        # 归一化强度用于画图控制点的大小
        max_intensity = np.max(city_intensity) + 1e-5
        normalized_intensity = (city_intensity / max_intensity) * 1000

        fig, ax = plt.subplots(figsize=(10, 8))

        # 1. 画水质节点 (作为参考背景)
        ax.scatter(water_df[lon_col], water_df[lat_col], c='#2E86AB', s=100, marker='^', alpha=0.6, label='Water Nodes')

        # 特别标出目标水质站点
        target_lon = water_df.iloc[self.target_water_idx][lon_col]
        target_lat = water_df.iloc[self.target_water_idx][lat_col]
        ax.scatter(target_lon, target_lat, c='red', s=400, marker='*', edgecolors='black', label='Target Water Node',
                   zorder=5)

        # 2. 画城市节点气泡
        scatter = ax.scatter(city_df[lon_col], city_df[lat_col],
                             s=normalized_intensity + 20,
                             c=city_intensity, cmap='Reds',
                             alpha=0.7, edgecolors='gray', label='City Intervention Needed')

        for i, row in city_df.iterrows():
            if city_intensity[i] > max_intensity * 0.1:
                ax.text(row[lon_col] + 0.01, row[lat_col] + 0.01, f"C{i}", fontsize=10, fontweight='bold')

        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('Total Physical Modification Required ($\Sigma|\Delta_{real}|$)', fontsize=12)

        ax.set_title("Spatial Counterfactual Intervention Map", fontsize=16, fontweight='bold')
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        ax.legend(loc='upper right')
        ax.grid(True, linestyle='--', alpha=0.5)

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300)
            print(f" -> Map saved to {save_path}")
        plt.show()

    # ================= 高级可视化 2：图5 成本-效益帕累托前沿 =================
    def plot_pareto_frontier(self, sample_idx, target_val_real, lambda_list, save_path=None):
        print(f"\nRunning Pareto Analysis over {len(lambda_list)} budget scenarios...")
        costs = []
        benefits = []

        for l1 in lambda_list:
            delta_real, baseline_real, pred_hist = self.generate_counterfactual(
                sample_idx, target_val_real, feature_type='city_static',
                lambda_l1=l1, verbose=False
            )
            final_pred_real = pred_hist[-1]

            # Cost = 真实物理空间的干预幅度总和
            cost = np.sum(np.abs(delta_real))
            # Benefit = 真实污染浓度的下降量
            benefit = baseline_real - final_pred_real

            costs.append(cost)
            benefits.append(benefit)

        fig, ax = plt.subplots(figsize=(8, 6))

        sort_idx = np.argsort(costs)
        costs = np.array(costs)[sort_idx]
        benefits = np.array(benefits)[sort_idx]

        ax.plot(costs, benefits, 'o-', color='#B00000', linewidth=2, markersize=8)

        ax.fill_between(costs, benefits, color='#B00000', alpha=0.1)
        ax.set_title("Cost-Benefit Pareto Trade-off for Quality Management", fontsize=14, fontweight='bold')
        ax.set_xlabel("Intervention Cost (Total Physical Alterations $\Sigma|\Delta_{real}|$)", fontsize=12)
        ax.set_ylabel("Environmental Benefit (Pollution Reduction mg/L)", fontsize=12)

        ax.annotate('Low Budget\nSparse Changes', xy=(costs[0], benefits[0]), xytext=(costs[0], benefits[0] - 0.005),
                    arrowprops=dict(facecolor='black', shrink=0.05, width=1, headwidth=5))
        ax.annotate('High Budget\nMax Reduction', xy=(costs[-1], benefits[-1]),
                    xytext=(costs[-1], benefits[-1] - 0.005),
                    arrowprops=dict(facecolor='black', shrink=0.05, width=1, headwidth=5), ha='right')

        ax.grid(True, linestyle='--', alpha=0.5)
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300)
            print(f" -> Pareto curve saved to {save_path}")
        plt.show()


# ================= 执 行 主 函 数 =================
if __name__ == "__main__":
    from src.utils.load_data import build_edge_index_dict

    # --- 1. 配置真实路径 (请根据你电脑实际位置微调) ---
    dir_info = {
        'city_to_water': r'D:\Program\HTGNN\data/info_data/city_to_water.csv',
        'water_to_water': r'D:\Program\HTGNN\data/info_data/water_to_water.csv',
        'city_to_city': r'D:\Program\HTGNN\data/info_data/city_to_city.csv',
        'water_points': r'D:\Program\HTGNN\data/info_data/water_points.csv',
        'city_points': r'D:\Program\HTGNN\data/info_data/city_points.csv'
    }

    dataset_path = r'D:\Program\HTGNN\data\dataset\Test_dataset.pt'
    train_stats_path = r'D:\Program\HTGNN\data\dataset\train_stats.json'
    model_path = r'D:\Program\HTGNN\Random42OutPut\GruHANModel_B32_H16_L4_NL2_NH4_lr0.0010\best_model.pt'

    save_dir = 'cf_physical_results'
    os.makedirs(save_dir, exist_ok=True)

    # --- 2. 加载数据与模型 ---
    print("Loading model and dataset...")
    dataset = torch.load(dataset_path)
    model = torch.load(model_path, map_location='cuda', weights_only=False)
    edge_index_dict = build_edge_index_dict(dir_info)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # 设定目标：分析0号水质站点的 TP (0号变量)
    target_water_idx = 11
    target_var_idx = 0

    cf_explainer = FullSpaceCounterfactualOptimizer(
        model, dataset, edge_index_dict, train_stats_path,
        target_water_idx, target_var_idx, device=device
    )

    sample_idx = 10

    # 【业务目标】: 将真实水质浓度降低到 0.04 mg/L
    target_pollution_value_real = 0.04

    # ================= 实验一：单次反事实生成与空间可视化 =================
    delta_static_real, base_val_real, pred_curve_real = cf_explainer.generate_counterfactual(
        sample_idx=sample_idx,
        target_val_real=target_pollution_value_real,
        feature_type='city_static',
        max_iter=300,
        lr=0.01,
        lambda_l1=0.04  # 调整成本惩罚项
    )

    # 画物理浓度收敛图
    plt.figure(figsize=(6, 4))
    plt.plot(pred_curve_real, color='#B00000', linewidth=2)
    plt.axhline(y=target_pollution_value_real, color='grey', linestyle='--', label='Target (mg/L)')
    plt.axhline(y=base_val_real, color='blue', linestyle='--', label='Baseline Prediction')
    plt.title("Optimization Convergence in Physical Space")
    plt.ylabel("Predicted TP Value (mg/L)")
    plt.xlabel("Iteration Step")
    plt.legend()
    plt.savefig(os.path.join(save_dir, "1_Real_Convergence.png"), dpi=300)
    plt.close()

    # 画图3：基于物理改变量的空间干预地图
    cf_explainer.plot_spatial_intervention_map(
        delta_real=delta_static_real,
        city_points_path=dir_info['city_points'],
        water_points_path=dir_info['water_points'],
        save_path=os.path.join(save_dir, "2_Spatial_Map_Real_Cost.png")
    )

    # ================= 实验二：帕累托前沿分析 =================
    # 遍历不同大小的干预预算 (从极度苛刻 到 允许大改)
    lambda_list = [2.0, 1.0, 0.5, 0.2, 0.1, 0.05, 0.01]

    cf_explainer.plot_pareto_frontier(
        sample_idx=sample_idx,
        target_val_real=0.03,  # 在此实验中设置一个更苛刻的目标，迫使模型权衡
        lambda_list=lambda_list,
        save_path=os.path.join(save_dir, "3_Pareto_Tradeoff_Real.png")
    )

    print("\nAll tasks completed successfully. Check the output folder!")
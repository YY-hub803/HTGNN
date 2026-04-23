import torch
from tqdm import tqdm
import numpy as np
import pandas as pd
from captum.attr import IntegratedGradients
from torch_geometric.data import HeteroData
from data.load_data import build_edge_index_dict
def global_explain_target(model, dataset, edge_index_dict, target_water_idx, target_var_idx, device='cuda'):
    """
    对时间序列数据集执行全局 IG 归因分析。
    """


    total_samples = len(dataset)
    sample_indices = range(total_samples)
    num_samples = total_samples

    print(f"\n🌍 开始全局解释计算，共处理 {num_samples} 个时间窗口...")

    # 初始化全局累加器 (用于存放绝对值累加的结果)
    global_water_attr = 0.0
    global_city_dyn_attr = 0.0
    global_city_static_attr = 0.0

    # 构造 Captum 需要的前向包装器
    def custom_forward(w_x, c_dyn, c_static):
        data = HeteroData()
        data['water'].x = w_x.squeeze(0)
        data['city'].x_dyn = c_dyn.squeeze(0)
        data['city'].x_static = c_static.squeeze(0)
        data['water'].num_nodes = 14
        data['city'].num_nodes = 28

        for edge_type, edge_index in edge_index_dict.items():
            data[edge_type].edge_index = edge_index.to(device)

        data['water'].batch = torch.zeros(14, dtype=torch.long).to(device)
        data['city'].batch = torch.zeros(28, dtype=torch.long).to(device)

        out = model(data)
        # 返回目标站点的预测值作为标量
        return out[target_water_idx, :,target_var_idx].unsqueeze(0)

    ig = IntegratedGradients(custom_forward)

    # ==========================================
    # 核心循环：遍历时间序列，累加【绝对值】
    # ==========================================
    for idx in tqdm(sample_indices, desc="计算 IG 积分梯度"):
        sample = dataset[idx]

        water_x = sample['water'].x.to(device).requires_grad_()
        city_x_dyn = sample['city'].x_dyn.to(device).requires_grad_()
        city_x_static = sample['city'].x_static.to(device).requires_grad_()
        # 基线：全零张量
        baseline_water = torch.zeros_like(water_x)
        baseline_city_dyn = torch.zeros_like(city_x_dyn)
        baseline_city_static = torch.zeros_like(city_x_static)
        # 执行 IG
        model.train()
        attributions = ig.attribute(
            inputs=(water_x, city_x_dyn, city_x_static),
            baselines=(baseline_water, baseline_city_dyn, baseline_city_static),
            n_steps=20
        )
        model.eval()

        # 加上 np.abs() 提取特征的影响力强度
        attr_w = np.abs(attributions[0].squeeze(0).cpu().detach().numpy())
        attr_c_dyn = np.abs(attributions[1].squeeze(0).cpu().detach().numpy())
        attr_c_static = np.abs(attributions[2].squeeze(0).cpu().detach().numpy())

        global_water_attr += attr_w
        global_city_dyn_attr += attr_c_dyn
        global_city_static_attr += attr_c_static

    # 1. 求时间序列上的全局平均
    global_water_attr /= num_samples
    global_city_dyn_attr /= num_samples
    global_city_static_attr /= num_samples

    # 2. 空间聚合：把所有特征和时间步的影响力加和，得到每个【节点】的单一分数
    # 形状压缩至 [14] 和 [28]
    water_node_importance = np.sum(global_water_attr, axis=(1, 2))

    city_dyn_importance = np.sum(global_city_dyn_attr, axis=(1, 2))
    city_static_importance = np.sum(global_city_static_attr, axis=1)
    city_total_importance = city_dyn_importance + city_static_importance

    print("\n✅ 全局重要性计算完成！")
    return water_node_importance, city_total_importance, global_water_attr, global_city_dyn_attr,global_city_static_attr

dir_info = {
    'city_to_water': r'D:\Program\HTGNN\data/info_data/city_to_water.csv',
    'water_to_water':  r'D:\Program\HTGNN\data/info_data/water_to_water.csv',
    'water_points':  r'D:\Program\HTGNN\data/info_data/water_points.csv',
    'city_points': r'D:\Program\HTGNN\data/info_data/city_points.csv'}

dataset = torch.load(r'D:\Program\HTGNN\data\dataset\Test_dataset.pt')
model = torch.load(r'D:\Program\HTGNN\OutPut_4h/GruHANModel_B32_H64_L32_P1_dr0.60_lr0.0001/best_model.pt',weights_only=False)
edge_index_dict = build_edge_index_dict(dir_info)
target_water_idx=7
target_var_idx=0
w_imp, c_imp, raw_w_attr, raw_c_attr,raw_c_static_attr = global_explain_target(model, dataset, edge_index_dict, target_water_idx, target_var_idx)
total_system_impact = np.sum(w_imp) + np.sum(c_imp)
water_contribution_ratio = w_imp / total_system_impact
city_contribution_ratio = c_imp / total_system_impact
import matplotlib.pyplot as plt
# 绘制上游水质节点的全局贡献
plt.bar(range(len(c_imp)), c_imp)
plt.title("全局解释：各水质节点对3 站 TN 的影响强度")
plt.xticks(rotation=45)
plt.show()

# 水质特征重要性 (形状: [5])
water_feat_importance = np.sum(raw_w_attr, axis=(0, 1))
# 城市动态特征重要性 (形状: [1])
city_dyn_feat_importance = np.sum(raw_c_attr, axis=(0, 1))
# 城市静态特征重要性 (形状: [F_static])
city_static_feat_importance = np.sum(raw_c_static_attr, axis=0)
total_feat_impact = (np.sum(water_feat_importance) +
                     np.sum(city_dyn_feat_importance) +
                     np.sum(city_static_feat_importance))
water_feat_ratio = water_feat_importance / total_feat_impact
city_dyn_feat_ratio = city_dyn_feat_importance / total_feat_impact
city_static_feat_ratio = city_static_feat_importance / total_feat_impact

######################### 全局特征重要性 #########################
def global_feature_importance(model, dataset, edge_index_dict, target_var_idx=1, device='cuda'):
    """
    计算真正的全局特征重要性（解释所有站点的总预测值）
    参数:
    - target_var_idx: 0 代表解释 TP(总磷)，1 代表解释 TN(总氮)
    """

    total_samples = len(dataset)
    sample_indices = range(total_samples)

    print(f"\n🌍 开始全局特征重要性计算 (解释全流域所有站点的总和)...")
    # ==========================================
    # 核心修改点：重新定义前向传播的目标
    # ==========================================
    def global_system_forward(w_x, c_dyn, c_static):
        data = HeteroData()
        data['water'].x = w_x.squeeze(0)
        data['city'].x_dyn = c_dyn.squeeze(0)
        data['city'].x_static = c_static.squeeze(0)

        data['water'].num_nodes = 14
        data['city'].num_nodes = 28
        for edge_type, edge_index in edge_index_dict.items():
            data[edge_type].edge_index = edge_index.to(device)

        data['water'].batch = torch.zeros(14, dtype=torch.long).to(device)
        data['city'].batch = torch.zeros(28, dtype=torch.long).to(device)

        out = model(data)  # 输出形状: [14, pred,2]

        # 全局目标：把所有 14 个站点的目标污染物浓度加起来！
        regional_total_pollution = out[:, :,target_var_idx].sum()

        return regional_total_pollution.unsqueeze(0)


    ig = IntegratedGradients(global_system_forward)

    # 累加器
    global_w_imp = 0.0
    global_c_dyn_imp = 0.0
    global_c_static_imp = 0.0

    for idx in tqdm(sample_indices, desc="计算全流域 IG"):
        sample = dataset[idx]

        water_x = sample['water'].x.to(device).requires_grad_()
        city_x_dyn = sample['city'].x_dyn.to(device).requires_grad_()
        city_x_static = sample['city'].x_static.to(device).requires_grad_()

        baseline_water = torch.zeros_like(water_x)
        baseline_city_dyn = torch.zeros_like(city_x_dyn)
        baseline_city_static = torch.zeros_like(city_x_static)
        model.train()
        attributions = ig.attribute(
            inputs=(water_x, city_x_dyn, city_x_static),
            baselines=(baseline_water, baseline_city_dyn, baseline_city_static),
            n_steps=20
        )
        model.eval()
        # 取绝对值
        attr_w = np.abs(attributions[0].squeeze(0).cpu().detach().numpy())  # [14, 32, 7]
        attr_c_dyn = np.abs(attributions[1].squeeze(0).cpu().detach().numpy())  # [28, 32, 1]
        attr_c_static = np.abs(attributions[2].squeeze(0).cpu().detach().numpy())  # [28, F_static]

        # 直接在这个循环里，沿着空间(节点)和时间(序列)维度把特征拍扁！
        # 只保留特征维度的长度，从而得到“某个特征在全局所有时空中的总活动量”
        global_w_imp += np.sum(attr_w, axis=(0, 1))  # 形状: [5]
        global_c_dyn_imp += np.sum(attr_c_dyn, axis=(0, 1))  # 形状: [1]
        global_c_static_imp += np.sum(attr_c_static, axis=0)  # 形状: [F_static]

    # ==========================================
    # 归一化为 100% 百分比
    # ==========================================
    total_impact = np.sum(global_w_imp) + np.sum(global_c_dyn_imp) + np.sum(global_c_static_imp)

    w_ratio = (global_w_imp / total_impact) * 100
    c_dyn_ratio = (global_c_dyn_imp / total_impact) * 100
    c_static_ratio = (global_c_static_imp / total_impact) * 100

    return w_ratio, c_dyn_ratio, c_static_ratio

w_r, c_dyn_r, c_static_r = global_feature_importance(model, dataset, edge_index_dict, target_var_idx)
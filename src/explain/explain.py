import torch

import numpy as np
import pandas as pd

from data.load_data import build_edge_index_dict
from local_explainer import LocalExplanation
from global_explainer import GlobalExplanation


dir_info = {
    'city_to_water': r'D:\Program\HTGNN\data/info_data/city_to_water.csv',
    'water_to_water':  r'D:\Program\HTGNN\data/info_data/water_to_water.csv',
    'water_points':  r'D:\Program\HTGNN\data/info_data/water_points.csv',
    'city_points': r'D:\Program\HTGNN\data/info_data/city_points.csv'}

dataset = torch.load(r'D:\Program\HTGNN\data\dataset\Test_dataset.pt')
model = torch.load(r'D:\Program\HTGNN\OutPut_4h/GruHANModel_B32_H32_L32_P1_dr0.20_lr0.0010/best_model.pt',weights_only=False)
edge_index_dict = build_edge_index_dict(dir_info)
target_water_idx=7
target_var_idx=0

######################### 全局特征重要性 #########################
explainer_glob = GlobalExplanation(model, dataset, edge_index_dict, target_var_idx,device='cuda')
'''
    results = {
        'water': global_w_imp,                      水质特征重要性
        'city_dyn': global_c_dyn_imp,               城市节点动态指标重要性
        'city_static': global_c_static_imp,         城市节点社会经济指标重要性
        'ratios': {                                         归一化各个指标占比
            'water': (global_w_imp / total_impact) * 100,   
            'city_dyn': (global_c_dyn_imp / total_impact) * 100,
            'city_static': (global_c_static_imp / total_impact) * 100
        }
    }
'''
results_glob = explainer_glob.explain()


######################### 分站点全局特征重要性 #########################
explainer_node = LocalExplanation(model, dataset, edge_index_dict, target_var_idx,device='cuda')

'''
    results = {
        'water_node_importance': water_node_importance,
        'city_node_importance': city_total_importance,
        'water_attr': global_water_attr,
        'city_dyn_attr': global_city_dyn_attr,
        'city_static_attr':global_city_static_attr,
    }
'''
results_node = explainer_node.explain(target_water_idx)



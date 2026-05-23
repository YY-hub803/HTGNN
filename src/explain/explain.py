import os
import torch
import pickle
import numpy as np
import pandas as pd
from data.load_data import build_edge_index_dict
from local_explainer import LocalExplanation


dir_info = {
    'city_to_water': r'D:\Program\HTGNN\data/info_data/city_to_water.csv',
    'water_to_water':  r'D:\Program\HTGNN\data/info_data/water_to_water.csv',
    'city_to_city': r'D:\Program\HTGNN\data/info_data/city_to_city.csv',
    'water_points':  r'D:\Program\HTGNN\data/info_data/water_points.csv',
    'city_points': r'D:\Program\HTGNN\data/info_data/city_points.csv'}

dataset = torch.load(r'D:\Program\HTGNN\data\dataset\Test_dataset.pt')
model = torch.load(r'D:\Program\HTGNN\Random42OutPut\GruHANModel_B32_H16_L4_NL2_NH4_lr0.0010\best_model.pt',weights_only=False)
edge_index_dict = build_edge_index_dict(dir_info)
target_var_idx=0        # 0:TP 1:TN
features_nm_dict = {
    'water':["TP"],
    'city_dyn':['Pre','Pet','TEMP'],
    'city_static':['Cropland','Forest','Impervious','Other',"In_TP",'NDVI','Light']
}
explain_vis = 'vis_explain'
os.makedirs(explain_vis, exist_ok=True)
result_folder = 'result'
os.makedirs(result_folder, exist_ok=True)
'''
    results = {
        'water': global_w_imp,                      水质特征重要性[Nw,T,Fw]
        'city_dyn': global_c_dyn_imp,               城市节点动态指标重要性[Nc,T,Fcd]
        'city_static': global_c_static_imp,         城市节点社会经济指标重要性[Nc,Fcs]
    }
'''
######################### 分站点重要性 #########################
for target_idx in range(14):
    local_explainer = LocalExplanation(model, dataset, edge_index_dict,target_idx, target_var_idx,device='cuda')
    results_node = local_explainer.explain()
    with open(os.path.join(result_folder, f"results_{target_idx}.pkl"), "wb") as f:
        pickle.dump(results_node, f)
    # with open(os.path.join(result_folder, f"results_{target_idx}.pkl"), "rb") as f:
    #     results_node = pickle.load(f)
    local_explainer.plot_node_pie(results_node,target_idx,explain_vis)
    local_explainer.plot_node_seq(results_node,target_idx, explain_vis)
    local_explainer.plot_features_importance(results_node,target_idx,features_nm_dict,explain_vis)




import os
import torch
import pickle
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
model = torch.load(r'D:\Program\HTGNN\OutPut\GruHANModel_B16_H32_L32_NL2_NH4_lr0.0010/best_model.pt',weights_only=False)
edge_index_dict = build_edge_index_dict(dir_info)
target_water_idx=7
target_var_idx=0
features_nm_dict = {
    'water':['TP','TN','DO','TEMP','COD'],
    'city_dyn':['Pre','Pet'],
    'city_static':['Cropland','Impervious','Forest','Other','Light','Pop','P_gdp',
                'First_industry','Second_industry','Third_industry','Per_indeustrial_value',
                'cow','Goat','Pig','Poultry','NF','PF','KF','MF','Film']
}
saveFolder = 'vis_explain'
os.makedirs(saveFolder, exist_ok=True)

######################### 全局特征重要性 #########################
glob_explainer = GlobalExplanation(model, dataset, edge_index_dict, target_var_idx,device='cuda')
'''
    results = {
        'water': global_w_imp,                      水质特征重要性[Nw,T,Fw]
        'city_dyn': global_c_dyn_imp,               城市节点动态指标重要性[Nc,T,Fcd]
        'city_static': global_c_static_imp,         城市节点社会经济指标重要性[Nc,Fcs]
    }
'''
# results_global = glob_explainer.explain()
# with open("results_global.pkl", "wb") as f:
#     pickle.dump(results_global, f)
with open("results.pkl", "rb") as f:
    results_global = pickle.load(f)
# 全局特征排序，前十个
glob_explainer.plot_importance(results_global,features_nm_dict,saveFolder)
# 每个节点的时间序列贡献，前16步
for target_idx in range(14):
    glob_explainer.plot_node_seq(results_global, target_idx,saveFolder)

######################### 分站点重要性 #########################
for target_idx in range(14):
    local_explainer = LocalExplanation(model, dataset, edge_index_dict,target_idx, target_var_idx,device='cuda')
    results_node = local_explainer.explain()
    local_explainer.plot_node_pie(results_node,target_idx,saveFolder)




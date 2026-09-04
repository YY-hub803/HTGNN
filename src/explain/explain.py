import os
import torch
import pickle
from local_explainer import  LocalExplanation
from configs import config as cfg
from src.utils.load_data import load_siteInfo, load_timeSeries, load_data

water_nm,city_nm, num_cities,num_water_nodes = load_siteInfo(cfg)
full_date_range, DATE_LENGTH = load_timeSeries(cfg)
_, _, _, _, edge_attr, edge_index_dict = load_data(cfg,full_date_range,DATE_LENGTH,num_cities)

dataset = torch.load(r'D:\Program\HTGNN\data\dataset\ALL_dataset.pt')
model = torch.load(r'D:\Program\HTGNN\OutPut18\GruEAHGTModel_B8_H32_L12_NL2_NH4_lr0.0010\best_model.pt',weights_only=False)
target_var_idx=1        # 0:TP 1:TN
features_nm_dict = {
    'water':["TP","TN"],
    'city_dyn':['Pre','TEMP'],
    'city_static':['ALP','ISA','FA',"FERT",'GDPpc','Pop','NTL']
}
explain_vis = '830vis_explain_TN'
os.makedirs(explain_vis, exist_ok=True)
result_folder = '830result_TN'
os.makedirs(result_folder, exist_ok=True)
'''
    results = {
        'water': global_w_imp,                      水质特征重要性[Nw,T,Fw]
        'city_dyn': global_c_dyn_imp,               城市节点动态指标重要性[Nc,T,Fcd]
        'city_static': global_c_static_imp,         城市节点社会经济指标重要性[Nc,Fcs]
    }
'''
# explainer = GlobalExplanation(model, dataset, edge_index_dict, target_var_idx, device='cuda')
# results_global, sample= explainer.explain()

######################### 分站点重要性 #########################
for target_idx in range(18):
    local_explainer = LocalExplanation(model, dataset, edge_index_dict,target_idx, target_var_idx,device='cuda')
    results_node,sample_importance = local_explainer.explain()
    with open(os.path.join(result_folder, f"results_{target_idx}.pkl"), "wb") as f:
        pickle.dump(results_node, f)
    with open(os.path.join(result_folder, f"importance_{target_idx}.pkl"), "wb") as f:
        pickle.dump(sample_importance, f)
    # with open(os.path.join(result_folder, f"results_{target_idx}.pkl"), "rb") as f:
    #     results_node = pickle.load(f)
    local_explainer.plot_node_pie(results_node,target_idx,explain_vis)
    local_explainer.plot_node_seq(results_node,target_idx, explain_vis)
    local_explainer.plot_features_importance(results_node,target_idx,features_nm_dict,explain_vis)




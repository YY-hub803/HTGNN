import os
import torch
import numpy as np
import pandas as pd
from collections import defaultdict
from ..utils.crit import R2,NSE,RMSE,KGE,MAE

def evaluate(model,Test,y_mean, y_std,
             site_names, num_nodes,Target_Name,
             pred_len, saveFolder,device):

    model.eval()
    model_name = model.__class__.__name__
    nF = model.output_size
    if saveFolder is not None:
        if not os.path.exists(saveFolder):
            os.makedirs(saveFolder)
        runFile = os.path.join(saveFolder, f'{model_name}_perform.csv')
        rf = open(runFile, 'w')
    else:
        rf = None

    # 追踪当前滑窗在测试集时间轴上的绝对起始位置
    time_to_preds = defaultdict(list)
    time_to_trues = {}
    global_window_idx = 0

    with torch.no_grad():
        for batch in Test:
            batch_y = batch['water'].y
            batch = batch.to(device)
            output = model(batch)
            current_batch_size = int(batch['water'].batch.max()) + 1
            output = output.view(current_batch_size, num_nodes, pred_len,).detach().cpu().numpy()
            y = batch_y.view(current_batch_size, num_nodes, pred_len)
        for b in range(current_batch_size):
            for step in range(pred_len):
                # 物理日历上的绝对时间索引
                target_time_idx = global_window_idx + step
                # 提取该时刻的所有节点预测值 (形状: [num_nodes])
                pred_values = output[b, :, step]
                time_to_preds[target_time_idx].append(pred_values)

                # 真实值只需要记录一次即可
                if target_time_idx not in time_to_trues:
                    time_to_trues[target_time_idx] = y[b, :, step]
            # 当前窗口处理完毕，绝对索引步进 1
            global_window_idx += 1


    pred_dfs = {}
    obs_dfs = {}

    for i, var_name in enumerate(Target_Name):
        print(f"\n--- 评估预测变量: {var_name} ---")
        pred_raw = x_batch_list[:, :, i]  # [N, T]
        obs_raw = y_batch_list[:, :, i]  # [N, T]

        try:
            cur_std = y_std.flat[i] if isinstance(y_std, np.ndarray) else y_std
            cur_mean = y_mean.flat[i] if isinstance(y_mean, np.ndarray) else y_mean
        except:
            cur_std = y_std[:, :, i][0][0]
            cur_mean = y_mean[:, :, i][0][0]

        # 反归一化
        pred_inv = pred_raw * cur_std + cur_mean
        obs_inv = obs_raw * cur_std + cur_mean

        df_pred = pd.DataFrame(pred_inv, index=site_names).T
        df_obs = pd.DataFrame(obs_inv, index=site_names).T

        pred_dfs[var_name] = df_pred
        obs_dfs[var_name] = df_obs

        # 保存结果文件
        if saveFolder:
            filePath = os.path.join(saveFolder, f'Evaluate_{model_name}_{var_name}.csv')
            if os.path.exists(filePath):
                os.remove(filePath)
            df_pred.to_csv(filePath, index=False)
            df_obs.to_csv(os.path.join(saveFolder, f'Obs_{model_name}_{var_name}.csv'), index=False)

        all_valid_obs = []
        all_valid_preds = []

        # --- 站点级评估 ---
        for site in site_names:

            valid_obs = df_obs[site].values
            valid_pred = df_pred[site].values

            all_valid_obs.append(valid_obs)
            all_valid_preds.append(valid_pred)

            r2 = R2(valid_pred, valid_obs)
            rmse = RMSE(valid_pred, valid_obs)
            nse = NSE(valid_pred, valid_obs)
            kge, r, alpha, beta = KGE(valid_pred, valid_obs)
            mae = MAE(valid_pred, valid_obs)

            logStr = f'Variable:{var_name}, Site:{site}, R2:{r2:.3f}, NSE:{nse:.3f}, KGE:{kge:.3f}, MAE:{mae:.3f}, RMSE:{rmse:.3f}'
            print(logStr)
            if rf: rf.write(logStr + '\n')

        # --- 整体评估指标 ---
        if len(all_valid_obs) > 0:
            total_obs = np.concatenate(all_valid_obs)
            total_preds = np.concatenate(all_valid_preds)

            if len(total_obs) > 0:
                total_r2 = R2(total_preds, total_obs)
                total_rmse = RMSE(total_preds, total_obs)
                total_nse = NSE(total_preds, total_obs)
                total_kge, total_r, total_alpha, total_beta = KGE(total_preds, total_obs)
                total_mae = MAE(total_preds, total_obs)

                logStr_overall = f'Variable:{var_name}, == OVERALL ==, R2:{total_r2:.3f}, NSE:{total_nse:.3f}, KGE:{total_kge:.3f}, MAE:{total_mae:.3f}, RMSE:{total_rmse:.3f}'
                print(logStr_overall)
                if rf: rf.write(logStr_overall + '\n')

    if rf: rf.close()

    return pred_dfs, obs_dfs

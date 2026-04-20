import os
import torch
import numpy as np
import pandas as pd
from ..utils.crit import R2,NSE,RMSE,KGE,MAE

def evaluate_model(model, test_loader, device='cpu'):
    model.eval()
    all_preds = []
    all_trues = []

    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(device)
            out = model(batch)

            # 将张量转移回 CPU 并转换为 Numpy 数组用于评估
            all_preds.append(out.cpu().numpy())
            all_trues.append(batch['water'].y.cpu().numpy())

    # 拼接所有的预测值和真实值
    all_preds = np.vstack(all_preds)
    all_trues = np.vstack(all_trues)

    # 如果你在特征预处理时对 Y 进行了归一化，这里需要执行【反标准化】操作
    # 例如: all_preds = all_preds * Y_std + Y_mean

    rmse = np.sqrt(mean_squared_error(all_trues, all_preds))
    mae = mean_absolute_error(all_trues, all_preds)

    print("\n=== 测试集评估结果 ===")
    print(f"RMSE (均方根误差): {rmse:.4f}")
    print(f"MAE  (平均绝对误差): {mae:.4f}")

    return all_preds, all_trues, rmse, mae



def Prediction(model,Test,y_mean, y_std, sites_ID, saveFolder, Target_Name, device, seq_len, pred_len,batch_size):
    """
    时空预测推理函数
    :param seq_len: 历史回溯窗口长度 (输入模型的序列长度)
    :param pred_len: 预测未来步长 (模型输出的序列长度)
    """
    model.eval()
    model_name = model.__class__.__name__

    if saveFolder is not None:
        if not os.path.exists(saveFolder):
            os.makedirs(saveFolder)
        runFile = os.path.join(saveFolder, f'{model_name}_forecast_perform.csv')
        rf = open(runFile, 'w')
    else:
        rf = None

    if isinstance(x, torch.Tensor):
        x = x.cpu().numpy()
    N_nodes, T_total, n_features = x.shape
    out_dim = model.ny if hasattr(model, 'ny') else len(Target_Name)

    print(f"启动时空预测模式... 总时长: {T_total}, 历史窗口: {seq_len}, 预测步长: {pred_len}")

    # --- 滑窗预测及聚合 ---

    prediction_sum = np.zeros((N_nodes, T_total, out_dim))
    prediction_counts = np.zeros((N_nodes, T_total, out_dim))

    # 确保有足够的长度进行至少一次预测
    total_steps = T_total - seq_len - pred_len + 1
    if total_steps <= 0:
        raise ValueError("序列总长度不足以划分输入窗口和预测窗口！")

    start_indices = np.arange(0, total_steps, 1)  # 步长为1，实现最大重叠预测
    total_batches = (len(start_indices) + batch_size - 1) // batch_size
    print(f"开始滑动预测... 总样本数: {len(start_indices)}, 总 Batch 数: {total_batches}")

    with torch.no_grad():
        for batch_idx, i in enumerate(range(0, len(start_indices), batch_size)):
            batch_starts = start_indices[i: i + batch_size]

            # 1. 构建 Batch 输入数据 (仅取 seq_len 长度作为输入)
            x_batch_list = []
            for start in batch_starts:
                x_batch_list.append(x[:, start: start + seq_len, :])

            x_batch_tensor = torch.tensor(np.array(x_batch_list), dtype=torch.float32).to(device)

            # 2. 模型推理 -> 预期输出形状: [Batch, N_nodes, pred_len, out_dim]
            if model_name in ("PhysicsSTNNModel"):
                batch_preds = model(x_batch_tensor, A_list)
            elif model_name in ("LSTMModel", "GcnLstmModel"):
                batch_preds = model(x_batch_tensor)


            batch_preds = batch_preds.detach().cpu().numpy()
            # 3. 将预测结果累加到对应的时间轴上 (时间轴向后偏移 seq_len)
            for j, start in enumerate(batch_starts):
                pred_start = start + seq_len
                pred_end = pred_start + pred_len
                prediction_sum[:, pred_start:pred_end, :] += batch_preds[j]
                prediction_counts[:, pred_start:pred_end, :] += 1

            if (batch_idx + 1) % 5 == 0 or (batch_idx + 1) == total_batches:
                print(f"进度: Batch {batch_idx + 1}/{total_batches} 已完成...")

    print("滑动预测完成，正在聚合平均并进行反归一化...")

    prediction_counts[prediction_counts == 0] = 1
    final_outputs = prediction_sum / prediction_counts  # [N, T, Out]

    site_names = sites_ID["P_nm"].values if isinstance(sites_ID, pd.DataFrame) else sites_ID
    forecast_dfs = {}
    obs_dfs = {}

    for i, var_name in enumerate(Target_Name):
        print(f"\n--- 评估预测变量: {var_name} ---")

        pred_raw = final_outputs[:, :, i]  # [N, T]
        obs_raw = y[:, :, i]  # [N, T]

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

        # 清洗真实值：0值或Nan视作缺失
        df_obs_clean = df_obs.replace(0, np.nan)

        forecast_dfs[var_name] = df_pred
        obs_dfs[var_name] = df_obs

        # 保存结果文件
        if saveFolder:
            filePath = os.path.join(saveFolder, f'forecast_{model_name}_{var_name}.csv')
            if os.path.exists(filePath):
                os.remove(filePath)
            df_pred.to_csv(filePath, index=False)

        all_valid_obs = []
        all_valid_preds = []

        # --- 站点级评估 ---
        for site in site_names:

            if site in ("GL","DTMDQ"):
                continue
            mask = (~np.isnan(df_obs_clean[site])) & (~np.isnan(df_pred[site]))
            # 补充排除未预测部分(即原始值为0, count=1计算出来的无效值)
            valid_time_mask = np.arange(T_total) >= seq_len
            mask = mask & valid_time_mask

            if np.sum(mask) < 2: continue

            valid_obs = df_obs_clean[site][mask].values
            valid_pred = df_pred[site][mask].values

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

    return forecast_dfs, obs_dfs

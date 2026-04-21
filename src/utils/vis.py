import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates


plt.style.use('seaborn-v0_8-white')
plt.rcParams['font.family'] = ['Times New Roman',"SimSun",'SimHei']
plt.rcParams['axes.unicode_minus'] = False
def visualize_loss(saveFolder,lossFun_name):
    """
    读取 run_printLoss.csv 并绘制 Loss 曲线
    """
    log_path = os.path.join(saveFolder, 'run_printLoss.csv')
    if not os.path.exists(log_path):
        print(f"错误：找不到日志文件 {log_path}")
        return

    epochs = []
    train_losses = []
    val_losses = []

    try:
        with open(log_path, 'r') as f:
            lines = f.readlines()

        for line in lines:
            line = line.strip()
            if not line: continue

            # 解析日志格式：
            # Epoch 1, time 0.50, RMSE_train 0.123,RMSE_val 0.145,LR:0.001000
            parts = line.split(',')

            # 提取数据 (根据你的 format 格式进行解析)
            # parts[0] -> "Epoch 1"
            ep = int(parts[0].split()[1])

            # parts[2] -> " RMSE_train 0.123" (注意可能有空格)
            tr_loss = float(parts[2].strip().split()[1])

            # parts[3] -> "RMSE_val 0.145"
            val_loss = float(parts[3].strip().split()[1])

            epochs.append(ep)
            train_losses.append(tr_loss)
            val_losses.append(val_loss)

        # 开始绘图
        plt.figure(figsize=(10, 6))
        plt.plot(epochs, train_losses, label=f'Train {lossFun_name}', color='blue', linewidth=2)
        plt.plot(epochs, val_losses, label=f'Val {lossFun_name}', color='orange', linewidth=2)

        plt.title('Training and Validation Loss Curves')
        plt.xlabel('Epoch')
        plt.ylabel(f'{lossFun_name}')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)

        # 保存图片
        img_path = os.path.join(saveFolder, 'loss_curve.png')
        plt.savefig(img_path, dpi=300)
        plt.close()
        print(f"Loss 曲线已保存至: {img_path}")

    except Exception as e:
        print(f"绘图失败: {e}")
        print("请检查日志文件格式是否被修改。")


def vis_filled(obs_input, pred_input, full_date_range, save_floder,var_nm):
    global y_label

    if full_date_range is None:
        if isinstance(obs_input, pd.DataFrame):
            full_date_range = obs_input.index
        else:
            raise ValueError("请提供 full_date_range 或 确保输入 DataFrame 包含时间索引")

    save_path = os.path.join(save_floder, f'{var_nm}')
    if not os.path.exists(save_path):
        # 创建文件夹，如果有必要会创建中间目录
        os.makedirs(save_path, exist_ok=True)
    else:
        shutil.rmtree(save_path, ignore_errors=True)
        os.makedirs(save_path, exist_ok=True)

    y_label = "Conc (mg/L)"

    # 确保时间轴是 datetime 格式 (防止绘图报错)
    full_date_range = pd.to_datetime(full_date_range)
    columns_nm = obs_input.columns

    for siteid in columns_nm:
        print(f"正在绘图: {siteid} ...")

        # 提取真实值和模拟值
        obs_values = obs_input[siteid]
        sim_values = pred_input[siteid]
        mask_obs = pd.notna(obs_values)
        obs_valid = obs_values[mask_obs]
        dates_valid = full_date_range[mask_obs]
        # 创建主图和放大图的 Axes
        fig, ax = plt.subplots(figsize=(12, 6))

        # Layer 1: 绘制模型预测/插补值 (红色连续线)
        ax.plot(full_date_range, sim_values,
                color='#d62728', linestyle='-', linewidth=1.2, alpha=0.8,
                label='Model', zorder=1)
        # Layer 2: 绘制真实观测值 (空心圆点)
        ax.plot(dates_valid, obs_valid,
                color='black', linestyle='--', linewidth=1.2, alpha=0.8,
                label='Observed Data', zorder=2)


        ax.set_title(f"Water Quality Pred Results - Site: {siteid}", fontsize=14, fontweight='bold')
        ax.set_ylabel(y_label, fontsize=12)
        ax.set_xlabel("Date", fontsize=12)

        ax.legend(loc='best', frameon=True, fancybox=True, framealpha=0.9)

        ax.grid(True, which='major', linestyle='--', alpha=0.5)

        # 设置X轴日期格式
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        ax.xaxis.set_major_locator(mdates.AutoDateLocator())
        plt.xticks(rotation=30, ha='right')

        plt.tight_layout()

        if save_floder:
            file_nm = f"{siteid}_{var_nm}.png"
            plt.savefig(os.path.join(save_path,file_nm), dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.show()
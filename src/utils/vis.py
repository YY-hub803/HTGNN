import os
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

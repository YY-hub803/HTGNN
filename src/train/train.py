import os
import time
import torch
from ..utils import vis

def saveModel(outFolder, model, epoch, modelName='model'):
    modelFile = os.path.join(outFolder, modelName + '_ep' + str(epoch) + '.pt')
    torch.save(model, modelFile)

def loadModel(outFolder, epoch, modelName='model'):
    modelFile = os.path.join(outFolder, modelName + '_ep' + str(epoch) + '.pt')
    model = torch.load(modelFile, weights_only=False)
    return model


def train(model, Train,Val, LossFun, num_epochs,base_lr,saveFolder,device):

    model = model.to(device)
    LossFun = LossFun.to(device)
    optim = torch.optim.Adam(model.parameters(),lr=base_lr, weight_decay=1e-5)


    model_name = model.__class__.__name__
    lossFun_name = LossFun.__class__.__name__

    if saveFolder is not None:
        if not os.path.isdir(saveFolder):
            os.makedirs(saveFolder)
        runFile = os.path.join(saveFolder, f'run_printLoss.csv')
        rf = open(runFile, 'w+')

    pltRMSE_train = []
    pltRMSE_val = []

    early_stop_counter = 0
    early_stop_patience = 10
    min_delta = 1e-4

    best_val_loss = float('inf')

    print(f"\n--- 开始训练 {model_name} 模型 ({device}) ---")
    for epoch in range(1,num_epochs+1):
        t0 = time.time()
        model.train()
        total_train_loss = 0

        for batch in Train:

            batch = batch.to(device)
            optim.zero_grad()
            outputs = model(batch)
            # 只需要对水质节点计算损失
            loss = LossFun(outputs, batch['water'].y)


            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optim.step()

            total_train_loss += loss.item() * batch.num_graphs

        avg_train_loss = total_train_loss / len(Train.dataset)
        pltRMSE_train.append([epoch, avg_train_loss])
        #----------------------------------------------------------------------------------#

        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for batch in Val:
                batch = batch.to(device)
                outputs = model(batch)

                loss_test = LossFun(outputs, batch['water'].y)
                total_val_loss += loss_test.item() * batch.num_graphs

            avg_val_loss = total_val_loss / len(Val.dataset)
            pltRMSE_val.append([epoch, avg_val_loss])


            if avg_val_loss < best_val_loss - min_delta:
                best_val_loss = avg_val_loss
                early_stop_counter = 0

                if saveFolder is not None:
                    modelFile = os.path.join(saveFolder, 'best_model.pt')
                    torch.save(model, modelFile)
            else:
                early_stop_counter += 1
                print(f"EarlyStopping counter: {early_stop_counter}/{early_stop_patience}")

                if early_stop_counter >= early_stop_patience:
                    print(f"\n 验证集 loss 连续 {early_stop_patience} 个 epoch 未下降，提前停止训练")
                    break
            current_lr = optim.param_groups[0]['lr']
            if current_lr < 1.1e-6 and early_stop_counter >= 3:
                print(f"\nSTOP: 学习率已降至最低 ({current_lr}) 且 Loss 无提升，提前结束。")
                break

        # printing loss
        logStr = ('Epoch {}, time {:.2f}, {}_train {:.3f}, {}_val {:.3f},LR {:.6f}'.format(
            epoch, time.time() - t0, lossFun_name,avg_train_loss,lossFun_name,avg_val_loss,optim.param_groups[0]['lr']))
        logStr_screen = ('Epoch {}, time {:.2f}, {}_train {:.3f}, {}_val {:.3f},LR {:.6f}'.format(
            epoch, time.time() - t0, lossFun_name,avg_train_loss,lossFun_name,avg_val_loss,optim.param_groups[0]['lr']))

        print(logStr_screen)
        if saveFolder is not None:
            rf.write(logStr + '\n')

    if saveFolder is not None:
        rf.close()
        vis.visualize_loss(saveFolder,lossFun_name)
    return model


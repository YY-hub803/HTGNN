import torch


def split_dataset(X, Y, X_city,X_city_static,train_ratio=0.6, val_ratio=0.2):
    """
    按比例切分数据集为 Train (60%), Val (20%), Test (20%)
    返回切割后的数据以及对应的 Mask（掩码在 PyG 中很常用）
    """
    Total_seq = X.shape[1]
    train_end = int(Total_seq * train_ratio)
    val_end = train_end + int(Total_seq * val_ratio)

    X_scaled = X.clone()
    Y_scaled = Y.clone()
    X_city_scaled = X_city.clone()
    X_static_scaled = X_city_static.clone()

    X_train = X_scaled[:,:train_end,:]
    X_city_train = X_city_scaled[:,:train_end,:]
    Y_train = Y_scaled[:,:train_end,:]

    x_mean = X_train.mean(dim=(0,1), keepdim=True)
    x_std = X_train.std(dim=(0,1), keepdim=True)

    y_mean = Y_train.mean(dim=(0,1), keepdim=True)
    y_std = Y_train.std(dim=(0,1), keepdim=True)

    x_city_mean = X_city_train.mean(dim=(0,1), keepdim=True)
    x_city_std = X_city_train.std(dim=(0,1), keepdim=True)

    x_static_mean = X_static_scaled.mean(dim=(0,1), keepdim=True)
    x_static_std = X_static_scaled.std(dim=(0,1), keepdim=True)

    x_std[x_std == 0] = 1e-8
    y_std[y_std == 0] = 1e-8
    x_city_std[x_city_std == 0] = 1e-8
    x_static_std[x_static_std == 0] = 1e-8

    X_scaled = (X_scaled - x_mean) / x_std
    Y_scaled = (Y_scaled - y_mean) / y_std
    X_city_scaled = (X_city_scaled - x_city_mean) / x_city_std
    X_static_scaled = (X_static_scaled - x_static_mean) / x_static_std

    data_splits = {
        'train_x': X_scaled[:,:train_end,:],
        'train_y': Y_scaled[:,:train_end,:],
        'val_x': X_scaled[:,train_end:val_end,:],
        'val_y': Y_scaled[:,train_end:val_end,:],
        'test_x': X_scaled[:,val_end:,:],
        'test_y': Y_scaled[:,val_end:,:],
        'train_x_city': X_city_scaled[:,:train_end,:],
        'val_x_city': X_city_scaled[:,train_end:val_end,:],
        'test_x_city': X_city_scaled[:,val_end:,:],
        'X_city_static': X_static_scaled
    }
    train_stats = {
        "x_mean": x_mean.tolist(),
        "x_std": x_std.tolist(),
        "y_mean": y_mean.tolist(),
        "y_std": y_std.tolist(),
        'x_city_mean': x_city_mean.tolist(),
        'x_city_std': x_city_std.tolist(),
        'x_static_mean': x_static_mean.tolist(),
        'x_static_std': x_static_std.tolist(),
    }

    print(f"切分完成: 训练集 {train_end} 条, 验证集 {val_end - train_end} 条, 测试集 {Total_seq - val_end} 条")
    return data_splits, train_stats


def create_sliding_windows(X, Y,x_city, window_size,pred_len):
    xs, ys,xs_city = [], [],[]
    T = X.size(1)
    for t in range(T - window_size-pred_len+1):
        xs.append(X[:,t:t+window_size,:])
        ys.append(Y[:,t:t+window_size:t+window_size+pred_len,:])
        xs_city.append(x_city[:,t:t+window_size,:])

    X_seq = torch.stack(xs)
    Y_seq = torch.stack(ys)
    X_city_seq = torch.stack(xs_city)
    print(f"滑动窗口生成完毕！")
    print(f"输入 X 形状: {X_seq.shape} -> (Samples,Num_sites,Sequence_length,Num_Features)")
    print(f"标签 Y 形状: {Y_seq.shape} -> (Samples,Num_sites,Pred_length,Num_Targets)")
    return X_seq, Y_seq,X_city_seq

def get_windows(X,Y,X_city,X_city_static,train_ratio, val_ratio,window_size, pred_len):
    data_splits, train_stats = split_dataset(X, Y,X_city,X_city_static, train_ratio, val_ratio)
    Train_sample_x,Train_sample_y,Train_sample_x_city = create_sliding_windows(data_splits['train_x'],data_splits['train_y'],data_splits['train_x_city'], window_size, pred_len)
    Val_sample_x,Val_sample_y ,Val_sample_x_city= create_sliding_windows(data_splits['val_x'], data_splits['val_y'],data_splits['val_x_city'], window_size, pred_len)
    Test_sample_x,Test_sample_y,Test_sample_x_city = create_sliding_windows(data_splits['test_x'], data_splits['test_y'], data_splits['test_x_city'],window_size, pred_len)

    Sample_data = {
        'train_x': Train_sample_x,
        'train_y': Train_sample_y,
        'val_x': Val_sample_x,
        'val_y': Val_sample_y,
        'test_x': Test_sample_x,
        'test_y': Test_sample_y,
        'train_x_city': Train_sample_x_city,
        'val_x_city': Val_sample_x_city,
        'test_x_city': Test_sample_x_city,
        'X_city_static':data_splits['X_city_static'],
    }

    return Sample_data,data_splits, train_stats

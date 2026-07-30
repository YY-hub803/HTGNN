import torch
import json

def split_dataset(window_input):
    X, Y, X_city, X_city_static, edge_attr, TRAIN_RATIO, VAL_RATIO = window_input

    DATA_LENGTH = X.shape[1]
    TRAIN_END = int(DATA_LENGTH * TRAIN_RATIO)
    VAL_END = TRAIN_END + int(DATA_LENGTH * VAL_RATIO)
    data = (X, Y, X_city, X_city_static, edge_attr)
    var_name = ('x', 'y', 'x_city', 'x_static', 'edge_attr')

    def clone_data():
        clone_data = {}
        for name, item in zip(var_name,data):
            clone_data[f'{name}_scaled'] = item.clone()
        return clone_data
    clone_data = clone_data()
    def get_train_data():
        train_data = {}
        for name, value in zip(var_name,clone_data.values()):
            if name == 'x_static':
                train_data[f'{name}_train'] = value[:TRAIN_END]
            else:
                train_data[f'{name}_train'] = value[:,:TRAIN_END,:]

        return train_data
    train_data = get_train_data()

    def get_stats():
        train_stats = {}
        for name, value in zip(var_name,train_data.values()):
            train_stats[f'{name}_mean'] = (value.mean(dim=(0,1), keepdim=True))
            train_stats[f'{name}_std'] = (value.std(dim=(0,1), keepdim=True)+1e-8)
        return train_stats
    train_stats = get_stats()
    def get_normalized_data():
        normalized_data = {}
        for name, value in zip(var_name,clone_data.values()):
            normalized_data[f'{name}_normal'] = (value - train_stats[f'{name}_mean']) / train_stats[f'{name}_std']
        return normalized_data
    normalized_data = get_normalized_data()

    def get_splits_data():
        data_splits = {}
        split_ranges = {
            'train': (0, TRAIN_END),
            'val': (TRAIN_END, VAL_END),
            'test': (VAL_END, DATA_LENGTH)
        }
        for split in ['train','val','test']:
            start, end = split_ranges[split]
            for name, value in zip(var_name, normalized_data.values()):
                if name == 'x_static':
                    data_splits[f'{split}_{name}'] = value[start:end]
                else:
                    data_splits[f'{split}_{name}'] = value[:,start:end,:]

        return data_splits
    data_splits = get_splits_data()
    print(f"切分完成: 训练集 {TRAIN_END} 条, 验证集 {VAL_END - TRAIN_END} 条, 测试集 {DATA_LENGTH - VAL_END} 条")

    def save_train_stats(train_stats, save_path='data/dataset/train_stats.json'):
        """保存训练统计量到 JSON 文件"""
        stats_dict = {k: v.tolist() if hasattr(v, 'tolist') else v
                    for k, v in train_stats.items()}
        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(stats_dict, f, indent=4)
        print(f"训练统计量已保存到: {save_path}")
    save_train_stats(train_stats)

    return data_splits, train_stats


def create_sliding_windows(data_splits,window_size,pred_len):
    Sample_data = {}
    for split in ['train','val','test']:
        X = data_splits[split + '_x']
        Y = data_splits[split + '_y']
        x_city = data_splits[split + '_x_city']
        x_static = data_splits[split + '_x_static']
        edge_attr = data_splits[split + '_edge_attr']
        xs, ys, xs_city, xs_static, edge_attr_seq = [], [], [], [], []
        T = X.size(1)
        for t in range(T - window_size):
            xs.append(X[:,t:t+window_size,:])
            ys.append(Y[:,t+window_size,:])
            xs_city.append(x_city[:,t:t+window_size,:])
            xs_static.append(x_static[t+window_size])
            edge_attr_seq.append(edge_attr[:,t:t+window_size,:])
        X_seq = torch.stack(xs)
        Y_seq = torch.stack(ys)
        X_city_seq = torch.stack(xs_city)
        X_static_seq = torch.stack(xs_static)
        edge_attr_seq = torch.stack(edge_attr_seq)

        Sample_data[split + '_x'] = X_seq
        Sample_data[split + '_y'] = Y_seq
        Sample_data[split + '_x_city'] = X_city_seq
        Sample_data[split + '_x_static'] = X_static_seq
        Sample_data[split + '_edge_attr'] = edge_attr_seq
    print(f"滑动窗口生成完毕！")
    return Sample_data

def get_windows(window_input,cfg):
    window_size, pred_len = cfg.get("train_config")['history'],cfg.get("train_config")['pred']
    data_splits, train_stats = split_dataset(window_input)
    Sample_data= create_sliding_windows(data_splits,window_size, pred_len)
    return Sample_data,data_splits, train_stats



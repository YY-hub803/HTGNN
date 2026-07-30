import pandas as pd
import numpy as np
import torch
import random
from src.models import model,myModel
from src.utils import crit
import os
import shutil


MODEL_FACTORY = {
    "GruHANModel": model.GruHANModel,
    'GruModel': model.GruModel,
    "MeteoModel":model.MeteoModel,
    "GnnModel":model.GnnModel,
    "SocioEcoModel":model.SocioEcoModel,
    "GruEAHGTModel":myModel.GruEAHGTModel,
}
LOSS_FACTORY = {
    "MSE": crit.MSELoss,
    "MAE": crit.MAELoss,
    "RMSE": crit.RMSELoss,
    "Huber": crit.HuberLoss,
    "MixLoss": crit.MixLoss,
}


def set_seeds(seed_value):
    """Set seeds for reproducibility."""
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed(seed_value)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def set_device():
    """Set device for training."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
def set_model(model_name):
    """Set model."""
    model = MODEL_FACTORY[model_name]
    return model
def set_loss(loss_name):
    """Set loss function."""
    loss = LOSS_FACTORY[loss_name]
    return loss

def set_all(cfg):
    DEVICE = set_device()
    model = set_model(cfg.get("train_config")['model'])
    Loss = set_loss(cfg.get("train_config")['loss_fun'])
    DIR_MODEL = "%s_B%d_H%d_L%d_NL%d_NH%d_lr%.4f" % (
        cfg.get("train_config")['model'],
        cfg.get("train_config")['batch'],
        cfg.get("train_config")['hidden'],
        cfg.get("train_config")['history'],
        cfg.get("train_config")['num_layers'],
        cfg.get("train_config")['num_heads'],
        cfg.get("train_config")['lr'],
    )
    OUTPUT_DIR = cfg.get("output_dir")
    check_folder(OUTPUT_DIR)
    DIR_OUTPUT = os.path.join(OUTPUT_DIR, DIR_MODEL)
    check_folder(DIR_OUTPUT)
    VIS_FOLDER = os.path.join(DIR_OUTPUT, 'visualization')
    check_folder(VIS_FOLDER)
    return model,Loss,DEVICE,DIR_MODEL,DIR_OUTPUT,VIS_FOLDER


def load_timeseries(dict_data, date_length):
    """Load data_1D from time-series inputs"""
    data_list = []
    for path in dict_data.values():
        loaded_data = pd.read_csv(path, delimiter=",").to_numpy()
        _,number = loaded_data.shape
        reshaped_data = np.reshape(np.ravel(loaded_data.T), (number, date_length, 1))
        data_list.append(reshaped_data)
    return np.concatenate(data_list, axis=2)

def load_attribute(dict_data):
    """Load data from constant attributes"""
    data_dict = {}
    for key,value in dict_data.items():
        data_dict[key] = np.loadtxt(value, delimiter=",", skiprows=1)
    return data_dict

def check_folder(folder):
    contains_vis = any(keyword.lower() in folder.lower() for keyword in ['vis', 'visual', 'visualization'])
    if contains_vis:
        if not os.path.exists(folder):
            os.makedirs(folder, exist_ok=True)
            print(f"成功创建模型输出文件夹: {folder}")
        else:
            shutil.rmtree(folder, ignore_errors=True)
            os.makedirs(folder, exist_ok=True)
    else:
        if not os.path.exists(folder):
            os.makedirs(folder, exist_ok=True)
            print(f"成功创建模型输出文件夹: {folder}")

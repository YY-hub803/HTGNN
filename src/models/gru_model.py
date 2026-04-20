import torch
import torch.nn as nn
import torch.nn.functional as F

class GRULayer(nn.Module):
    def __init__(self, input_size,hidden_size,num_layers, drop_rate):
        super().__init__()
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=drop_rate if num_layers > 1 else 0
        )
    def forward(self, x):
        _,(lstm_out,_) = self.gru(x)
        return lstm_out[-1,:,:]
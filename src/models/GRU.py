import torch
import torch.nn as nn
import torch.nn.functional as F

class GRULayer(nn.Module):
    def __init__(self, input_size,hidden_size,num_layers,pred_len, drop_rate):
        super().__init__()
        self.hidden_size=hidden_size
        self.pred_len = pred_len
        self.fc = nn.Linear(input_size,hidden_size)
        self.gru = nn.GRU(
            input_size=self.hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=drop_rate if num_layers > 1 else 0
        )
        self.dense = nn.Linear(self.hidden_size, self.pred_len*self.ny)
    def forward(self, x):
        B, N, T, _ = x.shape
        x_in = x.reshape(B * N, T, -1)
        lstm_out,_ = self.gru(x_in)
        mlp_out = self.dense(lstm_out[:,-1,:])
        return mlp_out.reshape(B, N, self.pred_len, self.ny)
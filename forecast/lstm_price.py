import torch
import torch.nn as nn

class PriceLSTM(nn.Module):
    def __init__(self, input_dim = 1, hidden_dim = 64, num_layers = 2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_dim,
            hidden_dim,
            num_layers,
            batch_first=True,
        )
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        h_t = out[:, -1, :]
        y_hat = self.fc(h_t)
        return y_hat
import torch
from torch.utils.data import Dataset

class StockDataset(Dataset):
    def __init__(self, data, window=20):
        self.data = data
        self.window = window

    def __len__(self):
        return len(self.data) - self.window

    def __getitem__(self, idx):
        x = self.data[idx:idx + self.window]
        y = self.data[idx + self.window]
        return torch.tensor(x, dtype=torch.float32), \
               torch.tensor(y, dtype=torch.float32)

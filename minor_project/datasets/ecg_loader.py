import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset

class ECGDataset(Dataset):
    def __init__(self, csv_path, max_samples=None):
        df = pd.read_csv(csv_path, header=None)
        
        # Last column is the label
        self.labels = df.iloc[:, -1].values.astype(int)
        self.signals = df.iloc[:, :-1].values.astype(float)
        
        # Collapse to binary: 0 = Normal, 1 = Abnormal
        self.labels = (self.labels > 0).astype(int)
        
        # Balance classes
        normal_idx = np.where(self.labels == 0)[0]
        abnormal_idx = np.where(self.labels == 1)[0]
        n = min(len(normal_idx), len(abnormal_idx))
        if max_samples:
            n = min(n, max_samples // 2)
        
        np.random.seed(42)
        chosen = np.concatenate([
            np.random.choice(normal_idx, n, replace=False),
            np.random.choice(abnormal_idx, n, replace=False)
        ])
        np.random.shuffle(chosen)
        
        self.signals = self.signals[chosen]
        self.labels = self.labels[chosen]
        
        print(f"Loaded {len(self.labels)} samples | Normal: {(self.labels==0).sum()} | Abnormal: {(self.labels==1).sum()}")

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        signal = self.signals[idx]
        # Reshape to (188, 1) — single feature, 188 timesteps
        signal = (signal - signal.mean()) / (signal.std() + 1e-5)
        x = torch.tensor(signal, dtype=torch.float32).unsqueeze(-1)
        y = torch.tensor(self.labels[idx], dtype=torch.long)
        return x, y
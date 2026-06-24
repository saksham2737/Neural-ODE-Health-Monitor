import wfdb
import numpy as np
import torch
from torch.utils.data import Dataset

class MITBIHDataset(Dataset):

    def __init__(self, records):
        self.records = records

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):

        record = wfdb.rdrecord(self.records[idx])
        signal = record.p_signal

        signal = np.array(signal, dtype=np.float32)
        signal = torch.tensor(signal)

        label = 0

        return signal, label
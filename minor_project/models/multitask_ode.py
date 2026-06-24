import torch
import torch.nn as nn
from torchdiffeq import odeint

class MultiTaskODE(nn.Module):

    def __init__(self, input_dim, hidden_dim):

        super().__init__()

        self.encoder = nn.GRU(input_dim, hidden_dim)

        self.odefunc = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.Tanh(),
            nn.Linear(64, hidden_dim)
        )

        self.ecg_head = nn.Linear(hidden_dim, 5)
        self.eeg_head = nn.Linear(hidden_dim, 2)
        self.icu_head = nn.Linear(hidden_dim, 2)

    def forward(self, x, t):

        h0, _ = self.encoder(x)

        h0 = h0[-1]

        ht = odeint(self.odefunc, h0, t)

        h_final = ht[-1]

        ecg = self.ecg_head(h_final)
        eeg = self.eeg_head(h_final)
        icu = self.icu_head(h_final)

        return ecg, eeg, icu
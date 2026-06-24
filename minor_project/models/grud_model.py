import torch
import torch.nn as nn

class GRUD(nn.Module):

    def __init__(self, input_size, hidden_size):

        super().__init__()

        self.gru = nn.GRU(input_size, hidden_size)

        self.fc = nn.Linear(hidden_size, 2)

    def forward(self, x):

        out, _ = self.gru(x)

        out = out[-1]

        out = self.fc(out)

        return out
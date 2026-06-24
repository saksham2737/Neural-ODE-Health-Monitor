import torch
import torch.nn as nn
from torchdiffeq import odeint

class ODEFunc(nn.Module):

    def __init__(self, hidden_dim):
        super().__init__()

        self.net = nn.Sequential(
    nn.Linear(hidden_dim, hidden_dim * 2),
    nn.Tanh(),
    nn.Linear(hidden_dim * 2, hidden_dim * 2),
    nn.Tanh(),
    nn.Linear(hidden_dim * 2, hidden_dim)
)

    def forward(self, t, h):
        return self.net(h)


class LatentODE(nn.Module):

    def __init__(self, input_dim, hidden_dim):
        super().__init__()

        self.encoder = nn.GRU(input_dim, hidden_dim, batch_first=True)

        self.odefunc = ODEFunc(hidden_dim)

        self.classifier = nn.Linear(hidden_dim, 2)

    def forward(self, x, t):

        # Encode sequence
        _, h = self.encoder(x)
        h = h.squeeze(0)

        # Solve ODE
        out = odeint(self.odefunc, h, t, method='euler')

        # Take final state
        final_state = out[-1]

        return self.classifier(final_state)
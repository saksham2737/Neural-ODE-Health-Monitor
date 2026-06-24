import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from models.latent_ode import LatentODE

def train(model, dataloader):

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=0.001
    )

    criterion = nn.CrossEntropyLoss()

    for epoch in range(30):

        total_loss = 0

        for x, y in dataloader:

            optimizer.zero_grad()

            t = torch.linspace(0,1,5)

            pred = model(x, t)

            loss = criterion(pred, y)

            loss.backward()

            optimizer.step()

            total_loss += loss.item()

        print("Epoch:", epoch, "Loss:", total_loss)
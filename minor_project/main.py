import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
from models.latent_ode import LatentODE
from datasets.ecg_loader import ECGDataset

def main():
    dataset = ECGDataset("data/mitbih_kaggle/mitbih_train.csv", max_samples=20000)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    model = LatentODE(input_dim=1, hidden_dim=64)
    import os
    if os.path.exists("neural_ode_model.pth"):
        model.load_state_dict(torch.load("neural_ode_model.pth", map_location='cpu'))
        print("Resuming from saved model...")

    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    EPOCHS = 30
    for epoch in range(EPOCHS):
        total_loss = 0
        for x, y in dataloader:
            t = torch.linspace(0, 1, x.shape[1])
            output = model(x, t)
            loss = criterion(output, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}")

    torch.save(model.state_dict(), "neural_ode_model.pth")
    print("Training complete. Model saved.")

if __name__ == "__main__":
    main()
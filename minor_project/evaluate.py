import torch
import numpy as np
from torch.utils.data import DataLoader, Subset
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
from datasets.ecg_loader import ECGDataset
from models.latent_ode import LatentODE
from torch.utils.data import DataLoader

dataset = ECGDataset("data/mitbih_kaggle/mitbih_test.csv")
test_loader = DataLoader(dataset, batch_size=32, shuffle=False)

model = LatentODE(input_dim=1, hidden_dim=64)
model.load_state_dict(torch.load("neural_ode_model.pth", map_location='cpu'))
model.eval()

all_preds = []
all_labels = []

with torch.no_grad():
    for x, y in test_loader:
        t = torch.linspace(0, 1, x.shape[1])
        output = model(x, t)
        preds = torch.argmax(output, dim=1)
        all_preds.extend(preds.numpy())
        all_labels.extend(y.numpy())

all_preds = np.array(all_preds)
all_labels = np.array(all_labels)

print("=== Evaluation Results ===")
print(f"Unique labels in test set: {np.unique(all_labels)}")
print(f"Unique predictions: {np.unique(all_preds)}")
print(f"Accuracy:  {accuracy_score(all_labels, all_preds):.4f}")
print(f"F1 Score:  {f1_score(all_labels, all_preds, average='weighted'):.4f}")
print("\nClassification Report:")
print(classification_report(all_labels, all_preds, target_names=['Normal', 'Abnormal'], labels=[0,1]))
print("Confusion Matrix:")
print(confusion_matrix(all_labels, all_preds, labels=[0,1]))
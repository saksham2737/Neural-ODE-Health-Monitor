from datasets.ecg_loader import ECGDataset

ds = ECGDataset("data/mitbih")
for i in range(5):
    x, y = ds[i]
    print(ds.files[i], y.item())
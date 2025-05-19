# AI4EO - Part 1: Land Cover Classification
## Download Data
1. Go to EO Browser → Analytical → select bands B02, B03, B04, B08 → TIFF (32-bit float) → Download  
2. Rename the files to `data/B02.tif`, `data/B03.tif`, etc.  
3. Run preprocessing:
   ```bash
   python src/data_preprocessing.py --input-dir data/ --output-dir data/processed/

    
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
import rasterio
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
import matplotlib.pyplot as plt

# Directory paths (customize as needed)
DATA_DIR = "data"
RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)

# Thresholds for label generation
NDWI_THRESH = 0.3
NDVI_THRESH = 0.4

# Custom Dataset for multi-index input and auto-labels
class LandCoverDataset(Dataset):
    def __init__(self, inputs, labels, transform=None):
        self.inputs = inputs  # numpy array [N,H,W,channels]
        self.labels = labels  # numpy array [N,H,W]
        self.transform = transform

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        x = self.inputs[idx]
        y = self.labels[idx]
        if self.transform:
            x = self.transform(x)
        return x, y

# Utility to load preprocessed indices
def load_indices(image_path):
    with rasterio.open(image_path) as src:
        return src.read(1)

# Generate dataset arrays from preprocessed GeoTIFFs
def prepare_dataset():
    # Assume preprocessed files named ndvi.tif, ndwi.tif, evi.tif
    ndvi = load_indices(os.path.join(DATA_DIR, 'ndvi.tif'))
    ndwi = load_indices(os.path.join(DATA_DIR, 'ndwi.tif'))
    # Stack indices into a multi-channel input
    H, W = ndvi.shape
    inputs = np.stack([ndvi, ndwi], axis=0)  # shape [2,H,W]

    # Generate labels per-pixel
    labels = np.zeros((H, W), dtype=np.int64)
    # class 1: water
    labels[ndwi > NDWI_THRESH] = 1
    # class 2: vegetation
    labels[(ndvi > NDVI_THRESH) & (ndwi <= NDWI_THRESH)] = 2
    # class 3: urban/other
    labels[(ndvi <= NDVI_THRESH) & (ndwi <= NDWI_THRESH)] = 3

    # Flatten to patches for dataset
    # Here we sample small patches or use whole image as one sample
    # For simplicity treat each pixel as sample
    X = inputs.reshape(2, -1).T  # [N_pixels,2]
    y = labels.reshape(-1)
    return X, y, H, W

# Simple MLP as pixel-wise classifier (placeholder for CNN)
class PixelMLP(nn.Module):
    def __init__(self, in_features=2, num_classes=4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, 16),
            nn.ReLU(),
            nn.Linear(16, 32),
            nn.ReLU(),
            nn.Linear(32, num_classes)
        )
    def forward(self, x):
        return self.net(x)

# Train and evaluate pixel-wise
def train_and_evaluate():
    X, y, H, W = prepare_dataset()
    # Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Convert to tensors
    X_train = torch.from_numpy(X_train).float()
    y_train = torch.from_numpy(y_train).long()
    X_test = torch.from_numpy(X_test).float()
    y_test = torch.from_numpy(y_test).long()

    # DataLoader
    train_ds = torch.utils.data.TensorDataset(X_train, y_train)
    test_ds = torch.utils.data.TensorDataset(X_test, y_test)
    train_loader = DataLoader(train_ds, batch_size=1024, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=1024)

    # Model, loss, optimizer
    model = PixelMLP().to('cpu')
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    # Training loop
    epochs = 10
    for epoch in range(1, epochs+1):
        model.train()
        total_loss = 0
        for xb, yb in train_loader:
            optimizer.zero_grad()
            preds = model(xb)
            loss = criterion(preds, yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * xb.size(0)
        avg_loss = total_loss / len(train_loader.dataset)
        print(f"Epoch {epoch}/{epochs}, Loss: {avg_loss:.4f}")

    # Evaluation
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for xb, yb in test_loader:
            preds = model(xb).argmax(dim=1)
            all_preds.append(preds.numpy())
            all_labels.append(yb.numpy())
    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='weighted')
    print(f"Test Accuracy: {acc:.4f}, F1 Score: {f1:.4f}")

    # Save model
    torch.save(model.state_dict(), os.path.join(RESULTS_DIR, 'pixel_mlp.pth'))

    # Reconstruct full classification map
    preds_full = np.zeros_like(y)
    X_full = torch.from_numpy(X).float()
    with torch.no_grad():
        out_full = model(X_full).argmax(dim=1).numpy()
    preds_full = out_full.reshape(H, W)
    # Save map
    plt.imsave(os.path.join(RESULTS_DIR, 'classification_map.png'), preds_full, cmap='tab20')
    print(f"Classification map saved to {RESULTS_DIR}")

if __name__ == "__main__":
    train_and_evaluate()

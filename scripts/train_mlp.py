import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from models.mlp import SimpleMLP

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 1. Buat dataset sederhana ( misal : titik 2D dalam 2 kelas )
N = 500
X = torch.randn(N, 2)
y = (X[:, 0] * X[:, 1] > 0).long()  # label 1 jika x*y > 0, selainnya 0

# Normalisasi ( mean=0 , std=1 di sini karena konstruksinya , tapi kita tunjukkan caranya )
mean = X.mean(dim=0, keepdim=True)
std = X.std(dim=0, keepdim=True) + 1e-8
X_norm = (X - mean) / std

dataset = TensorDataset(X_norm, y)
loader = DataLoader(dataset, batch_size=32, shuffle=True)

model = SimpleMLP().to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.1)

# 3. Putaran pelatihan ( training loop )
for epoch in range(50):
    model.train()
    total_loss = 0.0
    correct, total = 0, 0

    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)

        # Maju ( Forward )
        logits = model(xb)
        loss = criterion(logits, yb)

        # Mundur ( Backward )
        optimizer.zero_grad()
        loss.backward()

        # Perbarui bobot ( Update )
        optimizer.step()

        total_loss += loss.item() * xb.size(0)
        _, preds = torch.max(logits, 1)
        correct += (preds == yb).sum().item()
        total += yb.size(0)

    avg_loss = total_loss / total
    acc = correct / total
    print(f"Epoch {epoch+1:2d} | Loss: {avg_loss:.4f} | Akurasi: {acc:.4f}")

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

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

# 2. Definisikan Model MLP: input_dim=2 , hidden_dim=16 , num_classes=2
class SimpleMLP(nn.Module):
    def __init__(self, input_dim=2, hidden_dim=16, num_classes=2):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.act = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        # Perhitungan maju ( forward propagation )
        h = self.act(self.fc1(x))
        out = self.fc2(h)
        return out

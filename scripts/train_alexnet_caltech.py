import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from sklearn.metrics import classification_report

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from models.alexnet import AlexNetCaltech
from utils.datasets import CustomImageDataset

# Gunakan GPU jika tersedia , jika tidak gunakan CPU
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DATA_DIR = "data/caltech101"  # Sesuaikan dengan jalur folder Anda
BATCH_SIZE = 32
LR = 1e-4
EPOCHS = 10

# Persiapkan transformasi gambar
data_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# Muat seluruh dataset
full_dataset = CustomImageDataset(root_dir=DATA_DIR,
                                  transform=data_transforms)

# Bagi dataset : 80% untuk latihan , 20% untuk validasi
train_size = int(0.8 * len(full_dataset))
val_size = len(full_dataset) - train_size
train_set, val_set = random_split(full_dataset, [train_size, val_size])

train_loader = DataLoader(train_set, batch_size=BATCH_SIZE,
                          shuffle=True)
val_loader = DataLoader(val_set, batch_size=BATCH_SIZE,
                        shuffle=False)

# Inisialisasi model , fungsi loss , dan optimizer
model = AlexNetCaltech(num_classes=len(full_dataset.classes)).to(DEVICE)
criterion = nn.CrossEntropyLoss()  # fungsi loss
optimizer = optim.Adam(model.parameters(), lr=LR)

# Mulai proses pelatihan
for epoch in range(EPOCHS):
    model.train()
    running_loss = 0.0
    for images, labels in train_loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)

        # Reset gradien , hitung output , hitung loss , dan perbarui bobot
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)

    avg_loss = running_loss / len(train_loader.dataset)

    # Tahap validasi setelah setiap epoch
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (preds == labels).sum().item()

    acc = correct / total
    print(f"Epoch {epoch+1}/{EPOCHS} - Loss: {avg_loss:.4f}, Akurasi Val: {acc:.4f}")

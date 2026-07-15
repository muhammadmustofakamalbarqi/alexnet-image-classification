import torch

@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()  # Matikan dropout dan batch norm saat evaluasi
    correct, total = 0, 0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        logits = model(xb)          # Hitung prediksi
        _, preds = torch.max(logits, 1)  # Ambil kelas dengan skor tertinggi
        correct += (preds == yb).sum().item()  # Hitung prediksi yang benar
        total += yb.size(0)
    acc = correct / total  # Akurasi = jumlah benar / total data
    return acc

# AlexNet Image Classification

Proyek pembelajaran mesin (Modul 1) berisi implementasi beberapa arsitektur jaringan saraf dengan **PyTorch**:

1. **AlexNet** untuk klasifikasi gambar pada dataset **Caltech-101**
2. **MLP (Multi-Layer Perceptron)** sederhana untuk klasifikasi titik 2D
3. **LeNet-MNIST** dengan aplikasi GUI interaktif (Pygame) untuk mengenali tulisan tangan angka

## Struktur Proyek

```
alexnet-image-classification/
├── models/
│   ├── alexnet.py            # Arsitektur AlexNet (AlexNetCaltech)
│   └── mlp.py                # Arsitektur MLP sederhana (SimpleMLP)
├── scripts/
│   ├── train_alexnet_caltech.py  # Skrip pelatihan AlexNet pada Caltech-101
│   └── train_mlp.py              # Skrip pelatihan MLP pada data sintetis 2D
├── utils/
│   ├── datasets.py           # CustomImageDataset (loader folder gambar per kelas)
│   └── training_loops.py     # Fungsi evaluasi akurasi model
├── mnist_app.py              # Aplikasi GUI klasifikasi digit MNIST (LeNet + Pygame)
├── requirements.txt
└── README.md
```

## Instalasi

Disarankan menggunakan virtual environment (Python 3.9+):

```bash
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Linux / macOS

pip install -r requirements.txt
```

> Jika memiliki GPU NVIDIA, pasang PyTorch versi CUDA dari
> https://pytorch.org/get-started/locally/ agar pelatihan lebih cepat.
> Semua skrip otomatis memakai GPU jika tersedia (`torch.cuda.is_available()`).

## Cara Menjalankan

### 1. Melatih AlexNet pada Caltech-101

Unduh dataset [Caltech-101](https://data.caltech.edu/records/mzrjq-6wc02) lalu susun dalam struktur folder berikut (satu subfolder per kelas):

```
data/caltech101/
├── accordion/
│   ├── image_0001.jpg
│   └── ...
├── airplanes/
└── ...
```

Kemudian jalankan:

```bash
python scripts/train_alexnet_caltech.py
```

Konfigurasi (dapat diubah di bagian atas skrip):

| Parameter  | Nilai              |
|------------|--------------------|
| Ukuran input | 224 × 224 (RGB)  |
| Batch size | 32                 |
| Learning rate | 1e-4 (Adam)     |
| Epoch      | 10                 |
| Pembagian data | 80% latih / 20% validasi |

Skrip mencetak *loss* pelatihan dan akurasi validasi setiap epoch.

### 2. Melatih MLP pada data sintetis

MLP dilatih untuk memisahkan titik-titik 2D berdasarkan tanda hasil kali koordinatnya (masalah mirip XOR):

```bash
python scripts/train_mlp.py
```

### 3. Aplikasi GUI Klasifikasi Digit MNIST

Aplikasi interaktif untuk menggambar angka dengan mouse dan mengklasifikasikannya:

```bash
python mnist_app.py
```

- Jika file bobot `mnist_lenet.pth` belum ada, aplikasi otomatis mengunduh dataset MNIST dan melatih model LeNet (15 epoch) terlebih dahulu.
- Setelah jendela terbuka: gambar angka di kanvas putih, klik **Klasifikasi** untuk melihat prediksi dan tingkat keyakinan, klik **Hapus** untuk membersihkan kanvas.

## Arsitektur Model

### AlexNetCaltech (`models/alexnet.py`)
- 5 lapisan konvolusi + ReLU + MaxPooling untuk ekstraksi fitur
- Adaptive Average Pooling (6×6)
- 3 lapisan fully-connected (4096 → 4096 → jumlah kelas) dengan Dropout 0.5 untuk mencegah overfitting

### SimpleMLP (`models/mlp.py`)
- Input 2 dimensi → hidden layer 16 neuron (ReLU) → output 2 kelas

### LeNetMNIST (`mnist_app.py`)
- 4 lapisan konvolusi dengan Batch Normalization dan Dropout
- Classifier fully-connected 256 neuron
- Augmentasi data (rotasi ± 10°, translasi 10%) saat pelatihan

## Dependensi Utama

- `torch`, `torchvision` — pembangunan dan pelatihan model
- `pillow` — pemrosesan gambar
- `scikit-learn` — metrik evaluasi (classification report)
- `pygame` — GUI aplikasi MNIST

## Lisensi

Proyek ini dibuat untuk keperluan pembelajaran (tugas kuliah Pembelajaran Mesin, Semester 4).

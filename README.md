# MNIST Rakam Tanıyıcı 🔢

PyTorch ile eğitilmiş, el yazısı rakamları **%98+ doğrulukla** tanıyan derin öğrenme modeli.

## Hakkında
Bu proje, MNIST veri seti üzerinde 0-9 arası el yazısı rakamları sınıflandırmak için bir sinir ağı eğitir.

## Sonuçlar
| Metrik | Değer |
|--------|-------|
| Eğitim Doğruluğu | %97.98 |
| Test Doğruluğu | %98.16 |
| Epoch | 10 |
| Optimizer | Adam |

## Kullanılan Teknolojiler
- Python 3.13
- PyTorch 2.6 (CUDA 12.4)
- NVIDIA RTX 4050 GPU

## Proje Yapısı
```
mnist-classifier/
├── model.py      # Sinir ağı mimarisi
├── train.py      # Eğitim döngüsü
├── utils.py      # Veri yükleme ve hazırlama
└── README.md
```

## Kurulum
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
```

## 🏃 Kullanım
```bash
python train.py
```
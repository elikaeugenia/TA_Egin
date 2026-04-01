# 🛍️ Shopee Comment Sentiment Analysis

Sistem analisis sentimen komentar Shopee menggunakan Arsitektur **TextCNN** dengan preprocessing 7 tahap, data augmentation, dan 5-fold cross-validation.

---

## 📋 Daftar Isi
- [Requirements](#-requirements)
- [Instalasi](#-instalasi)
- [Struktur Folder](#-struktur-folder)
- [Cara Menjalankan](#-cara-menjalankan)
- [Preprocessing Pipeline](#-preprocessing-pipeline)
- [Troubleshooting](#-troubleshooting)

---

## 🔧 Requirements

| Komponen | Versi |
|----------|-------|
| Python | `3.8+` |
| PyTorch | `2.0+` |
| CUDA | `11.8+` (opsional) |

> **Spesifikasi minimum:** RAM 16GB, GPU NVIDIA RTX 3060+ (recommended)

---

## 📦 Instalasi

### 1. Clone & Buat Virtual Environment

```bash
git clone https://github.com/username/shopee-sentiment.git
cd shopee-sentiment

# Windows
python -m venv venv && venv\Scripts\activate

# macOS / Linux
python3 -m venv venv && source venv/bin/activate
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt

# PyTorch dengan CUDA (GPU)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### 3. Download NLTK Data

```bash
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords')"
```

> **Google Colab?** Gunakan `!pip install -q torch datasets transformers nltk scikit-learn openpyxl` lalu mount Google Drive.

---

## 📁 Struktur Folder

```
baruu/
├── code/
│   ├── dataset.xlsx          # Dataset utama
│   ├── datareader.py         # Preprocessing & dataset loader
│   ├── model.py              # Arsitektur model
│   ├── train.py              # Training script
│   ├── evaluate.py           # Evaluasi
│   └── demo.py               # Inference
├── debug/                    # Jupyter notebooks (step-by-step)
├── models/                   # Hasil training (fold_0 - fold_4)
├── outputs/                  # Logs & metrics
├── requirements.txt
└── environment.yml
```

---

## 🎯 Cara Menjalankan

### Jupyter Notebook (Recommended)

```bash
jupyter notebook
```

Jalankan notebook secara berurutan di folder `debug/`:
1. `01_DataReader_Complete.ipynb`
2. `02_Model_Architecture.ipynb`
3. `03_Training_Complete.ipynb`
4. `04_Evaluation_Complete.ipynb`
5. `05_Demo.ipynb`

### Python Scripts

```bash
# Training (5-fold CV)
python code/train.py --multi_fold --epochs 10 --batch_size 32

# Evaluasi
python code/evaluate.py --multi_fold

# Inference / Demo
python code/demo.py --text "Barang bagus banget, pengiriman cepat!"
```

---

## 🔄 Preprocessing Pipeline

```
Raw Text → Casefolding → Cleansing → Phrase Replacement
        → Tokenization → Normalization → Stopwords Removal
        → Augmentation (train only) → IndoBERT Tokenization
```

---

## 📊 Dataset

File: `code/dataset.xlsx`

| Column | Deskripsi |
|--------|-----------|
| `userName` | Nama pengguna |
| `rating` | Rating bintang (1–5) |
| `timestamp` | Waktu review |
| `comment` | Teks komentar |

---

## 🧠 Model

**Base:** `TextCNN`  
**Head:** Dropout(0.1) → Linear(768→256) → Linear(256→2)  
**Task:** Binary sentiment classification (Positif / Negatif)

---

## 🐛 Troubleshooting

| Error | Solusi |
|-------|--------|
| `No module named 'torch'` | `pip install torch --index-url https://download.pytorch.org/whl/cu118` |
| `CUDA out of memory` | Kurangi `batch_size` ke 16 |
| `FileNotFoundError: dataset.xlsx` | Pastikan file ada di `code/dataset.xlsx` |
| `punkt not found` | `python -c "import nltk; nltk.download('punkt')"` |
| GPU tidak terdeteksi | Cek dengan `torch.cuda.is_available()` |


---

## 📖 How to Cite

### APA Style
Ramadhania, E. E. (2025). *Penerapan Arsitektur TextCNN untuk Analisis Sentimen Ulasan Pengguna Aplikasi Shopee di Google Play Store* [Skripsi]. Institut Teknologi Sumatera.

### BibTeX
```bibtex
@thesis{ramadhania2025textcnn,
  author  = {Ramadhania, Elika Eugenia},
  title   = {Penerapan Arsitektur TextCNN untuk Analisis Sentimen Ulasan Pengguna Aplikasi Shopee di Google Play Store},
  school  = {Institut Teknologi Sumatera},
  year    = {2025},
  type    = {Undergraduate Thesis},
  program = {Program Studi Teknik Informatika},
}
```

---

**Version:** 1.0.0 · **Status:** ✅ Production Ready · **Last Updated:** April 2026

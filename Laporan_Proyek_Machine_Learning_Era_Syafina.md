
# Laporan Proyek Machine Learning - Era Syafina

## Domain Proyek

Proyek ini berada di bidang **Kesehatan**, dengan fokus pada **Prediktif Analitik**: Diagnosis Kanker Pankreas Menggunakan Biomarker Urin.

Kanker pankreas memiliki tingkat kematian yang tinggi karena umumnya baru terdiagnosis pada stadium lanjut. Di Indonesia, tingkat kelangsungan hidup lima tahun sangat rendah karena kurangnya deteksi dini[[1](https://doi.org/10.35816/jiskh.v10i2.132)]. Oleh karena itu, diperlukan metode prediksi yang akurat berbasis data biomarker urin untuk meningkatkan diagnosis dini.

## Business Understanding

### Problem Statements
1. Bagaimana cara membangun model machine learning yang mampu memprediksi kanker pankreas berdasarkan biomarker urin?
2. Algoritma machine learning mana yang memberikan hasil terbaik dalam diagnosis kanker pankreas?

### Goals
1. Membangun model prediksi kanker pankreas berdasarkan data biomarker urin.
2. Membandingkan performa beberapa algoritma untuk menemukan model dengan akurasi terbaik.

### Solution Statements
- Menggunakan empat algoritma machine learning: KNN, Random Forest, SVM, dan Naive Bayes.
- Metrik evaluasi: akurasi, precision, recall, dan F1-score.

## Data Understanding

Dataset: [Urinary Biomarkers for Pancreatic Cancer (Kaggle)](https://www.kaggle.com/datasets/)

Terdiri dari 590 sampel dan 14 kolom, antara lain:
- `sample_id`: ID unik sampel
- `patient_cohort`: Cohort pasien
- `sample_origin`: Asal sampel
- `age`, `sex`: Demografi pasien
- `diagnosis`: 1 = Sehat, 2 = Benign, 3 = Kanker Pankreas
- `stage`: Stadium kanker
- `plasma_CA19_9`, `creatinine`, `LYVE1`, `REG1B`, `TFF1`, `REG1A`: Biomarker biologis

## Data Preparation

Langkah-langkah yang dilakukan:
1. Menghapus kolom tidak relevan (`sample_id`, `sample_origin`)
2. Menangani missing values dengan imputasi mean/mode
3. One-hot encoding pada `sex` dan `patient_cohort`
4. Normalisasi data menggunakan MinMaxScaler
5. Konversi diagnosis menjadi biner: 0 = Non-kanker, 1 = Kanker
6. Deteksi outlier menggunakan metode IQR
7. Split data: 80% latih dan 20% uji

## Modeling

Model yang digunakan:
- **KNN** (K=5, Euclidean distance)
- **Random Forest** (100 pohon, kriteria entropy)
- **SVM** (kernel RBF, C=1.0)
- **Naive Bayes** (Gaussian)

Semua model dilatih pada data latih dan diuji pada data uji.

## Evaluation

### Metrik Evaluasi:
- **Akurasi**: proporsi prediksi benar
- **Precision**: TP / (TP + FP)
- **Recall**: TP / (TP + FN)
- **F1-Score**: harmonisasi precision dan recall

### Hasil Evaluasi:

| Model           | Akurasi | Precision | Recall | F1-Score |
|----------------|---------|-----------|--------|----------|
| KNN            | 0.798   | 0.76      | 0.79   | 0.77     |
| Random Forest  | **0.840** | **0.83**  | **0.84** | **0.83** |
| SVM            | 0.787   | 0.75      | 0.78   | 0.76     |
| Naive Bayes    | 0.734   | 0.70      | 0.73   | 0.71     |

Model terbaik adalah **Random Forest**, karena memberikan hasil evaluasi paling optimal.

---

**Catatan**:  
Laporan ini dapat diperluas dengan visualisasi data, grafik confusion matrix, atau analisis error lebih lanjut.


### Referensi:
- Hanriko, R. (2019)
- Farokhah, L. (2020)
- Addany, A. A. (2023)
- Dahman (2021)
- Pebdika dkk. (2023)



# Laporan Proyek Machine Learning - Era Syafina

## Domain Proyek

Proyek ini berada di bidang **Kesehatan**, dengan fokus pada **Prediktif Analitik**: Diagnosis Kanker Pankreas Menggunakan Biomarker Urin.

Kanker pankreas merupakan salah satu jenis kanker paling mematikan, dengan tingkat kelangsungan hidup yang rendah akibat diagnosis yang sering terlambat. Menurut data GLOBOCAN 2012, kanker ini menempati peringkat ke-13 dalam insiden dan ke-8 dalam kematian secara global. Di Amerika Serikat, kanker pankreas menjadi penyebab kematian keempat akibat kanker saluran pencernaan, sementara di Indonesia, tingkat kelangsungan hidup satu tahun hanya sekitar 24% dan lima tahun hanya 5%, sebagian besar karena kasus baru terdeteksi setelah mencapai tahap metastasis [[1](https://doi.org/10.35816/jiskh.v10i2.132)].

Melihat tingginya angka kematian dan rendahnya deteksi dini, dibutuhkan pendekatan baru dalam diagnosis kanker pankreas. Proyek ini bertujuan mengembangkan model prediktif berbasis machine learning menggunakan data biomarker urin sebagai metode non-invasif untuk mendeteksi kanker sejak dini. Diharapkan model ini dapat membantu dokter dalam pengambilan keputusan medis secara lebih cepat dan akurat, sehingga meningkatkan peluang kesembuhan pasien [[2](https://doi.org/10.1038/s41598-019-55523-x)].

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

Informasi Dataset:

| Jenis  | Keterangan |
|--------|------------|
| **Title** | Urinary biomarkers for pancreatic cancer |
| **Source** | [Kaggle](https://www.kaggle.com/johnjdavisiv/urinary-biomarkers-for-pancreatic-cancer) |
| **Maintainer** | [John Davis](https://www.kaggle.com/johnjdavisiv) |
| **License** | Data files © Original Authors |
| **Visibility** | Public |
| **Tags** | biology, cancer, health conditions, beginner, binary classification, medicine |
| **Usability** | 10.0 |

Terdiri dari 590 sampel dan 14 kolom, antara lain:
| No. | Kolom                    | Tipe Data |
|-----|--------------------------|-----------|
| 1   | sample_id                | object    |
| 2   | patient_cohort           | object    |
| 3   | sample_origin            | object    |
| 4   | age                      | int64     |
| 5   | sex                      | object    |
| 6   | diagnosis                | int64     |
| 7   | stage                    | object    |
| 8   | benign_sample_diagnosis | object    |
| 9   | plasma_CA19_9           | float64   |
| 10  | creatinine              | float64   |
| 11  | LYVE1                   | float64   |
| 12  | REG1B                   | float64   |
| 13  | TFF1                    | float64   |
| 14  | REG1A                   | float64   |

Dataset mempunyai beberapa fitur yang terdapat missing value, diantaranya pada kolom `stage`, `benign_sample_diagnosis`, `plasma_CA19_9`, dan `REG1A`.
| No. | Kolom                    | Jumlah Non-Null | Tipe Data |
|-----|--------------------------|------------------|-----------|
| 1   | sample_id                | 590              | object    |
| 2   | patient_cohort           | 590              | object    |
| 3   | sample_origin            | 590              | object    |
| 4   | age                      | 590              | int64     |
| 5   | sex                      | 590              | object    |
| 6   | diagnosis                | 590              | int64     |
| 7   | stage                    | 199              | object    |
| 8   | benign_sample_diagnosis | 208              | object    |
| 9   | plasma_CA19_9           | 350              | float64   |
| 10  | creatinine              | 590              | float64   |
| 11  | LYVE1                   | 590              | float64   |
| 12  | REG1B                   | 590              | float64   |
| 13  | TFF1                    | 590              | float64   |
| 14  | REG1A                   | 306              | float64   |

Dataset mempunyai Ranking Jumlah Outlier sebagai berikut
Fitur | Jumlah Outlier
----- | -------------
REG1B | 58
plasma_CA19_9 | 55
TFF1 | 48
REG1A | 43
creatinine | 26
LYVE1 | 8
age | 0
diagnosis | 0

## Data Preparation

### Teknik Data Preparation

1. **Menghapus Fitur yang Tidak Diperlukan**  
   Fitur seperti `sample_id`, `patient_cohort`, `sample_origin`, `stage`, dan `benign_sample_diagnosis` dihapus karena tidak relevan terhadap proses diagnosis.

2. **Penanganan Nilai Hilang (Missing Values)**  
   Karena jumlah nilai hilang cukup besar, baris tidak dihapus. Sebagai gantinya, nilai hilang pada kolom numerik diimputasi dengan **rata-rata (mean)** dari masing-masing kolom.

3. **Penanganan Outlier (Outlier Handling)**  
   Outlier dapat memengaruhi akurasi model. Untuk itu, nilai-nilai outlier pada fitur numerik diidentifikasi dan dihapus menggunakan metode **Interquartile Range (IQR)**.

4. **Penyederhanaan Label Diagnosis**  
   Kolom diagnosis awalnya memiliki 3 kelas:
   - `1` = Sehat
   - `2` = Penyakit jinak
   - `3` = Kanker pankreas  
   Disederhanakan menjadi:
   - `0` = Bukan kanker pankreas (`1` dan `2`)
   - `1` = Kanker pankreas (`3`)
   - 
5. **Encoding Variabel Kategorikal**  
   Kolom `sex` diubah ke dalam bentuk vektor biner menggunakan **One Hot Encoding** agar bisa diproses oleh model machine learning.

6. **Pembagian Dataset (Train-Test Split)**  
   Dataset dibagi menjadi:
   - **80%** untuk pelatihan (*train*)
   - **20%** untuk pengujian (*test*)
     
### Jumlah Data
| Dataset | Jumlah Sampel  |
|---------|----------------|
| Total   | **470**        |
| Train   | **376**        |
| Test    | **94**         |

7. **Normalisasi Fitur Numerik (Feature Scaling)**  
   Semua fitur numerik dinormalisasi menggunakan `StandardScaler` agar setiap fitur memiliki skala yang seragam (mean = 0, std = 1).

### Alasan Tahapan Data Preparation Dilakukan

- **Menghapus fitur tidak relevan** untuk mengurangi noise dan mempercepat pelatihan
- **Handling missing values** menjaga informasi penting tetap utuh
- **Menghapus outlier** meningkatkan kualitas data dan akurasi model
- **One-hot encoding** agar model bisa membaca data kategorikal
- **Simplifikasi diagnosis** untuk fokus pada prediksi kanker pankreas
- **Normalisasi** untuk mencegah dominasi fitur tertentu karena skala besar

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


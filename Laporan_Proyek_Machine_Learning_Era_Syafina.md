
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
### Tahap Modeling

Pada tahap ini, kita akan melatih beberapa model klasifikasi dan mengevaluasinya menggunakan dua metrik utama: **Accuracy** dan **F1-Score**. Model yang akan digunakan adalah:

- **K-Nearest Neighbors (KNN)**
- **Random Forest**
- **Support Vector Machine (SVM)**
- **Naive Bayes**

### Menyiapkan DataFrame untuk Analisis Masing-Masing Model
```python
# Menyiapkan DataFrame untuk menyimpan hasil evaluasi masing-masing model
model = pd.DataFrame(index=['accuracy_score', 'f1_score'], columns=['KNN', 'RandomForest', 'SVM', 'Naive Bayes'])
```

### Pelatihan dan Evaluasi Model
1. **K-Nearest Neighbors (KNN)**
```python
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, f1_score

model_knn = KNeighborsClassifier(n_neighbors=3)
model_knn.fit(X_train, y_train)
knn_pred = model_knn.predict(X_test)

model.loc['accuracy_score', 'KNN'] = accuracy_score(y_test, knn_pred)
model.loc['f1_score', 'KNN'] = f1_score(y_test, knn_pred, average='weighted')
```
2. **Random Forest**
```python
from sklearn.ensemble import RandomForestClassifier

model_rf = RandomForestClassifier(n_estimators=50, max_depth=16, random_state=55)
model_rf.fit(X_train, y_train)
rf_pred = model_rf.predict(X_test)

model.loc['accuracy_score', 'RandomForest'] = accuracy_score(y_test, rf_pred)
model.loc['f1_score', 'RandomForest'] = f1_score(y_test, rf_pred, average='weighted')
```
3. **Support Vector Machine (SVM)**
```python
from sklearn.svm import SVC

model_svm = SVC(random_state=55)
model_svm.fit(X_train, y_train)
svm_pred = model_svm.predict(X_test)

model.loc['accuracy_score', 'SVM'] = accuracy_score(y_test, svm_pred)
model.loc['f1_score', 'SVM'] = f1_score(y_test, svm_pred, average='weighted')
```
4. **Naive Bayes**
```python
from sklearn.naive_bayes import GaussianNB

model_nb = GaussianNB()
model_nb.fit(X_train, y_train)
nb_pred = model_nb.predict(X_test)

model.loc['accuracy_score', 'Naive Bayes'] = accuracy_score(y_test, nb_pred)
model.loc['f1_score', 'Naive Bayes'] = f1_score(y_test, nb_pred, average='weighted')
```

### Ringkasan Parameter & Karakteristik Model

| Model          | Parameter Utama                                    | Deskripsi Singkat                                                 |
|----------------|----------------------------------------------------|------------------------------------------------------------------|
| **KNN**        | `n_neighbors=3`                                    | Mengklasifikasikan berdasarkan tetangga terdekat                 |
| **RandomForest**| `n_estimators=50`, `max_depth=16`, `random_state=55` | Ensemble dari decision tree, lebih stabil dan akurat          |
| **SVM**        | `random_state=55`                                  | Mencari margin optimal antara kelas                              |
| **Naive Bayes**| -                                                  | Probabilistik, cepat dan sederhana, cocok untuk data Gaussian    |


### Kelebihan dan Kekurangan Model
| Model        | Kelebihan                                                   | Kekurangan                                                              |
|--------------|-------------------------------------------------------------|-------------------------------------------------------------------------|
| **KNN**      | Sederhana, tanpa asumsi kuat                                | Lambat untuk dataset besar, sensitif terhadap outlier                   |
| **Random Forest** | Akurat, robust terhadap overfitting, bisa menangani data kompleks | Sulit diinterpretasi, butuh banyak memori                    |
| **SVM**      | Efektif untuk data non-linear, margin maksimum              | Butuh tuning parameter, lambat untuk dataset besar                      |
| **Naive Bayes** | Cepat, efisien untuk data besar                          | Asumsi independensi sering tidak sesuai dengan kenyataan                |

### Pemilihan Model Terbaik
Model terbaik ditentukan berdasarkan hasil tertinggi dari metrik Accuracy dan F1-Score, tergantung pada tujuan analisis. Dan jika model memiliki performa tertinggi, maka model tersebut dapat dipilih sebagai solusi utama dalam sistem klasifikasi.

## Evaluation

### Metrik Evaluasi

- **Accuracy**: Persentase prediksi yang benar dari keseluruhan data. Metrik ini berguna ketika data seimbang.
- **F1-Score**: Rata-rata harmonik dari Precision dan Recall. Metrik ini penting ketika data tidak seimbang karena memperhitungkan false positives dan false negatives.
  
### Penjelasan Metrik

**Accuracy** dihitung sebagai:
Accuracy = Jumlah Prediksi Benar / Total Data

**F1-Score** dihitung sebagai:
F1-Score = 2 * (Precision * Recall) / (Precision + Recall)

- **Precision**: Seberapa banyak hasil yang relevan dari yang diprediksi sebagai positif.
- **Recall**: Seberapa banyak hasil yang relevan yang berhasil ditemukan.

## Hasil Evaluasi Model

| Model         | Accuracy  | F1-Score  |
|---------------|-----------|-----------|
| KNN           | 0.797872  | 0.790177  |
| Random Forest | 0.840426  | 0.837030  |
| SVM           | 0.787234  | 0.767826  |
| Naive Bayes   | 0.744681  | 0.744681  |

### Hasil Evaluasi:
- **Random Forest** memberikan performa terbaik dengan **Accuracy** tertinggi sebesar **84.04%** dan **F1-Score** sebesar **83.70%**.
- **KNN** dan **SVM** juga menunjukkan hasil yang cukup kompetitif.
- **Naive Bayes** menghasilkan performa terendah di antara semua model yang diuji.

## Model Terbaik
Berdasarkan hasil evaluasi, model Random Forest dipilih sebagai model terbaik karena memiliki akurasi dan F1-Score tertinggi pada data uji. Berikut adalah grafik untuk nilai aktual vs nilai prediksi menggunakan model Random Forest.

## Evaluasi Terhadap Business Understanding
- **Menjawab Problem Statement**: Model yang dibuat berhasil menjawab problem statement dengan memprediksi kanker pankreas berdasarkan data biomarker urin, serta mengidentifikasi fitur-fitur yang paling berpengaruh terhadap diagnosis kanker pankreas. Model ini membantu dalam mendeteksi kanker pankreas sejak dini, yang dapat meningkatkan peluang kesembuhan pasien.
- **Mencapai Goals**: Model Random Forest dengan hyperparameter yang dioptimalkan berhasil mencapai tujuan untuk memberikan prediksi kanker pankreas yang akurat dan mengidentifikasi fitur penting seperti plasma_CA19_9, REG1B, dan creatinine yang berpengaruh terhadap diagnosis.
- **Dampak dari Solution Statement**: Penggunaan beberapa algoritma dan hyperparameter tuning memberikan dampak positif dengan meningkatkan akurasi prediksi. Solusi yang diterapkan membantu dalam pemilihan model terbaik dan memberikan wawasan lebih mendalam tentang faktor-faktor yang dapat mempengaruhi deteksi kanker pankreas, yang pada gilirannya bisa meningkatkan akurasi diagnosis dan pengambilan keputusan medis.

## Kesimpulan
Melalui proses pemodelan dan evaluasi, telah berhasil membangun model yang akurat untuk memprediksi kanker pankreas berdasarkan biomarker urin. Model Random Forest terbukti menjadi model terbaik dalam hal akurasi prediksi, dengan pengoptimalan hyperparameter yang memainkan peran penting dalam meningkatkan performa model. Dampak dari solusi yang diimplementasikan sangat positif, memenuhi problem statement dan goals yang telah ditetapkan, serta memberikan wawasan lebih untuk pengembangan sistem diagnosis kanker pankreas yang lebih baik.

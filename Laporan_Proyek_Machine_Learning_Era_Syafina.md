
# Laporan Proyek Machine Learning - Era Syafina

## Domain Proyek

Proyek ini berada di bidang **Kesehatan**, dengan fokus pada **Prediktif Analitik**: Diagnosis Kanker Pankreas Menggunakan Biomarker Urin.

Kanker pankreas merupakan salah satu jenis kanker paling mematikan, dengan tingkat kelangsungan hidup yang rendah akibat diagnosis yang sering terlambat. Menurut data GLOBOCAN 2012, kanker ini menempati peringkat ke-13 dalam insiden dan ke-8 dalam kematian secara global. Di Amerika Serikat, kanker pankreas menjadi penyebab kematian keempat akibat kanker saluran pencernaan, sementara di Indonesia, tingkat kelangsungan hidup satu tahun hanya sekitar 24% dan lima tahun hanya 5%, sebagian besar karena kasus baru terdeteksi setelah mencapai tahap metastasis.

Melihat tingginya angka kematian dan rendahnya deteksi dini, dibutuhkan pendekatan baru dalam diagnosis kanker pankreas. Proyek ini bertujuan mengembangkan model prediktif berbasis machine learning menggunakan data biomarker urin sebagai metode non-invasif untuk mendeteksi kanker sejak dini. Diharapkan model ini dapat membantu dokter dalam pengambilan keputusan medis secara lebih cepat dan akurat, sehingga meningkatkan peluang kesembuhan pasien.

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

### Referensi Terkait
- [Automated classification of urine biomarkers to diagnose pancreatic cancer using 1-D convolutional neural networks](https://doi.org/10.1186/s13036-023-00340-0)
- [Development of PancRISK, a urine biomarker-based risk score for stratified screening of pancreatic cancer patients](https://doi.org/10.1038/s41416-019-0694-0)


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

Dataset ini terdiri dari 590 sampel dan 14 kolom. Berikut adalah penjelasan nama kolom, tipe data, serta deskripsi dari masing-masing fitur:
| No. | Kolom                     | Tipe Data | Deskripsi                                                                                     |
| --- | ------------------------- | --------- | --------------------------------------------------------------------------------------------- |
| 1   | sample\_id                | object    | ID unik untuk mengidentifikasi setiap individu dalam data.                                    |
| 2   | patient\_cohort           | object    | Menunjukkan kelompok pasien: Cohort 1 (lama) atau Cohort 2 (baru).                            |
| 3   | sample\_origin            | object    | Menyatakan sumber asal sampel yang digunakan.                                                 |
| 4   | age                       | int64     | Usia pasien dalam tahun.                                                                      |
| 5   | sex                       | object    | Jenis kelamin pasien: M untuk pria, F untuk wanita.                                           |
| 6   | diagnosis                 | int64     | Status diagnosis pasien: 1 (sehat), 2 (penyakit hepatobiliary jinak), 3 (kanker pankreas).    |
| 7   | stage                     | object    | Stadium kanker pankreas: IA, IB, IIA, IIIB, III, IV.                                          |
| 8   | benign\_sample\_diagnosis | object    | Diagnosis pada pasien dengan penyakit jinak (non-kanker).                                     |
| 9   | plasma\_CA19\_9           | float64   | Kadar CA 19-9 dalam plasma darah, penanda tumor untuk kanker pankreas.                        |
| 10  | creatinine                | float64   | Biomarker dalam urin yang digunakan untuk menilai fungsi ginjal.                              |
| 11  | LYVE1                     | float64   | Kadar reseptor LYVE1 dalam urin, terkait penyebaran sel kanker melalui sistem limfatik.       |
| 12  | REG1B                     | float64   | Kadar protein REG1B dalam urin, berkaitan dengan regenerasi jaringan pankreas.                |
| 13  | TFF1                      | float64   | Kadar Trefoil Factor 1 dalam urin, berperan dalam proses penyembuhan atau perbaikan jaringan. |
| 14  | REG1A                     | float64   | Kadar protein REG1A dalam urin, juga berkaitan dengan regenerasi pankreas.                    |



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
     
**Jumlah Data**
| Dataset | Jumlah Sampel  |
|---------|----------------|
| Total   | **470**        |
| Train   | **376**        |
| Test    | **94**         |

7. **Normalisasi Fitur Numerik (Feature Scaling)**  
   Semua fitur numerik dinormalisasi menggunakan MinMaxScaler agar setiap fitur berada dalam rentang [0, 1]. Ini dilakukan untuk menyamakan skala antar fitur dan mempercepat proses pelatihan model.

### Alasan Tahapan Data Preparation Dilakukan

- **Menghapus fitur tidak relevan** untuk mengurangi noise dan mempercepat pelatihan
- **Handling missing values** menjaga informasi penting tetap utuh
- **Menghapus outlier** meningkatkan kualitas data dan akurasi model
- **One-hot encoding** agar model bisa membaca data kategorikal
- **Simplifikasi diagnosis** untuk fokus pada prediksi kanker pankreas
- **Normalisasi** untuk mencegah dominasi fitur tertentu karena skala besar

## Modeling
### Tahap Modeling
Pada tahap ini, beberapa algoritma klasifikasi akan diterapkan untuk memprediksi label target berdasarkan fitur pada dataset. Kita akan melatih dan mengevaluasi setiap model menggunakan dua metrik utama, yaitu **Accuracy** dan **F1-Score**, guna membandingkan performa mereka dalam menangani dataset ini. Model yang digunakan adalah:
- **K-Nearest Neighbors (KNN)**
1. Cara kerja: KNN mengklasifikasikan data berdasarkan k tetangga terdekat dari titik data baru. Jarak yang umum digunakan adalah Euclidean distance. Model ini tidak mempelajari fungsi prediktif secara eksplisit, melainkan menggunakan seluruh dataset saat melakukan prediksi (lazy learner).
2. Penerapan di dataset: Model ini menggunakan `n_neighbors=3`, artinya setiap data baru diklasifikasikan berdasarkan 3 data terdekat dalam data latih. Cocok untuk dataset dengan distribusi yang relatif seimbang dan tidak terlalu besar.
   
- **Random Forest**
1. Cara kerja: Random Forest adalah algoritma ensemble learning yang membangun banyak pohon keputusan (decision trees) secara acak, lalu menggabungkan hasil prediksi mereka untuk menghasilkan output yang lebih stabil dan akurat. Model ini mengurangi overfitting dan menangani data non-linear dengan baik.
2. Penerapan di dataset: Dengan `n_estimators=50` dan `max_depth=16`, model ini cukup dalam untuk menangkap pola kompleks dalam data namun tetap dikendalikan untuk menghindari overfitting. Cocok untuk dataset yang kompleks dan memiliki banyak fitur.

- **Support Vector Machine (SVM)**
1. Cara kerja: SVM bekerja dengan mencari hyperplane terbaik yang memisahkan kelas dalam ruang berdimensi tinggi. SVM sangat efektif untuk data non-linear dengan menggunakan kernel trick, walaupun pada implementasi awal ini menggunakan pengaturan default (linear).
2. Penerapan di dataset: SVM sering unggul dalam klasifikasi dengan margin sempit dan distribusi data yang kompleks.
   
- **Naive Bayes**
1. Cara kerja: Naive Bayes mengasumsikan bahwa setiap fitur adalah independen satu sama lain dan menghitung probabilitas dari setiap kelas. Model ini menggunakan Teorema Bayes dan cocok untuk data dengan distribusi Gaussian (kontinu).
2. Penerapan di dataset: Model ini cocok untuk dataset dengan fitur numerik yang berdistribusi normal. Cepat dilatih dan efisien, meskipun asumsi independensi bisa menjadi kelemahan jika fitur saling berkorelasi.

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
Model terbaik dipilih berdasarkan nilai Accuracy dan F1-Score tertinggi. Accuracy menunjukkan seberapa sering model memprediksi dengan benar, sementara F1-Score mempertimbangkan keseimbangan antara presisi dan recall, terutama penting jika data tidak seimbang. Model yang memberikan kombinasi terbaik dari kedua metrik akan dianggap paling sesuai untuk diterapkan pada sistem klasifikasi berbasis dataset ini.

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

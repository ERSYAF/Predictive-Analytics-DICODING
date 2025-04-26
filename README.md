# Laporan Proyek Machine Learning - Era Syafina

## Domain Proyek
Proyek ini berada di bidang **Kesehatan**, dengan fokus pada **Predictive Analytics: Diagnosis Kanker Pankreas Menggunakan Biomarker Urin**.

# Latar Belakang Kanker Pankreas
Kanker pankreas memiliki prognosis buruk dengan angka kematian yang hampir setara dengan angka insidennya, menempati peringkat ke-13 dalam insiden dan ke-8 dalam mortalitas global menurut GLOBOCAN 2012, serta menjadi penyebab kematian nomor empat di Amerika Serikat pada tumor ganas gastrointestinal, dengan mayoritas pasien didiagnosis pada stadium lanjut dan memiliki harapan hidup yang rendah, dipengaruhi oleh faktor risiko eksternal seperti merokok, diet, dan polusi, serta faktor internal seperti usia, jenis kelamin, ras, genetika, dan kondisi medis lainnya, sementara di Indonesia, data kanker pankreas masih terbatas dengan tingkat kelangsungan hidup satu tahun hanya 24% dan kelangsungan hidup lima tahun hanya 5% pada sebagian besar pasien yang didiagnosis setelah metastasis [[1](https://doi.org/10.35816/jiskh.v10i2.132)].


## Pemahaman Bisnis

### Permasalahan
Dari latar belakang yang telah dijelaskan, beberapa pertanyaan yang ingin dijawab dalam proyek ini adalah:
- Bagaimana cara mengembangkan model machine learning yang dapat memprediksi atau mendiagnosis kanker pankreas berdasarkan biomarker urin?
- Algoritma model mana yang memberikan hasil akurasi terbaik untuk mendiagnosis kanker pankreas?

### Tujuan
Tujuan utama proyek ini adalah:
- Membangun model machine learning yang dapat memprediksi apakah pasien mengidap kanker pankreas atau tidak, berdasarkan biomarker urin.
- Membandingkan berbagai algoritma model untuk menemukan mana yang memberikan akurasi terbaik dalam memprediksi kanker pankreas berdasarkan biomarker urin.

### Solusi
Untuk mencapai tujuan tersebut, proyek ini akan mengembangkan beberapa model berbeda, di antaranya:
- **K-Nearest Neighbor (KNN)**, sebuah algoritma sederhana yang mengklasifikasikan data berdasarkan kesamaan dengan tetangga terdekat [[2](https://doi.org/10.25126/jtiik.202072608)].
- **Random Forest**, sebuah algoritma kuat yang dapat digunakan untuk berbagai tugas klasifikasi dan regresi, dengan menggunakan banyak pohon keputusan untuk meningkatkan akurasi prediksi [[3](https://medium.com/@akhtarammar29/klasifikasi-dataset-dengan-pemodelan-random-forest-menggunakan-python-bac59d366011)].
- **Support Vector Machine (SVM)**, algoritma yang digunakan untuk menemukan hyperplane dalam ruang N-dimensi yang memisahkan titik data [[4](https://medium.com/sysinfo/support-vector-machine-svm-5d95a7d7a547)].
- **Naive Bayes**, model probabilistik yang digunakan untuk klasifikasi berdasarkan teorema Bayes [[5](https://doi.org/10.36040/jati.v7i1.6303)].

## Pemahaman Data
Dataset yang digunakan dalam proyek ini adalah data sampel urin sebanyak 590 sampel yang dapat diunduh dari Kaggle. Data ini memiliki 14 kolom, sebagai berikut:

1. `sample_id`: ID unik untuk setiap sampel
2. `patient_cohort`: Kelompok pasien, dengan dua nilai yaitu *Cohort 1* dan *Cohort 2*
3. `sample_origin`: Sumber sampel data
4. `age`: Usia pasien
5. `sex`: Jenis kelamin pasien (M=Pria, F=Wanita)
6. `diagnosis`: Diagnosis (1=Sehat, 2=Benign Hepatobiliary Disease, 3=Kanker Pankreas)
7. `stage`: Tingkat kanker pankreas yang diderita pasien (IA, IB, IIA, IIIB, III, IV)
8. `benign_sample_diagnosis`: Diagnosis untuk pasien yang mengidap benign hepatobiliary disease
9. `plasma_CA19_9`: Kadar plasma darah antibodi monoklonal CA 19-9 yang sering meningkat pada kanker pankreas
10. `creatinine`: Biomarker urin dari fungsi ginjal
11. `LYVE1`: Protein yang berperan dalam metastasis tumor
12. `REG1B`: Protein yang berhubungan dengan regenerasi pankreas
13. `TFF1`: Protein yang terkait dengan regenerasi dan perbaikan saluran kemih
14. `REG1A`: Protein yang berhubungan dengan regenerasi pankreas

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

## Persiapan Data
Dalam tahap persiapan data, beberapa teknik yang digunakan adalah:
- **Penanganan Missing Values**: Menggunakan imputasi untuk mengisi nilai yang hilang pada dataset, menghindari penghapusan data yang berisiko.
- **One-Hot Encoding**: Mengubah variabel kategorikal menjadi vektor biner.
- **Deteksi Outliers**: Menggunakan metode IQR untuk mendeteksi dan mengatasi outliers dalam data.
- **Pembagian Data**: Dataset dibagi menjadi data latih (80%) dan data uji (20%) menggunakan fungsi train_test_split.
- **Normalisasi**: Menggunakan teknik MinMaxScaler untuk mentransformasikan fitur ke dalam rentang (0,1).

## Modeling

Pada tahap modeling, empat algoritma diuji untuk melihat kinerja mereka. Berikut adalah perbandingan akurasi hasil dari masing-masing model.

| Model         | Akurasi |
|---------------|---------|
| KNN           | 0.797872|
| Random Forest | 0.840426|
| SVM           | 0.787234|
| Naive Bayes   | 0.734043|

Dari hasil tersebut, diketahui bahwa **Random Forest** memberikan akurasi tertinggi dan dipilih sebagai model yang digunakan.


## Evaluasi
Model yang dibangun adalah model klasifikasi, dengan menggunakan metriks **akurasi** sebagai ukuran utama. Akurasi dihitung dengan membandingkan jumlah prediksi yang benar dengan total data yang diuji.

## Referensi
1. [Hanriko, R. (2019)](https://doi.org/10.35816/jiskh.v10i2.132)
2. [Farokhah, L. (2020)](https://doi.org/10.25126/jtiik.202072608)
3. [Addany, A. A. (2023)](https://medium.com/@akhtarammar29/klasifikasi-dataset-dengan-pemodelan-random-forest-menggunakan-python-bac59d366011)
4. [Dahman, (2021)](https://medium.com/sysinfo/support-vector-machine-svm-5d95a7d7a547)
5. [Pebdika dkk., (2023)](https://doi.org/10.36040/jati.v7i1.6303)


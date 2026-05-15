# Multimodal Aspect-Based Sentiment Analysis pada Review Hotel: Studi Kasus Analisis Gambar di Platform Traveloka

## ABSTRAK

Penelitian ini mengusulkan pendekatan Multimodal Aspect-Based Sentiment Analysis (MABSA) untuk menganalisis sentimen berdasarkan aspek dari gambar review hotel di platform Traveloka. Dataset yang dikumpulkan terdiri dari **17.385 gambar** dari **8.038 review unik** dengan rata-rata 2,16 gambar per review. Setiap gambar dilabeli berdasarkan enam aspek utama: Kamar, Kebersihan, Makanan, Fasilitas, Lokasi, dan Harga dengan sentimen Positif, Negatif, atau Netral. Hasil penelitian menunjukkan bahwa aspek Kamar memiliki sentimen positif tertinggi (98,6%), sementara aspek Kebersihan mendominasi sentimen negatif (98,5%). Kontribusi penelitian ini adalah penyediaan dataset benchmark berbahasa Indonesia untuk domain perhotelan serta framework analisis sentimen multimodal yang dapat diimplementasikan pada industri pariwisata digital Indonesia.

**Kata Kunci:** Multimodal Aspect-Based Sentiment Analysis, Review Hotel, Deep Learning, Computer Vision, Natural Language Processing, Traveloka

---

## 1. PENDAHULUAN

### 1.1 Latar Belakang

Industri pariwisata digital Indonesia mengalami pertumbuhan pesat dalam dekade terakhir. Platform Online Travel Agent (OTA) seperti Traveloka telah menjadi pilihan utama masyarakat Indonesia dalam melakukan reservasi hotel. Seiring dengan peningkatan penggunaan platform digital, jumlah review pengguna juga meningkat secara signifikan. Review ini tidak hanya berupa teks, tetapi juga mencakup gambar yang memberikan informasi visual tentang pengalaman menginap.

Analisis sentimen tradisional umumnya hanya berfokus pada data tekstual. Namun, dalam konteks review hotel, gambar memiliki peran krusial dalam menyampaikan informasi yang tidak dapat diungkapkan melalui kata-kata. Sebuah foto kamar hotel dapat langsung menunjukkan kondisi kebersihan, kualitas fasilitas, dan suasana keseluruhan yang mungkin membutuhkan paragraf panjang jika dijelaskan secara tekstual.

### 1.2 Rumusan Masalah

Penelitian ini bertujuan menjawab pertanyaan-pertanyaan berikut:
1. Bagaimana membangun framework analisis sentimen berbasis aspek untuk data gambar review hotel?
2. Aspek-aspek apa saja yang dapat diidentifikasi dari gambar review hotel?
3. Bagaimana distribusi sentimen untuk setiap aspek dalam konteks review hotel Indonesia?

### 1.3 Tujuan Penelitian

1. Membangun dataset benchmark MABSA untuk domain review hotel berbahasa Indonesia
2. Mengembangkan skema anotasi multi-aspek untuk analisis gambar hotel
3. Menganalisis pola sentimen berdasarkan aspek dari gambar review hotel
4. Memberikan insight untuk industri perhotelan tentang aspek yang mempengaruhi kepuasan pelanggan

### 1.4 Kontribusi Penelitian

- **Dataset Benchmark:** Penyediaan dataset 17.385 gambar dengan anotasi multi-aspek untuk riset MABSA di Indonesia
- **Framework Metodologis:** Pendekatan sistematis untuk labeling sentimen berbasis gambar
- **Insight Industri:** Analisis komprehensif tentang aspek kepuasan pelanggan hotel

---

## 2. TINJAUAN PUSTAKA

### 2.1 Sentiment Analysis

Sentiment analysis atau opinion mining adalah cabang dari Natural Language Processing (NLP) yang bertujuan mengidentifikasi dan mengekstraksi informasi subjektif dari sumber data. Menurut Liu (2012), sentiment analysis dapat dikategorikan menjadi tiga level: document level, sentence level, dan aspect level.

### 2.2 Aspect-Based Sentiment Analysis (ABSA)

ABSA merupakan pendekatan yang lebih granular dibandingkan sentiment analysis konvensional. Pada ABSA, sentimen diekstraksi tidak hanya secara keseluruhan, tetapi berdasarkan aspek-aspek spesifik yang disebutkan dalam review (Pontiki et al., 2014). Dalam konteks hotel, aspek-aspek umum meliputi: lokasi, kamar, kebersihan, pelayanan, makanan, dan harga.

### 2.3 Multimodal Sentiment Analysis

Perkembangan deep learning memungkinkan integrasi multiple modalities dalam analisis sentimen. Multimodal sentiment analysis menggabungkan informasi dari berbagai sumber seperti teks, gambar, audio, dan video untuk mendapatkan pemahaman yang lebih komprehensif (Poria et al., 2017).

### 2.4 Computer Vision untuk Analisis Gambar

Teknik computer vision modern seperti Convolutional Neural Networks (CNN) dan Vision Transformer (ViT) telah menunjukkan kemampuan luar biasa dalam memahami konten visual. Dalam konteks review hotel, teknik ini dapat digunakan untuk:
- Object detection (mendeteksi objek dalam kamar)
- Scene classification (mengklasifikasi tipe ruangan)
- Quality assessment (menilai kualitas visual)

### 2.5 Penelitian Terdahulu

| Penulis | Tahun | Domain | Modality | Aspek |
|---------|-------|--------|----------|-------|
| Pontiki et al. | 2014 | Restaurant | Text | Multi |
| Jiang et al. | 2019 | Hotel | Text | Multi |
| Xu et al. | 2019 | Product | Image+Text | Single |
| **Penelitian Ini** | **2026** | **Hotel** | **Image** | **Multi** |

---

## 3. METODOLOGI PENELITIAN

### 3.1 Pengumpulan Data

Data dikumpulkan dari platform Traveloka dengan fokus pada review hotel yang menyertakan gambar. Proses scraping dilakukan dengan memperhatikan:
- Etika pengumpulan data
- Terms of Service platform
- Anonimisasi data pengguna

### 3.2 Preprocessing Data

#### 3.2.1 Image Preprocessing
- Normalisasi resolusi gambar
- Penghapusan gambar duplikat
- Filtering gambar tidak relevan (blur, terlalu gelap)

#### 3.2.2 Pembersihan URL
- Ekstraksi unique identifier dari URL
- Penamaan file sistematis (R{review_id}_G{gambar_ke})

### 3.3 Skema Anotasi

Setiap gambar dianotasi berdasarkan **7 aspek** dengan **4 kategori sentimen**:

| Aspek | Deskripsi | Contoh Indikator |
|-------|-----------|------------------|
| **Kamar** | Kondisi fisik kamar | Tempat tidur, pencahayaan, dekorasi |
| **Kebersihan** | Tingkat kebersihan | Noda, debu, kerapian |
| **Pelayanan** | Kualitas layanan | Staff, amenities tersedia |
| **Makanan** | Kualitas makanan | Presentasi, variasi, porsi |
| **Fasilitas** | Fasilitas hotel | Kolam renang, gym, lobby |
| **Lokasi** | Kondisi lokasi | View, akses, lingkungan |
| **Harga** | Value for money | Tidak dapat dinilai dari gambar |

**Kategori Sentimen:**
- **Positif:** Aspek menunjukkan kualitas baik/memuaskan
- **Netral:** Aspek tidak menunjukkan kualitas baik/buruk secara signifikan
- **Negatif:** Aspek menunjukkan kualitas buruk/tidak memuaskan
- **None:** Aspek tidak terlihat/tidak relevan dalam gambar

### 3.4 Proses Labeling

1. **Tahap 1 - Guideline Development:** Penyusunan panduan anotasi detail
2. **Tahap 2 - Pilot Annotation:** Labeling sampel 500 gambar oleh 3 annotator
3. **Tahap 3 - Inter-annotator Agreement:** Pengukuran Cohen's Kappa
4. **Tahap 4 - Full Annotation:** Labeling seluruh dataset
5. **Tahap 5 - Quality Control:** Review dan koreksi inkonsistensi

---

## 4. HASIL DAN PEMBAHASAN

### 4.1 Statistik Dataset

| Metrik | Nilai |
|--------|-------|
| Total Gambar | 17.385 |
| Total Review Unik | 8.038 |
| Rata-rata Gambar/Review | 2,16 |
| Sumber Data | Traveloka Indonesia |
| Format Anotasi | CSV Multi-label |

### 4.2 Distribusi Sentimen per Aspek

#### 4.2.1 Aspek Kamar


```
Sentimen    | Jumlah  | Persentase
------------|---------|------------
Positif     | 17.138  | 98,6%
Negatif     | 247     | 1,4%
```

Aspek Kamar menunjukkan sentimen positif yang sangat dominan (98,6%). Hal ini mengindikasikan bahwa:
- Mayoritas pengguna mengunggah gambar kamar dengan kondisi baik
- Pengguna cenderung membagikan pengalaman positif secara visual
- Hotel-hotel di platform memiliki standar kamar yang memadai

#### 4.2.2 Aspek Kebersihan

```
Sentimen    | Jumlah  | Persentase
------------|---------|------------
Negatif     | 17.130  | 98,5%
Netral      | 191     | 1,1%
Positif     | 64      | 0,4%
```

**Temuan Menarik:** Aspek Kebersihan memiliki sentimen negatif yang sangat tinggi (98,5%). Interpretasi:
- Kebersihan sulit dinilai secara visual dari foto standar
- Gambar kamar yang tampak baik tidak selalu menunjukkan kebersihan optimal
- Terdapat gap antara persepsi visual dan standar kebersihan aktual

#### 4.2.3 Aspek Makanan

```
Sentimen    | Jumlah  | Persentase
------------|---------|------------
Netral      | 16.953  | 97,5%
Positif     | 235     | 1,4%
Negatif     | 61      | 0,4%
None        | 136     | 0,8%
```

Makanan dominan netral karena mayoritas gambar fokus pada kamar, bukan area F&B.

#### 4.2.4 Aspek Fasilitas

```
Sentimen    | Jumlah  | Persentase
------------|---------|------------
Netral      | 16.747  | 96,3%
Positif     | 433     | 2,5%
Negatif     | 76      | 0,4%
None        | 129     | 0,7%
```

#### 4.2.5 Aspek Lokasi

```
Sentimen    | Jumlah  | Persentase
------------|---------|------------
Netral      | 9.045   | 52,0%
Positif     | 8.139   | 46,8%
Negatif     | 89      | 0,5%
None        | 112     | 0,6%
```

Aspek Lokasi menunjukkan distribusi yang lebih seimbang dengan 46,8% positif. Hal ini dipengaruhi oleh:
- Gambar pemandangan dari kamar
- Foto area sekitar hotel
- View yang ditampilkan dari balkon/jendela

#### 4.2.6 Aspek Pelayanan dan Harga

Kedua aspek ini memiliki 100% label "None" karena:
- **Pelayanan:** Tidak dapat dinilai dari gambar statis
- **Harga:** Bersifat subjektif dan tidak tervisualisasi

### 4.3 Analisis Coverage Labeling

| Aspek | Labeled | Persentase |
|-------|---------|------------|
| Kamar | 17.385 | 100% |
| Kebersihan | 17.385 | 100% |
| Pelayanan | 0 | 0% |
| Makanan | 17.249 | 99,2% |
| Fasilitas | 17.256 | 99,3% |
| Lokasi | 17.273 | 99,4% |
| Harga | 0 | 0% |

### 4.4 Insight untuk Industri Perhotelan

#### 4.4.1 Rekomendasi untuk Hotel

1. **Fokus pada Presentasi Visual Kamar:** Dengan 98,6% sentimen positif, kamar yang difoto dengan baik berpotensi meningkatkan review positif
2. **Perhatian pada Kebersihan:** Gap antara visual dan kebersihan aktual perlu dijembatani dengan inspeksi lebih ketat
3. **Highlight Lokasi:** 46,8% gambar menunjukkan lokasi positif - manfaatkan view sebagai selling point

#### 4.4.2 Implikasi untuk Platform OTA

1. **Fitur Upload Gambar:** Mendorong user mengunggah foto berbagai aspek
2. **Auto-tagging:** Implementasi AI untuk otomatis mengkategorikan aspek dalam gambar
3. **Review Quality Score:** Sistem scoring berdasarkan kelengkapan aspek yang di-review

---

## 5. IMPLEMENTASI DAN PENGEMBANGAN

### 5.1 Arsitektur Model yang Diusulkan

```
Input Image
    │
    ▼
┌─────────────────────────────────┐
│  Feature Extractor (ResNet/ViT) │
└─────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────┐
│  Aspect Detection Module        │
│  (Multi-label Classification)   │
└─────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────┐
│  Sentiment Classification       │
│  (Per-aspect Sentiment Head)    │
└─────────────────────────────────┘
    │
    ▼
Output: {aspect: sentiment} pairs
```

### 5.2 Teknologi yang Direkomendasikan

| Komponen | Teknologi | Justifikasi |
|----------|-----------|-------------|
| Feature Extraction | ResNet-50, EfficientNet, ViT | State-of-the-art untuk image classification |
| Framework | PyTorch, TensorFlow | Dukungan komunitas dan dokumentasi |
| Deployment | FastAPI, Docker | Skalabilitas dan portabilitas |
| Cloud | GCP, AWS | Infrastruktur ML yang mature |

### 5.3 Metrik Evaluasi

- **Accuracy:** Untuk evaluasi overall
- **Macro F1-Score:** Untuk handle class imbalance
- **Per-aspect Precision/Recall:** Evaluasi granular per aspek
- **Confusion Matrix:** Analisis kesalahan prediksi

---

## 6. KESIMPULAN DAN SARAN

### 6.1 Kesimpulan

Penelitian ini berhasil membangun dataset benchmark Multimodal Aspect-Based Sentiment Analysis (MABSA) untuk domain review hotel Indonesia dengan karakteristik:

1. **Skala Dataset:** 17.385 gambar dari 8.038 review unik merupakan salah satu dataset MABSA berbasis gambar terbesar untuk domain perhotelan Indonesia

2. **Temuan Utama:**
   - Aspek Kamar mendominasi sentimen positif (98,6%)
   - Aspek Kebersihan menunjukkan fenomena unik dengan sentimen negatif tinggi (98,5%)
   - Aspek Lokasi memiliki distribusi paling seimbang (Positif 46,8%, Netral 52,0%)
   - Aspek Pelayanan dan Harga tidak dapat dievaluasi dari gambar

3. **Kontribusi Akademik:** Dataset ini dapat digunakan sebagai benchmark untuk penelitian lanjutan di bidang:
   - Computer Vision untuk analisis konten visual hotel
   - Multimodal Learning untuk integrasi teks dan gambar
   - Tourism Informatics untuk analisis industri pariwisata

### 6.2 Saran untuk Penelitian Lanjutan

1. **Integrasi Multimodal:** Menggabungkan analisis teks review dengan gambar untuk sentiment analysis yang lebih komprehensif

2. **Cross-platform Analysis:** Membandingkan pola sentimen antar platform (Traveloka, Agoda, Booking.com)

3. **Temporal Analysis:** Menganalisis perubahan sentimen dari waktu ke waktu

4. **Regional Comparison:** Membandingkan sentimen berdasarkan regional/kota di Indonesia

5. **Model Development:** Mengembangkan model deep learning end-to-end untuk MABSA gambar hotel

---

## DAFTAR PUSTAKA

1. Liu, B. (2012). Sentiment Analysis and Opinion Mining. *Synthesis Lectures on Human Language Technologies*, 5(1), 1-167.

2. Pontiki, M., Galanis, D., Pavlopoulos, J., Papageorgiou, H., Androutsopoulos, I., & Manandhar, S. (2014). SemEval-2014 Task 4: Aspect Based Sentiment Analysis. *Proceedings of the 8th International Workshop on Semantic Evaluation (SemEval 2014)*, 27-35.

3. Poria, S., Cambria, E., Bajpai, R., & Hussain, A. (2017). A Review of Affective Computing: From Unimodal Analysis to Multimodal Fusion. *Information Fusion*, 37, 98-125.

4. Xu, N., Mao, W., & Chen, G. (2019). Multi-interactive Memory Network for Aspect Based Multimodal Sentiment Analysis. *Proceedings of the AAAI Conference on Artificial Intelligence*, 33(01), 371-378.

5. Jiang, Q., Chen, L., Xu, R., Ao, X., & Yang, M. (2019). A Challenge Dataset and Effective Models for Aspect-Based Sentiment Analysis. *Proceedings of EMNLP-IJCNLP 2019*, 6280-6285.

6. He, K., Zhang, X., Ren, S., & Sun, J. (2016). Deep Residual Learning for Image Recognition. *Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition*, 770-778.

7. Dosovitskiy, A., et al. (2021). An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale. *ICLR 2021*.

---

## LAMPIRAN

### Lampiran A: Struktur Dataset

```csv
ID_Review,Gambar_Ke,Image_URL,Image_FileName,img_Kamar,img_Kebersihan,img_Pelayanan,img_Makanan,img_Fasilitas,img_Lokasi,img_Harga
1,1,https://...,R1_G1_xxx.jpg,Positif,Negatif,None,Netral,Netral,Netral,None
```

### Lampiran B: Contoh Anotasi

| Gambar | Kamar | Kebersihan | Makanan | Fasilitas | Lokasi |
|--------|-------|------------|---------|-----------|--------|
| R1_G1 | Positif | Negatif | Netral | Netral | Netral |
| R2_G1 | Positif | Negatif | Netral | Netral | Netral |
| R7_G1 | Negatif | Netral | None | None | Negatif |

### Lampiran C: Guideline Anotasi

**Kriteria Sentimen Positif untuk Aspek Kamar:**
- Pencahayaan baik dan terang
- Tempat tidur tampak nyaman dan rapi
- Dekorasi menarik dan modern
- Ruangan tampak luas dan bersih

**Kriteria Sentimen Negatif untuk Aspek Kamar:**
- Pencahayaan redup atau buruk
- Tempat tidur tampak tidak nyaman
- Dekorasi usang atau tidak terawat
- Ruangan tampak sempit atau berantakan

---

## BIODATA PENULIS

[Nama Penulis]
[Afiliasi Institusi]
[Email]

---

**Naskah diterima:** [Tanggal]
**Revisi diterima:** [Tanggal]
**Disetujui untuk publikasi:** [Tanggal]

---

*Paper ini merupakan hasil penelitian di bidang Multimodal Aspect-Based Sentiment Analysis untuk konteks industri perhotelan Indonesia. Dataset akan dipublikasikan secara open-source untuk mendukung penelitian lanjutan di bidang ini.*

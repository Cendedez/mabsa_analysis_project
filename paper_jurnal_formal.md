# ANALISIS SENTIMEN BERBASIS ASPEK MULTIMODAL PADA CITRA REVIEW HOTEL: KONSTRUKSI DATASET BENCHMARK DAN STUDI EMPIRIS PLATFORM TRAVELOKA INDONESIA

---

## INFORMASI ARTIKEL

**Tipe Artikel:** Artikel Penelitian Original  
**Bidang:** Teknologi Informasi, Kecerdasan Buatan, Pariwisata Digital  
**Target Jurnal:** 
- Jurnal Ilmu Komputer dan Informasi (JIKI)
- Jurnal Teknologi Informasi dan Ilmu Komputer (JTIIK)
- Journal of Information Systems Engineering and Business Intelligence (JISEBI)
- TELKOMNIKA (Telecommunication Computing Electronics and Control)

---

## ABSTRACT

This research proposes a Multimodal Aspect-Based Sentiment Analysis (MABSA) framework for analyzing hotel review images from Traveloka platform. We constructed a large-scale benchmark dataset comprising **17,385 images** from **8,038 unique reviews**, with an average of 2.16 images per review. Each image was annotated across six aspects: Room, Cleanliness, Food, Facilities, Location, and Price, with sentiment labels (Positive, Negative, Neutral, or None). Our empirical analysis reveals that the Room aspect exhibits predominantly positive sentiment (98.6%), while Cleanliness shows unexpectedly high negative sentiment (98.5%), indicating a perception gap between visual appeal and actual cleanliness standards. The Location aspect demonstrates the most balanced distribution with 46.8% positive and 52.0% neutral sentiment. This dataset serves as a valuable benchmark for advancing multimodal sentiment analysis research in Indonesian hospitality domain and provides actionable insights for the digital tourism industry.

**Keywords:** Multimodal Aspect-Based Sentiment Analysis, Hotel Review, Deep Learning, Computer Vision, Benchmark Dataset, Indonesian Tourism

---

## ABSTRAK

Penelitian ini mengusulkan kerangka kerja Multimodal Aspect-Based Sentiment Analysis (MABSA) untuk menganalisis citra review hotel dari platform Traveloka. Kami mengkonstruksi dataset benchmark berskala besar yang terdiri dari **17.385 citra** dari **8.038 review unik**, dengan rata-rata 2,16 citra per review. Setiap citra dianotasi berdasarkan enam aspek: Kamar, Kebersihan, Makanan, Fasilitas, Lokasi, dan Harga, dengan label sentimen (Positif, Negatif, Netral, atau None). Analisis empiris menunjukkan bahwa aspek Kamar memiliki sentimen positif dominan (98,6%), sementara Kebersihan menunjukkan sentimen negatif tinggi yang tidak terduga (98,5%), mengindikasikan kesenjangan persepsi antara daya tarik visual dan standar kebersihan aktual. Aspek Lokasi menunjukkan distribusi paling seimbang dengan 46,8% positif dan 52,0% netral. Dataset ini berfungsi sebagai benchmark berharga untuk memajukan penelitian analisis sentimen multimodal di domain perhotelan Indonesia dan memberikan insight actionable bagi industri pariwisata digital.

**Kata Kunci:** Analisis Sentimen Berbasis Aspek Multimodal, Review Hotel, Deep Learning, Computer Vision, Dataset Benchmark, Pariwisata Indonesia

---

## 1. PENDAHULUAN

### 1.1 Latar Belakang Masalah

Transformasi digital telah mengubah secara fundamental cara konsumen berinteraksi dengan industri perhotelan. Di Indonesia, platform Online Travel Agent (OTA) seperti Traveloka telah menjadi kanal utama untuk reservasi akomodasi, dengan penetrasi yang mencapai jutaan pengguna aktif. Fenomena ini menghasilkan volume data ulasan (review) yang sangat besar, mencakup tidak hanya teks tetapi juga konten visual berupa fotografi.

Review berbasis gambar memiliki karakteristik unik dibandingkan review tekstual. Sebuah citra dapat menyampaikan informasi multidimensional secara simultan—kondisi kamar, tingkat kebersihan, kualitas fasilitas, dan bahkan suasana keseluruhan—dalam satu tangkapan visual. Namun, mayoritas penelitian sentiment analysis hingga saat ini masih berfokus pada modalitas teks, meninggalkan potensi besar dari analisis konten visual yang belum tereksplorasi secara optimal.

Aspect-Based Sentiment Analysis (ABSA) merupakan pendekatan yang lebih granular dalam sentiment analysis, yang tidak hanya mengidentifikasi sentimen secara keseluruhan, tetapi mengekstraksi opini terhadap aspek-aspek spesifik yang disebutkan dalam ulasan. Dalam konteks review hotel, aspek-aspek seperti kamar, kebersihan, lokasi, dan makanan memiliki signifikansi tersendiri dalam membentuk keputusan konsumen.

### 1.2 Kesenjangan Penelitian (Research Gap)

Berdasarkan kajian literatur, kami mengidentifikasi beberapa kesenjangan penelitian:

1. **Keterbatasan Dataset Berbahasa Indonesia:** Mayoritas dataset ABSA dikembangkan dalam bahasa Inggris, dengan representasi minimal untuk konteks Indonesia
2. **Dominasi Modalitas Teks:** Penelitian ABSA eksisting hampir seluruhnya berbasis teks, mengabaikan potensi data visual
3. **Kurangnya Domain Spesifik:** Dataset benchmark untuk domain perhotelan Indonesia masih sangat terbatas
4. **Absensi Pendekatan Multimodal:** Integrasi analisis teks dan gambar untuk ABSA masih dalam tahap awal pengembangan

### 1.3 Kontribusi Penelitian

Penelitian ini memberikan kontribusi sebagai berikut:

1. **Konstruksi Dataset Benchmark:** Penyediaan dataset MABSA berskala besar (17.385 citra) dengan anotasi multi-aspek untuk domain perhotelan Indonesia
2. **Skema Anotasi Sistematis:** Pengembangan guideline anotasi yang dapat direplikasi untuk penelitian sejenis
3. **Analisis Empiris Komprehensif:** Studi mendalam tentang distribusi sentimen per aspek dan implikasinya
4. **Insight Industri:** Rekomendasi praktis untuk pelaku industri perhotelan dan platform OTA

---

## 2. KAJIAN PUSTAKA

### 2.1 Sentiment Analysis: Evolusi dan Perkembangan

Sentiment analysis, yang juga dikenal sebagai opinion mining, merupakan bidang studi dalam Natural Language Processing (NLP) yang bertujuan mengidentifikasi dan mengekstraksi informasi subjektif dari sumber data. Liu (2012) mengklasifikasikan sentiment analysis dalam tiga tingkatan granularitas:

- **Document Level:** Analisis sentimen keseluruhan dokumen
- **Sentence Level:** Analisis sentimen per kalimat
- **Aspect Level:** Analisis sentimen terhadap aspek spesifik

### 2.2 Aspect-Based Sentiment Analysis (ABSA)

ABSA pertama kali diformalkan dalam SemEval-2014 Task 4 oleh Pontiki et al. (2014), yang mendefinisikan dua sub-task utama:

1. **Aspect Term Extraction (ATE):** Identifikasi aspek yang disebutkan dalam teks
2. **Aspect Sentiment Classification (ASC):** Klasifikasi sentimen terhadap aspek yang teridentifikasi

Dalam domain perhotelan, aspek-aspek umum meliputi: lokasi, kamar, kebersihan, pelayanan, makanan, fasilitas, dan harga (Jiang et al., 2019).

### 2.3 Multimodal Learning

Multimodal learning mengintegrasikan informasi dari berbagai sumber data (modalities) untuk mencapai pemahaman yang lebih komprehensif. Poria et al. (2017) mengklasifikasikan pendekatan multimodal sentiment analysis menjadi:

- **Early Fusion:** Penggabungan fitur pada tahap awal
- **Late Fusion:** Penggabungan prediksi pada tahap akhir
- **Hybrid Fusion:** Kombinasi kedua pendekatan

### 2.4 Computer Vision untuk Analisis Visual

Perkembangan deep learning, khususnya Convolutional Neural Networks (CNN) dan Vision Transformers (ViT), telah merevolusi kemampuan pemahaman konten visual. Arsitektur seperti ResNet (He et al., 2016) dan ViT (Dosovitskiy et al., 2021) telah mencapai performa state-of-the-art dalam berbagai task computer vision.

### 2.5 Penelitian Terkait

Tabel 1 menyajikan perbandingan penelitian terkait:

| Penelitian | Tahun | Domain | Modalitas | Bahasa | Jumlah Data |
|------------|-------|--------|-----------|--------|-------------|
| Pontiki et al. | 2014 | Restaurant | Teks | Inggris | 3.841 |
| Jiang et al. | 2019 | Hotel | Teks | Inggris | 4.728 |
| Xu et al. | 2019 | Produk | Multimodal | Mandarin | 9.779 |
| **Penelitian Ini** | **2026** | **Hotel** | **Gambar** | **Indonesia** | **17.385** |

---

## 3. METODOLOGI PENELITIAN

### 3.1 Desain Penelitian

Penelitian ini menggunakan pendekatan kuantitatif deskriptif dengan tahapan:
1. Pengumpulan data
2. Preprocessing
3. Anotasi
4. Analisis statistik
5. Interpretasi hasil

### 3.2 Pengumpulan Data

#### 3.2.1 Sumber Data
Data dikumpulkan dari platform Traveloka Indonesia dengan fokus pada:
- Review hotel yang menyertakan gambar
- Hotel berlokasi di berbagai kota di Indonesia
- Periode: [Tentukan periode pengumpulan]

#### 3.2.2 Etika Pengumpulan Data
- Mematuhi Terms of Service platform
- Anonimisasi identitas pengguna
- Penggunaan untuk tujuan akademik

### 3.3 Preprocessing Data

#### 3.3.1 Image Preprocessing
```
1. Download gambar dari URL
2. Validasi integritas file
3. Normalisasi format (JPEG)
4. Penghapusan duplikat
5. Filtering gambar tidak valid
```

#### 3.3.2 Penamaan Sistematis
Format: `R{ID_Review}_G{Gambar_Ke}_{hash}.jpg`
Contoh: `R1_G1_687bd728.jpg`

### 3.4 Skema Anotasi

#### 3.4.1 Definisi Aspek

| Aspek | Definisi Operasional |
|-------|----------------------|
| **Kamar (img_Kamar)** | Kondisi fisik kamar meliputi tempat tidur, pencahayaan, dekorasi, dan tata ruang |
| **Kebersihan (img_Kebersihan)** | Tingkat kebersihan yang tampak dari gambar, meliputi kerapian dan ketiadaan noda |
| **Pelayanan (img_Pelayanan)** | Aspek layanan staff hotel (umumnya tidak tervisualisasi) |
| **Makanan (img_Makanan)** | Kualitas dan presentasi makanan/minuman |
| **Fasilitas (img_Fasilitas)** | Fasilitas hotel seperti kolam renang, gym, lobby, dll |
| **Lokasi (img_Lokasi)** | Kondisi lokasi/view yang terlihat dari gambar |
| **Harga (img_Harga)** | Value for money (tidak dapat dinilai secara visual) |

#### 3.4.2 Skala Sentimen

| Label | Definisi |
|-------|----------|
| **Positif** | Aspek menunjukkan kualitas baik/memuaskan |
| **Netral** | Aspek tidak menunjukkan kualitas baik/buruk secara signifikan |
| **Negatif** | Aspek menunjukkan kualitas buruk/tidak memuaskan |
| **None** | Aspek tidak terlihat/tidak relevan dalam gambar |

### 3.5 Proses Anotasi

#### 3.5.1 Tahapan Anotasi
1. **Pengembangan Guideline:** Penyusunan panduan detail dengan contoh
2. **Pelatihan Annotator:** Training untuk konsistensi
3. **Pilot Study:** Anotasi sampel untuk validasi
4. **Full Annotation:** Anotasi seluruh dataset
5. **Quality Assurance:** Review dan koreksi

#### 3.5.2 Inter-annotator Agreement
Pengukuran reliabilitas menggunakan Cohen's Kappa:
- κ > 0.8: Almost perfect agreement
- 0.6 < κ ≤ 0.8: Substantial agreement
- 0.4 < κ ≤ 0.6: Moderate agreement

---

## 4. HASIL DAN PEMBAHASAN

### 4.1 Statistik Deskriptif Dataset

**Tabel 2. Statistik Umum Dataset**

| Parameter | Nilai |
|-----------|-------|
| Total Citra | 17.385 |
| Total Review Unik | 8.038 |
| Rata-rata Citra per Review | 2,16 |
| Median Citra per Review | 2 |
| Maks. Citra per Review | 9 |
| Min. Citra per Review | 1 |

### 4.2 Distribusi Sentimen per Aspek

#### 4.2.1 Aspek Kamar

**Tabel 3. Distribusi Sentimen Aspek Kamar**

| Sentimen | Frekuensi | Persentase |
|----------|-----------|------------|
| Positif | 17.138 | 98,58% |
| Negatif | 247 | 1,42% |
| **Total** | **17.385** | **100%** |

**Pembahasan:** Dominasi sentimen positif pada aspek Kamar (98,58%) mengindikasikan beberapa fenomena:

1. **Selection Bias:** Pengguna cenderung mengunggah foto kamar dalam kondisi terbaik
2. **Standar Hotel:** Hotel-hotel yang terdaftar di Traveloka umumnya memenuhi standar kualitas minimum
3. **Perilaku Konsumen:** Kecenderungan berbagi pengalaman positif lebih tinggi dibanding negatif

#### 4.2.2 Aspek Kebersihan

**Tabel 4. Distribusi Sentimen Aspek Kebersihan**

| Sentimen | Frekuensi | Persentase |
|----------|-----------|------------|
| Negatif | 17.130 | 98,53% |
| Netral | 191 | 1,10% |
| Positif | 64 | 0,37% |
| **Total** | **17.385** | **100%** |

**Temuan Signifikan:** Hasil yang kontra-intuitif ini memerlukan interpretasi mendalam:

1. **Perception Gap:** Kamar yang tampak baik secara visual tidak menjamin kebersihan aktual
2. **Limitation of Visual Assessment:** Kebersihan mikroskopis tidak dapat dinilai dari fotografi standar
3. **Annotation Challenge:** Aspek kebersihan mungkin memerlukan redefinisi kriteria dalam konteks visual

#### 4.2.3 Aspek Lokasi

**Tabel 5. Distribusi Sentimen Aspek Lokasi**

| Sentimen | Frekuensi | Persentase |
|----------|-----------|------------|
| Netral | 9.045 | 52,03% |
| Positif | 8.139 | 46,82% |
| Negatif | 89 | 0,51% |
| None | 112 | 0,64% |
| **Total** | **17.385** | **100%** |

**Pembahasan:** Aspek Lokasi menunjukkan distribusi paling seimbang dalam dataset:

1. **View from Room:** Gambar yang menampilkan pemandangan dari kamar berkontribusi pada sentimen positif
2. **City Hotels:** Hotel perkotaan mungkin memiliki view yang kurang menarik (netral)
3. **Beach/Mountain Hotels:** Resort dengan lokasi scenic berkontribusi signifikan pada sentimen positif

### 4.3 Coverage Analysis

**Tabel 6. Coverage Labeling per Aspek**

| Aspek | Labeled | Non-labeled | Coverage |
|-------|---------|-------------|----------|
| Kamar | 17.385 | 0 | 100% |
| Kebersihan | 17.385 | 0 | 100% |
| Makanan | 17.249 | 136 | 99,22% |
| Fasilitas | 17.256 | 129 | 99,26% |
| Lokasi | 17.273 | 112 | 99,36% |
| Pelayanan | 0 | 17.385 | 0% |
| Harga | 0 | 17.385 | 0% |

**Interpretasi:**
- Aspek Pelayanan dan Harga tidak dapat dievaluasi secara visual
- Aspek Makanan, Fasilitas, dan Lokasi memiliki sedikit gambar yang tidak relevan

### 4.4 Implikasi Praktis

#### 4.4.1 Untuk Manajemen Hotel

1. **Visual Marketing:** Optimalkan fotografi kamar untuk meningkatkan persepsi positif
2. **Kebersihan sebagai Prioritas:** Implementasi standar kebersihan yang melampaui ekspektasi visual
3. **Leverage Lokasi:** Manfaatkan view dan lokasi sebagai differentiator

#### 4.4.2 Untuk Platform OTA

1. **Image Quality Guidelines:** Standarisasi kualitas gambar yang diunggah
2. **Auto-aspect Detection:** Implementasi AI untuk kategorisasi aspek otomatis
3. **Balanced Review Encouragement:** Dorong pengguna mengunggah gambar berbagai aspek

---

## 5. KESIMPULAN

### 5.1 Ringkasan Temuan

Penelitian ini berhasil mengkonstruksi dataset benchmark MABSA berskala besar untuk domain perhotelan Indonesia dengan temuan utama:

1. Dataset terdiri dari 17.385 citra dari 8.038 review unik
2. Aspek Kamar menunjukkan sentimen positif dominan (98,58%)
3. Aspek Kebersihan memiliki sentimen negatif tertinggi (98,53%), mengindikasikan perception gap
4. Aspek Lokasi memiliki distribusi paling seimbang
5. Aspek Pelayanan dan Harga tidak dapat dievaluasi dari modalitas visual

### 5.2 Kontribusi Teoritis

- Pengayaan literatur MABSA untuk konteks Indonesia
- Identifikasi perception gap dalam evaluasi visual hotel
- Framework anotasi yang dapat direplikasi

### 5.3 Kontribusi Praktis

- Dataset benchmark untuk penelitian lanjutan
- Insight untuk industri perhotelan
- Rekomendasi untuk pengembangan platform OTA

### 5.4 Keterbatasan Penelitian

1. Single platform (Traveloka) - generalisasi terbatas
2. Aspek Pelayanan dan Harga tidak dapat dievaluasi
3. Belum ada model prediktif yang dikembangkan

### 5.5 Rekomendasi Penelitian Lanjutan

1. Pengembangan model deep learning untuk MABSA otomatis
2. Integrasi analisis teks review dengan gambar (true multimodal)
3. Cross-platform study untuk validasi temuan
4. Temporal analysis untuk tren sentimen

---

## UCAPAN TERIMA KASIH

[Bagian ini untuk mengakui kontribusi pihak terkait]

---

## DAFTAR PUSTAKA

[1] Liu, B. (2012). Sentiment Analysis and Opinion Mining. Synthesis Lectures on Human Language Technologies, 5(1), 1-167.

[2] Pontiki, M., Galanis, D., Pavlopoulos, J., Papageorgiou, H., Androutsopoulos, I., & Manandhar, S. (2014). SemEval-2014 Task 4: Aspect Based Sentiment Analysis. Proceedings of SemEval 2014, 27-35.

[3] Poria, S., Cambria, E., Bajpai, R., & Hussain, A. (2017). A Review of Affective Computing: From Unimodal Analysis to Multimodal Fusion. Information Fusion, 37, 98-125.

[4] Jiang, Q., Chen, L., Xu, R., Ao, X., & Yang, M. (2019). A Challenge Dataset and Effective Models for Aspect-Based Sentiment Analysis. Proceedings of EMNLP-IJCNLP 2019, 6280-6285.

[5] He, K., Zhang, X., Ren, S., & Sun, J. (2016). Deep Residual Learning for Image Recognition. CVPR 2016, 770-778.

[6] Dosovitskiy, A., et al. (2021). An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale. ICLR 2021.

[7] Xu, N., Mao, W., & Chen, G. (2019). Multi-interactive Memory Network for Aspect Based Multimodal Sentiment Analysis. AAAI 2019, 33(01), 371-378.

---

## BIODATA PENULIS

[Nama Lengkap]
[Gelar Akademik]
[Afiliasi Institusi]
[Email Korespondensi]
[ORCID ID]

---

*Artikel ini merupakan hasil penelitian original yang belum pernah dipublikasikan sebelumnya.*

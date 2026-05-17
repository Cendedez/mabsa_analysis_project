# Roadmap Labeling MABSA Hotel dengan Qwen2.5-VL 3B

## Spesifikasi Project
- **Task**: Multimodal Aspect-Based Sentiment Analysis (MABSA) untuk review hotel Traveloka
- **Data**: ~17.000 gambar hotel
- **Aspek**: Kamar, Kebersihan, Pelayanan, Harga, Lokasi, Fasilitas, Makanan
- **Sentimen**: Positif, Negatif, Netral, None
- **Hardware**: RTX 3060 Ti (8GB VRAM), RAM 16GB DDR4
- **Model**: Qwen2.5-VL 3B via Ollama

---

## Tahap 1: Persiapan Environment

### 1.1 Install Ollama

```bash
# Linux/WSL
curl -fsSL https://ollama.com/install.sh | sh

# Windows: Download dari https://ollama.com/download
```

### 1.2 Pull Model

```bash
ollama pull qwen2.5vl:3b
```

### 1.3 Verifikasi Model Berjalan

```bash
# Test sederhana
ollama run qwen2.5vl:3b "Apa yang kamu lihat di gambar ini?" --images ./R1_G1_687bd728.jpg
```

### 1.4 Install Python Dependencies

```bash
pip install ollama Pillow pandas tqdm
```

---

## Tahap 2: Preprocessing Gambar

### 2.1 Resize Gambar (Hemat VRAM)

Gambar besar akan makan VRAM lebih banyak. Resize ke max 512px untuk stabilitas.

```python
# preprocess_images.py
from PIL import Image
from pathlib import Path
import os

def resize_images(input_dir, output_dir, max_size=512):
    """Resize semua gambar ke max 512x512 untuk hemat VRAM"""
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    images = sorted(input_dir.glob("*.jpg"))
    print(f"Total gambar: {len(images)}")
    
    for i, img_path in enumerate(images):
        img = Image.open(img_path)
        img.thumbnail((max_size, max_size), Image.LANCZOS)
        
        output_path = output_dir / img_path.name
        img.save(output_path, "JPEG", quality=85)
        
        if (i + 1) % 500 == 0:
            print(f"  Resized: {i+1}/{len(images)}")
    
    print(f"Done! Output: {output_dir}")

if __name__ == "__main__":
    resize_images("./images_original", "./images_resized")
```

---

## Tahap 3: Prompt Engineering

### 3.1 Prompt Utama (Chain-of-Thought, Akurat)

```python
PROMPT = """Kamu adalah seorang annotator profesional untuk riset Multimodal Aspect-Based Sentiment Analysis (MABSA) pada review hotel.

TUGAS: Analisis gambar ini yang merupakan foto dari review tamu hotel di platform Traveloka. Tentukan sentimen visual untuk setiap aspek berikut.

LANGKAH 1 - OBSERVASI:
Perhatikan gambar dengan seksama. Identifikasi objek, kondisi, suasana, dan detail visual yang ada.

LANGKAH 2 - PENILAIAN PER ASPEK:
Untuk setiap aspek di bawah, tentukan apakah aspek tersebut TERLIHAT di gambar, lalu nilai sentimennya.

7 ASPEK YANG DINILAI:

1. Kamar
   - Apa yang dicari: tempat tidur, furniture, dinding, lantai, dekorasi, ukuran ruangan, pencahayaan kamar
   - Positif: kamar terlihat luas, bersih, tertata rapi, furniture bagus, pencahayaan baik
   - Negatif: kamar sempit, berantakan, furniture rusak/usang, gelap, cat mengelupas
   - None: gambar BUKAN menunjukkan kamar (misal: gambar outdoor, kolam, restoran)

2. Kebersihan
   - Apa yang dicari: noda, debu, sampah, kerapian, kondisi maintenance
   - Positif: terlihat bersih, mengkilap, rapi, terawat, tidak ada noda/sampah
   - Negatif: terlihat kotor, bernoda, berdebu, berantakan, ada sampah, jamur, rusak
   - None: tidak bisa menilai kebersihan dari gambar (misal: foto terlalu dekat/blur)

3. Pelayanan
   - Apa yang dicari: staf hotel, resepsionis, bellboy, pelayan restoran, interaksi dengan tamu
   - Positif: staf terlihat ramah, profesional, melayani tamu
   - Negatif: staf terlihat tidak profesional, area resepsi kosong/berantakan
   - None: TIDAK ada staf atau indikasi pelayanan terlihat di gambar (MAYORITAS gambar hotel tidak menunjukkan pelayanan)

4. Harga
   - Apa yang dicari: price tag, menu dengan harga, promosi, diskon, bill/struk
   - Positif: terlihat harga terjangkau, ada promo/diskon
   - Negatif: terlihat harga mahal/tidak wajar
   - None: TIDAK ada informasi harga terlihat di gambar (MAYORITAS gambar hotel tidak menunjukkan harga)

5. Lokasi
   - Apa yang dicari: pemandangan dari jendela/balkon, lingkungan sekitar hotel, view, akses jalan, landmark
   - Positif: pemandangan indah (laut, gunung, kota), lingkungan asri, lokasi strategis
   - Negatif: pemandangan buruk (tembok, kumuh), lingkungan kotor, terpencil
   - None: tidak ada pemandangan/lingkungan luar yang terlihat

6. Fasilitas
   - Apa yang dicari: kolam renang, gym, lobby, taman, playground, spa, wifi area, parkir, lift
   - Positif: fasilitas terlihat bagus, terawat, lengkap, modern
   - Negatif: fasilitas terlihat rusak, kotor, tidak terawat, ketinggalan zaman
   - None: tidak ada fasilitas umum hotel yang terlihat di gambar

7. Makanan
   - Apa yang dicari: makanan, minuman, penyajian, buffet, restoran hotel, sarapan
   - Positif: makanan terlihat menarik, porsi besar, penyajian rapi, variasi banyak
   - Negatif: makanan terlihat tidak menarik, porsi kecil, penyajian buruk
   - None: TIDAK ada makanan/minuman yang terlihat di gambar

LANGKAH 3 - ATURAN PENTING:
- Jika aspek TIDAK TERLIHAT sama sekali → WAJIB "None" (jangan menebak!)
- Jika terlihat tapi tidak jelas baik/buruk → "Netral"
- "Positif" HANYA jika ada bukti visual yang JELAS positif
- "Negatif" HANYA jika ada bukti visual yang JELAS negatif
- Pelayanan dan Harga biasanya "None" kecuali benar-benar terlihat di gambar

LANGKAH 4 - OUTPUT:
Berikan jawaban HANYA dalam format JSON berikut (tanpa teks lain sebelum atau sesudah):
{"Kamar":"...","Kebersihan":"...","Pelayanan":"...","Harga":"...","Lokasi":"...","Fasilitas":"...","Makanan":"..."}"""
```

### 3.2 Prompt Ringkas (Lebih Cepat, untuk Fallback)

```python
PROMPT_SHORT = """Analisis foto review hotel ini. Tentukan sentimen visual per aspek.

Aspek: Kamar, Kebersihan, Pelayanan, Harga, Lokasi, Fasilitas, Makanan

Label: "Positif" (jelas bagus), "Negatif" (jelas buruk), "Netral" (biasa saja), "None" (tidak terlihat di gambar)

PENTING: Jika aspek tidak terlihat di gambar = "None". Pelayanan dan Harga biasanya "None" karena jarang terlihat di foto.

Output JSON saja:
{"Kamar":"...","Kebersihan":"...","Pelayanan":"...","Harga":"...","Lokasi":"...","Fasilitas":"...","Makanan":"..."}"""
```

### 3.3 Catatan Prompt
- Prompt utama menggunakan Chain-of-Thought (langkah 1-4) untuk akurasi maksimal
- Setiap aspek dilengkapi contoh positif/negatif untuk mengurangi ambiguitas
- Pelayanan dan Harga secara eksplisit diarahkan ke "None" jika tidak terlihat
- `num_predict` disarankan 300 untuk prompt utama
- Estimasi speed: ~5-10 detik per gambar

Berikan jawaban AKHIR dalam format JSON:
{"Kamar":"...","Kebersihan":"...","Makanan":"...","Fasilitas":"...","Lokasi":"..."}"""
```

---

## Tahap 4: Script Batch Labeling

### 4.1 Script Utama

```python
# batch_labeling.py
import ollama
import json
import csv
import time
import sys
from pathlib import Path
from tqdm import tqdm

# === KONFIGURASI ===
MODEL = "qwen2.5vl:3b"
IMAGE_DIR = "./image_cache"          # Folder gambar resolusi asli
OUTPUT_CSV = "./hasil_labeling_qwen25vl.csv"
CHECKPOINT_CSV = "./checkpoint_labeling.csv"
BATCH_SAVE = 50                     # Save progress setiap N gambar
MAX_RETRIES = 3                     # Retry jika gagal
DELAY_BETWEEN = 0.5                 # Delay antar request (detik)

PROMPT = """Kamu adalah seorang annotator profesional untuk riset Multimodal Aspect-Based Sentiment Analysis (MABSA) pada review hotel.

TUGAS: Analisis gambar ini yang merupakan foto dari review tamu hotel di platform Traveloka. Tentukan sentimen visual untuk setiap aspek berikut.

LANGKAH 1 - OBSERVASI:
Perhatikan gambar dengan seksama. Identifikasi objek, kondisi, suasana, dan detail visual yang ada.

LANGKAH 2 - PENILAIAN PER ASPEK:
Untuk setiap aspek di bawah, tentukan apakah aspek tersebut TERLIHAT di gambar, lalu nilai sentimennya.

7 ASPEK YANG DINILAI:

1. Kamar
   - Apa yang dicari: tempat tidur, furniture, dinding, lantai, dekorasi, ukuran ruangan, pencahayaan kamar
   - Positif: kamar terlihat luas, bersih, tertata rapi, furniture bagus, pencahayaan baik
   - Negatif: kamar sempit, berantakan, furniture rusak/usang, gelap, cat mengelupas
   - None: gambar BUKAN menunjukkan kamar (misal: gambar outdoor, kolam, restoran)

2. Kebersihan
   - Apa yang dicari: noda, debu, sampah, kerapian, kondisi maintenance
   - Positif: terlihat bersih, mengkilap, rapi, terawat, tidak ada noda/sampah
   - Negatif: terlihat kotor, bernoda, berdebu, berantakan, ada sampah, jamur, rusak
   - None: tidak bisa menilai kebersihan dari gambar (misal: foto terlalu dekat/blur)

3. Pelayanan
   - Apa yang dicari: staf hotel, resepsionis, bellboy, pelayan restoran, interaksi dengan tamu
   - Positif: staf terlihat ramah, profesional, melayani tamu
   - Negatif: staf terlihat tidak profesional, area resepsi kosong/berantakan
   - None: TIDAK ada staf atau indikasi pelayanan terlihat di gambar (MAYORITAS gambar hotel tidak menunjukkan pelayanan)

4. Harga
   - Apa yang dicari: price tag, menu dengan harga, promosi, diskon, bill/struk
   - Positif: terlihat harga terjangkau, ada promo/diskon
   - Negatif: terlihat harga mahal/tidak wajar
   - None: TIDAK ada informasi harga terlihat di gambar (MAYORITAS gambar hotel tidak menunjukkan harga)

5. Lokasi
   - Apa yang dicari: pemandangan dari jendela/balkon, lingkungan sekitar hotel, view, akses jalan, landmark
   - Positif: pemandangan indah (laut, gunung, kota), lingkungan asri, lokasi strategis
   - Negatif: pemandangan buruk (tembok, kumuh), lingkungan kotor, terpencil
   - None: tidak ada pemandangan/lingkungan luar yang terlihat

6. Fasilitas
   - Apa yang dicari: kolam renang, gym, lobby, taman, playground, spa, wifi area, parkir, lift
   - Positif: fasilitas terlihat bagus, terawat, lengkap, modern
   - Negatif: fasilitas terlihat rusak, kotor, tidak terawat, ketinggalan zaman
   - None: tidak ada fasilitas umum hotel yang terlihat di gambar

7. Makanan
   - Apa yang dicari: makanan, minuman, penyajian, buffet, restoran hotel, sarapan
   - Positif: makanan terlihat menarik, porsi besar, penyajian rapi, variasi banyak
   - Negatif: makanan terlihat tidak menarik, porsi kecil, penyajian buruk
   - None: TIDAK ada makanan/minuman yang terlihat di gambar

LANGKAH 3 - ATURAN PENTING:
- Jika aspek TIDAK TERLIHAT sama sekali → WAJIB "None" (jangan menebak!)
- Jika terlihat tapi tidak jelas baik/buruk → "Netral"
- "Positif" HANYA jika ada bukti visual yang JELAS positif
- "Negatif" HANYA jika ada bukti visual yang JELAS negatif
- Pelayanan dan Harga biasanya "None" kecuali benar-benar terlihat di gambar

LANGKAH 4 - OUTPUT:
Berikan jawaban HANYA dalam format JSON berikut (tanpa teks lain sebelum atau sesudah):
{"Kamar":"...","Kebersihan":"...","Pelayanan":"...","Harga":"...","Lokasi":"...","Fasilitas":"...","Makanan":"..."}"""

ASPECTS = ['Kamar', 'Kebersihan', 'Pelayanan', 'Harga', 'Lokasi', 'Fasilitas', 'Makanan']
VALID_LABELS = ['Positif', 'Negatif', 'Netral', 'None']


def extract_json(text):
    """Ekstrak JSON dari response model"""
    try:
        start = text.find('{')
        end = text.rfind('}') + 1
        if start != -1 and end > start:
            json_str = text[start:end]
            result = json.loads(json_str)
            # Validasi keys
            for key in ASPECTS:
                if key not in result:
                    return None
                if result[key] not in VALID_LABELS:
                    result[key] = 'None'  # Default jika label tidak valid
            return result
    except (json.JSONDecodeError, ValueError):
        pass
    return None


def label_single_image(image_path):
    """Label satu gambar dengan retry mechanism"""
    for attempt in range(MAX_RETRIES):
        try:
            response = ollama.chat(
                model=MODEL,
                messages=[{
                    'role': 'user',
                    'content': PROMPT,
                    'images': [str(image_path)]
                }],
                options={
                    'temperature': 0,
                    'num_predict': 300,
                    'top_p': 0.1,
                }
            )
            
            raw_output = response['message']['content']
            result = extract_json(raw_output)
            
            if result:
                return result, raw_output
            else:
                if attempt < MAX_RETRIES - 1:
                    time.sleep(2)
                    
        except Exception as e:
            print(f"\n  Error attempt {attempt+1}: {e}")
            time.sleep(5)
    
    return None, None


def load_checkpoint(checkpoint_path):
    """Load gambar yang sudah diproses dari checkpoint"""
    done = set()
    if Path(checkpoint_path).exists():
        with open(checkpoint_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                done.add(row['filename'])
    return done


def save_results(results, output_path):
    """Save hasil ke CSV"""
    if not results:
        return
    fields = ['filename'] + ASPECTS
    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(results)


def main():
    image_dir = Path(IMAGE_DIR)
    images = sorted(image_dir.glob("*.jpg"))
    
    if not images:
        print(f"ERROR: Tidak ada gambar .jpg di {IMAGE_DIR}")
        sys.exit(1)
    
    # Load checkpoint (gambar yang sudah selesai)
    done_files = load_checkpoint(CHECKPOINT_CSV)
    remaining = [img for img in images if img.name not in done_files]
    
    print(f"=" * 60)
    print(f"MABSA Image Labeling - Qwen2.5-VL 3B")
    print(f"=" * 60)
    print(f"Total gambar     : {len(images)}")
    print(f"Sudah diproses   : {len(done_files)}")
    print(f"Sisa             : {len(remaining)}")
    print(f"Estimasi waktu   : {len(remaining) * 7 / 3600:.1f} jam")
    print(f"Aspek            : {', '.join(ASPECTS)}")
    print(f"=" * 60)
    
    if not remaining:
        print("Semua gambar sudah dilabel!")
        return
    
    # Load existing results
    results = []
    if Path(CHECKPOINT_CSV).exists():
        with open(CHECKPOINT_CSV, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            results = list(reader)
    
    # Proses gambar
    success = 0
    failed = 0
    
    for i, img_path in enumerate(tqdm(remaining, desc="Labeling")):
        label, raw = label_single_image(img_path)
        
        if label:
            label['filename'] = img_path.name
            results.append(label)
            success += 1
        else:
            error_row = {'filename': img_path.name}
            for asp in ASPECTS:
                error_row[asp] = 'ERROR'
            results.append(error_row)
            failed += 1
            tqdm.write(f"  FAILED: {img_path.name}")
        
        # Save checkpoint
        if (i + 1) % BATCH_SAVE == 0:
            save_results(results, CHECKPOINT_CSV)
        
        time.sleep(DELAY_BETWEEN)
    
    # Final save
    save_results(results, OUTPUT_CSV)
    save_results(results, CHECKPOINT_CSV)
    
    print(f"\n{'=' * 60}")
    print(f"SELESAI!")
    print(f"Success: {success} | Failed: {failed}")
    print(f"Output : {OUTPUT_CSV}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
```

---

## Tahap 5: Evaluasi Kualitas

### 5.1 Cek Distribusi Label

```python
# evaluate_labels.py
import pandas as pd
from collections import Counter

def evaluate_distribution(csv_path):
    """Analisis distribusi label untuk deteksi bias"""
    df = pd.read_csv(csv_path)
    
    aspects = ['Kamar', 'Kebersihan', 'Pelayanan', 'Harga', 'Lokasi', 'Fasilitas', 'Makanan']
    
    print("=" * 60)
    print("DISTRIBUSI LABEL PER ASPEK")
    print("=" * 60)
    
    for asp in aspects:
        counts = df[asp].value_counts()
        total = len(df)
        print(f"\n{asp}:")
        for label, count in counts.items():
            pct = count / total * 100
            bar = "█" * int(pct / 2)
            print(f"  {label:8s}: {count:5d} ({pct:5.1f}%) {bar}")
        
        # Deteksi bias
        max_pct = counts.iloc[0] / total * 100
        if max_pct > 90:
            print(f"  ⚠️  BIAS TERDETEKSI: {counts.index[0]} = {max_pct:.1f}%")
    
    # Cek jumlah pola unik
    patterns = df[aspects].apply(tuple, axis=1)
    unique_patterns = patterns.nunique()
    print(f"\n{'=' * 60}")
    print(f"Pola unik: {unique_patterns} dari {total} data")
    if unique_patterns < 50:
        print("⚠️  Variasi terlalu rendah! Model mungkin masih bias.")
    elif unique_patterns > 100:
        print("✅ Variasi cukup baik.")
    print(f"{'=' * 60}")

if __name__ == "__main__":
    evaluate_distribution("./hasil_labeling_qwen25vl.csv")
```

### 5.2 Validasi Sample Acak (Manual/GPT-4o)

```python
# sample_validation.py
import pandas as pd
import random

def create_validation_sample(csv_path, n=50):
    """Ambil sample acak untuk validasi manual"""
    df = pd.read_csv(csv_path)
    sample = df.sample(n=min(n, len(df)), random_state=42)
    sample.to_csv("./validation_sample.csv", index=False)
    print(f"Sample {len(sample)} gambar disimpan di validation_sample.csv")
    print("Buka gambar-gambar tersebut dan bandingkan dengan label model.")

if __name__ == "__main__":
    create_validation_sample("./hasil_labeling_qwen25vl.csv", n=50)
```

---

## Tahap 6: Post-Processing & Finalisasi

### 6.1 Gabungkan dengan Data Original

```python
# merge_results.py
import pandas as pd

def merge_with_original(original_csv, new_labels_csv, output_csv):
    """Gabungkan label baru dengan data original"""
    original = pd.read_csv(original_csv)
    new_labels = pd.read_csv(new_labels_csv)
    
    # Rename kolom label baru
    rename_map = {
        'Kamar': 'img_Kamar',
        'Kebersihan': 'img_Kebersihan',
        'Pelayanan': 'img_Pelayanan',
        'Harga': 'img_Harga',
        'Lokasi': 'img_Lokasi',
        'Fasilitas': 'img_Fasilitas',
        'Makanan': 'img_Makanan'
    }
    new_labels = new_labels.rename(columns=rename_map)
    
    # Merge berdasarkan filename
    merged = original.merge(
        new_labels[['filename'] + list(rename_map.values())],
        left_on='Image_FileName',
        right_on='filename',
        how='left',
        suffixes=('_old', '_new')
    )
    
    merged.to_csv(output_csv, index=False)
    print(f"Merged: {len(merged)} rows → {output_csv}")

if __name__ == "__main__":
    merge_with_original(
        "./hasil_labeling_image_per_gambar.csv",
        "./hasil_labeling_qwen25vl.csv",
        "./hasil_labeling_final.csv"
    )
```

---

## Timeline & Estimasi

| Tahap | Aktivitas | Estimasi Waktu |
|-------|-----------|---------------|
| 1 | Setup Environment | 30 menit |
| 2 | Preprocessing (resize) | 15-30 menit |
| 3 | Test prompt (50 gambar) | 1 jam |
| 4 | Evaluasi sample, perbaiki prompt | 2-3 jam |
| 5 | Batch labeling penuh (17k) | 14-24 jam |
| 6 | Evaluasi final & post-processing | 2 jam |
| **Total** | | **~2-3 hari** |

---

## Tips & Troubleshooting

### Monitor GPU
```bash
# Terminal terpisah, monitor VRAM
watch -n 2 nvidia-smi
```

### Jika OOM (Out of Memory)
1. Pastikan gambar sudah di-resize ke max 512px
2. Tutup semua aplikasi lain (browser, VSCode, dll)
3. Tambah delay antar request: `DELAY_BETWEEN = 2`
4. Restart Ollama: `ollama stop qwen2.5vl:3b && ollama run qwen2.5vl:3b`

### Jika Output Tidak Valid JSON
- Script sudah handle retry otomatis (3x)
- Gambar yang gagal ditandai "ERROR" dan bisa di-retry terpisah

### Resume dari Checkpoint
- Script otomatis skip gambar yang sudah ada di checkpoint
- Jika crash, tinggal jalankan ulang: `python batch_labeling.py`

### Kualitas Label Masih Buruk?
1. Coba PROMPT_COT (Chain-of-Thought) — lebih lambat tapi lebih akurat
2. Naikkan `num_predict` ke 300 jika pakai COT
3. Pertimbangkan 2-pass: label dulu, lalu verifikasi dengan prompt kedua

---

## Struktur File Project

```
mabsa_analysis_project/
├── images_original/          # Gambar asli dari Traveloka
├── images_resized/           # Gambar yang sudah di-resize 512px
├── preprocess_images.py      # Script resize gambar
├── batch_labeling.py         # Script labeling utama
├── evaluate_labels.py        # Script evaluasi distribusi
├── sample_validation.py      # Script ambil sample validasi
├── merge_results.py          # Script merge dengan data original
├── hasil_labeling_qwen25vl.csv    # Output final
├── checkpoint_labeling.csv   # Checkpoint (auto-save)
└── validation_sample.csv     # Sample untuk validasi manual
```

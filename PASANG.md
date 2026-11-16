# ImlaScan — paket siap pasang

Paket ini berisi semua yang dibutuhkan untuk menjalankan ImlaScan di komputer lain,
**kecuali** tiga hal yang sengaja tidak disertakan:

| Tidak disertakan | Alasan | Yang harus dilakukan |
|---|---|---|
| `.env` (API key Gemini, secret key) | rahasia | salin `.env.example` menjadi `.env`, lalu isi |
| Lingkungan Python (`venv311`, `venv_areta`) | khusus tiap komputer, sangat besar | dibuat ulang, lihat langkah 2 dan 3 |
| Data esai mahasiswa dan Excel verifikasi | data pribadi | tidak diperlukan untuk menjalankan aplikasi |

## Isi paket

| Folder / berkas | Isi |
|---|---|
| `home/`, `gramatical/`, `manage.py` | aplikasi web (Django) |
| `models/D_tibyan_v2/` | model koreksi AraT5 (model D, 1,4 GB) |
| `tessdata/ara.traineddata` | model bahasa Arab untuk Tesseract |
| `areta_service/` | layanan penanda jenis kesalahan (ARETA), termasuk `vendor/` dan kamus `camel_data/` |
| `pipeline/` | skrip pembuatan korpus dan notebook training |
| `jalankan_imlascan.bat` | peluncur satu klik (Windows) |
| `_uji_*.py` | uji fungsi, beban, dan konfigurasi produksi |
| `CARA_JALANKAN.md`, `.env.example` | petunjuk harian dan daftar pengaturan |

## Pemasangan di Windows

1. **Ekstrak** paket ke folder tanpa spasi di jalurnya, misalnya `C:\imlascan`.

2. **Lingkungan aplikasi web (Python 3.11):**
   ```powershell
   cd C:\imlascan
   py -3.11 -m venv venv311
   venv311\Scripts\pip install -r requirements.txt
   ```
   Untuk memakai GPU, pasang `torch` sesuai versi CUDA dari https://pytorch.org/get-started/locally/
   sebelum perintah `pip install -r requirements.txt`.

3. **Lingkungan ARETA (Python 3.8, terpisah):**
   ```powershell
   py -3.8 -m venv venv_areta
   venv_areta\Scripts\pip install --no-deps -r areta_service\requirements-areta-win.txt
   ```
   Kamus morfologi sudah ada di `areta_service\camel_data`, jadi tidak perlu diunduh lagi.

4. **Tesseract:** `winget install --id UB-Mannheim.TesseractOCR -e`
   (model bahasa Arab sudah ada di `tessdata\`).

5. **Pengaturan:** salin `.env.example` menjadi `.env`, lalu isi minimal:
   ```
   DJANGO_SECRET_KEY=<buat baru, lihat komentar di .env.example>
   IMLASCAN_MODEL_PATH=C:\imlascan\models\D_tibyan_v2
   IMLASCAN_GEMINI_KEY=<API key dari aistudio.google.com/apikey>
   IMLASCAN_DEVICE=cuda        (atau cpu bila tanpa GPU)
   ```

6. **Jalankan:** klik dua kali `jalankan_imlascan.bat`, lalu buka http://127.0.0.1:8000

7. **Periksa pemasangan** (jendela ARETA harus terbuka):
   ```powershell
   venv311\Scripts\python.exe -X utf8 _uji_produksi.py
   ```
   Uji lain (`_uji_menyeluruh.py`, `_uji_model_d.py`) memakai gambar dan data uji yang tidak ada di paket ini,
   jadi hanya bisa dijalankan di komputer asal.

## Server Linux

Paket ini bisa diekstrak di Linux, tetapi berkas `.bat` tidak berlaku di sana dan skrip deploy
(gunicorn, nginx, HTTPS, layanan systemd untuk ARETA) **belum dibuat**. Catatan untuk nanti:

- Pakai **satu proses dengan beberapa thread**; kuota Gemini dan pembatas laju dihitung per proses.
- Wajib di `.env`: `DJANGO_DEBUG=False`, `DJANGO_ALLOWED_HOSTS`, `DJANGO_SECRET_KEY`,
  `IMLASCAN_BEHIND_PROXY=True`, `DJANGO_CSRF_TRUSTED_ORIGINS`, `IMLASCAN_PRELOAD=True`.
- Kebutuhan memori terukur di laptop: proses web ±2,4 GB, ARETA ±1,1 GB.
- Kecepatan tanpa GPU belum diukur.

## Hal yang perlu diingat

- **Jangan sebarkan paket ini ke publik.** Model D dilatih lewat korpus Tibyan yang repositorinya
  tidak mencantumkan lisensi, dan `tokenizer.json` di folder model wajib ada (tanpa itu hasil koreksi
  menurun tanpa pesan error).
- Kuota Gemini OCR: 30 halaman per hari untuk semua pengguna; saat habis otomatis memakai Tesseract.
- Ini demo riset: model masih melewatkan lebih dari separuh kesalahan pada data uji.

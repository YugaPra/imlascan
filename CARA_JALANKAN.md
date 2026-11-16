# Cara Menjalankan ImlaScan (Gramatical)

## Cara tercepat

Klik dua kali **`jalankan_imlascan.bat`**. Berkas ini menyalakan layanan ARETA (jendela terpisah),
menunggu sampai siap, lalu menyalakan server web. Buka http://127.0.0.1:8000.

Di halaman hasil, baris "Mesin koreksi & penanda jenis kesalahan" harus bertuliskan **AraT5 + areta**.
Bila tertulis "AraT5 + heuristic", layanan ARETA tidak aktif.

Uji menyeluruh (ARETA harus aktif): `venv311\Scripts\python.exe -X utf8 _uji_menyeluruh.py`

## Kuota Gemini dan pengaman

- **Kuota global Gemini OCR: 30 halaman per hari** untuk semua pengguna (`IMLASCAN_GEMINI_DAILY_MAX` di `.env`).
  Saat habis, atau saat Gemini gagal / terlalu lama, unggahan otomatis dibaca Tesseract dan pengguna diberi tahu.
- Hitungan kuota dan pemakaian token riil ada di `.imlascan_kuota.json` (dasar hitung biaya).
- Batas lain (nilai bawaan): berkas 10 MB, PDF 5 halaman, gambar 40 megapiksel, teks 20.000 karakter,
  OCR maksimal 150 detik per unggahan, 30 analisis dan 10 OCR per 10 menit per alamat IP.
- Daftar lengkap variabel ada di `.env.example`.

## Uji sebelum deploy (ARETA harus aktif)

| Perintah (`venv311\Scripts\python.exe -X utf8 ...`) | Yang diuji |
|---|---|
| `_uji_menyeluruh.py` | semua fungsi: OCR, ARETA, batas, kuota, kebocoran key, ketahanan (36 uji) |
| `_uji_beban.py` | server sungguhan mode produksi: permintaan bersamaan, memori, pembatas laju |
| `_uji_produksi.py` | konfigurasi `DJANGO_DEBUG=False` |
| `_uji_model_d.py` | model terpasang benar (F0.5 0,579 di test v2) |

## Menjalankan aplikasi (manual)

Buka PowerShell, lalu:

```powershell
cd "$HOME\gramatical"
venv311\Scripts\Activate.ps1
python manage.py runserver
```

Buka browser: http://127.0.0.1:8000

> Kalau `Activate.ps1` diblokir: jalankan `Set-ExecutionPolicy -Scope Process -Bypass` lalu ulangi.

### Tagger ARETA (opsional, disarankan)

Sebelum `runserver`, klik dua kali **`areta_service\jalankan_areta.bat`** dan biarkan jendelanya terbuka.
Tanpa itu, ImlaScan tetap jalan memakai tagger heuristik lama. Detail ada di `areta_service\README.md`.

### Pilihan mesin OCR

Di form unggah ada tiga tombol untuk PDF/gambar:

- **Gemini OCR** (default): hanya Gemini yang dijalankan.
- **Tesseract OCR**: hanya Tesseract (lokal, tanpa internet, tanpa kuota API).
- **Bandingkan keduanya**: dua-duanya dijalankan; hasil tampil berdampingan + persentase kemiripan
  + kata yang berbeda disorot, dengan tombol "Analisis ulang" untuk memakai teks mesin lainnya.

Sudah terpasang: Tesseract 5.4.0 di `C:\Program Files\Tesseract-OCR\` dan model Arab di
`tessdata\ara.traineddata`. Bila pindah laptop:

```powershell
winget install --id UB-Mannheim.TesseractOCR -e
mkdir tessdata
Invoke-WebRequest https://github.com/tesseract-ocr/tessdata_best/raw/main/ara.traineddata -OutFile tessdata\ara.traineddata
venv311\Scripts\pip install pytesseract
```

## Menghentikan

Di jendela PowerShell server, tekan **Ctrl + C**.

## Penting

- **Selalu pakai `venv311`** (bukan `venv` lama yang Python 3.14).
- File `.env` menyimpan API key & path model — **jangan dibagikan / jangan di-commit ke GitHub**.
- Token OCR `AQ.` bersifat sementara; bila OCR tiba-tiba error auth, ambil token/key baru
  dari aistudio.google.com/apikey (idealnya API key `AIza` yang permanen).
- Model AraT5 yang dipakai: `models\D_tibyan_v2` (model D, F0.5 0,579 di test v2). Folder itu WAJIB berisi
  `tokenizer.json`; tanpa berkas itu hasil koreksi menurun tanpa pesan error.
  Model lama (checkpoint-540) masih ada di `D:\All projects\clone yuga\arat5-gec-20260720T113623Z-1-001\arat5-gec`;
  untuk kembali, ubah `IMLASCAN_MODEL_PATH` di `.env` (cadangan: `.env.bak_sebelum_model_D`).
- Cek pemasangan model: `venv311\Scripts\python.exe -X utf8 _uji_model_d.py` (harus F0.5 0,579).

## Konfigurasi (ringkas, di .env)

| Variabel | Fungsi |
|---|---|
| `IMLASCAN_MODEL_PATH` | Folder model AraT5 |
| `IMLASCAN_TOKENIZER` | Sumber tokenizer (UBC-NLP/AraT5v2-base-1024) |
| `IMLASCAN_DEVICE` | `cuda` (GPU) atau `cpu` |
| `IMLASCAN_OCR_PROVIDER` | `gemini` atau `vision` |
| `IMLASCAN_GEMINI_KEY` | API key Gemini |
| `IMLASCAN_GEMINI_MODEL` | `gemini-flash-latest` |

## File bantu (boleh dihapus)

`debug_model.py` dan `list_models.py` hanya untuk diagnosa — aman dihapus.

## Alur sistem

Teks/PDF/gambar → (OCR bila PDF/gambar: Gemini, Tesseract, atau keduanya sesuai tombol yang dipilih) → model AraT5 → deteksi imla'
(ortografi + morfologi, tag ARETA) → hasil di web.
Sintaksis/I'rab di luar cakupan (sesuai arahan Bu Mia).

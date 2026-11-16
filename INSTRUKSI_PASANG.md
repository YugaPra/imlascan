# INSTRUKSI PEMASANGAN — Patch ImlaScan (Langkah 1 + Keamanan)

Folder ini berisi **5 file hasil edit** untuk repo Gramatical (Yuga). Semua sudah
diuji: `manage.py check` = 0 issues, kalimat demo = 6 error.

## A. Salin project ke folder BERSIH (sekaligus fix "Access Denied" venv)

Masalah venv gagal (WinError 5) muncul karena folder hasil ekstrak ZIP read-only.
Solusi: pindahkan ke folder baru yang writable.

```powershell
mkdir D:\gramatical
Copy-Item "D:\All projects\clone yuga\Gramatical-main\Gramatical-main\*" D:\gramatical -Recurse
```

## B. Timpa 5 file dengan versi patch

Letakkan file dari folder `gramatical_patch/` ke lokasi berikut di `D:\gramatical\`:

| File patch | Tujuan di project |
|---|---|
| `home/views.py` | `D:\gramatical\home\views.py` |
| `home/templates/home/index.html` | `D:\gramatical\home\templates\home\index.html` |
| `gramatical/settings.py` | `D:\gramatical\gramatical\settings.py` |
| `.gitignore` | `D:\gramatical\.gitignore` (root) |
| `.env.example` | `D:\gramatical\.env.example` (root) |

## C. Buat file .env (berisi SECRET_KEY, JANGAN di-commit)

```powershell
Copy-Item D:\gramatical\.env.example D:\gramatical\.env
```
`.env` sudah diblokir `.gitignore`. SECRET_KEY di dalamnya sudah di-generate baru.

## D. Jalankan (disarankan Python 3.12; 3.14 berisiko untuk tahap model)

```powershell
cd D:\gramatical
python -m venv venv
venv\Scripts\Activate.ps1          # jika diblok: Set-ExecutionPolicy -Scope Process -Bypass
pip install django python-dotenv
python manage.py migrate
python manage.py runserver
```
Buka http://127.0.0.1:8000 → tab "Salin Teks" → ketik:
`احمد ذهب الي المدرسه فى مدينه جميله` → harus muncul **6 kesalahan**.

## E. Commit keamanan (jika repo git sudah tersedia)

```bash
git rm --cached db.sqlite3
git add .gitignore .env.example gramatical/settings.py home/views.py home/templates/home/index.html
git commit -m "Langkah 1: scope ortografi+morfologi, tabel tag ARETA, pipeline; Security: env-based secret key & gitignore"
```

---

## F. Mencolok model AraT5 (lokal, dari Google Drive)

File `home/gec_model.py` sudah menangani inferensi. Anda **tidak perlu menulis
kode** — cukup arahkan ke folder model.

1. **Download model dari GDrive** ke folder lokal, mis. `D:\models\arat5-gec\`.
   Folder itu harus berisi hasil `save_pretrained()`: `config.json`,
   `model.safetensors` (atau `pytorch_model.bin`), dan file tokenizer
   (`spiece.model` / `tokenizer.json`). **Jika di GDrive hanya ada checkpoint
   `.pt`/`.bin` mentah, itu belum cukup** — perlu diekspor via
   `model.save_pretrained()` + `tokenizer.save_pretrained()` dulu.

2. **Set path di `.env`:**
   ```
   IMLASCAN_MODEL_PATH=D:\models\arat5-gec
   IMLASCAN_DEVICE=cuda            # pakai GPU laptop; "cpu" bila tanpa GPU
   ```

3. **Pasang dependency (venv Python 3.12):**
   ```powershell
   pip install -r requirements.txt
   ```
   Untuk GPU, install `torch` sesuai versi CUDA dari pytorch.org (bukan sekadar
   `pip install torch`).

4. **Jalankan `runserver`.** Bila path valid & torch ada → pakai model.
   Bila belum → otomatis **fallback ke mesin lexicon** (app tidak crash).

**Yang perlu Anda cek/konfirmasi:**
- Saat training, apakah Anda memakai **prefix tugas** (mis. `"correct: "` atau
  teks Arab langsung)? Bila ya, tambahkan di `_correct_text()` (ada catatannya).
- Tagger saat ini **heuristik** (beda karakter → OH/OT/OA/...). Untuk tag ARETA
  presisi (evaluasi Bu Mia), perlu integrasi CAMeL Tools — tahap terpisah.

---

## Ringkasan perubahan

**views.py** — di-refactor jadi pipeline:
- `TAG_INFO`: tabel 13 tag ARETA (11 ortografi + MI, MT) dengan penjelasan Indonesia. Definisi dari Belkebir & Habash (2021), Tabel 1.
- `SCOPE_TAGS` + `_in_scope()`: filter scope. MG/SP **sengaja belum** dimasukkan (menunggu Bu Mia).
- `run_gec_model()`: **LUBANG COLOKAN** model AraT5 (Langkah 2). Kontrak I/O terdokumentasi. Sekarang berisi mesin sementara (lexicon rule-based, sudah diperluas).
- `analyze_imla()`: orkestrator; bentuk return tetap → template tak berubah.
- OCR upload dibuat **jujur**: hanya `.txt` diproses; gambar/PDF memberi pesan "belum aktif" (versi lama memalsukan 6 error).

**index.html** — branding diselaraskan scope:
- "I'rabScan" → **"ImlaScan"** (header, title, footer).
- Checkbox "Deteksi I'rab Lengkap" & "Visualisasi Struktur" **dihapus** → diganti panel "Cakupan Analisis" (Ortografi ✓, Morfologi ✓, Sintaksis/I'rab ✗).
- Teks "algoritma saraf deep-learning Nahwu" & "deteksi subjek/predikat" → diganti deskripsi Imla'/morfologi yang jujur.
- Bug JS `}x` (baris ~427) diperbaiki; warna `positive`/`positive-surface` yang hilang ditambahkan (state "Teks Sempurna" kini tampil benar); `© 2024` → `© 2026`.

**settings.py / .gitignore / .env.example** — keamanan:
- SECRET_KEY & DEBUG dari environment (`.env`), bukan hardcoded.
- `.gitignore` memblok `db.sqlite3`, `.env`, bobot model, data sensitif.

> Catatan: kunci lama tetap ada di **riwayat git** (repo sempat public). Tak berbahaya
> selama belum deploy publik; tak perlu rewrite history kecuali benar-benar go-live.

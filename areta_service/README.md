# Layanan ARETA untuk ImlaScan (opsi B)

ARETA memberi label jenis kesalahan (OH, OT, XG, ...) untuk pasangan *kalimat murid → kalimat koreksi*.
Layanan ini dipakai `home/gec_model.py` untuk menggantikan `infer_tag()`.

```
foto/teks ──► AraT5 (koreksi) ──► layanan ARETA :8765 (label) ──► filter scope (views.py) ──► tampilan
                                        │ mati / error?
                                        └──► otomatis kembali ke infer_tag() lama
```

## Menjalankan (setiap kali)

1. Klik dua kali **`areta_service\jalankan_areta.bat`**, lalu tunggu sampai muncul `Siap`.
2. Jalankan ImlaScan seperti biasa (`venv311`, `python manage.py runserver`).

Layanan ini internal: tidak ada halaman untuk dibuka pengguna. Pengguna cukup memakai ImlaScan
(http://127.0.0.1:8000). Untuk mengecek layanan hidup: http://127.0.0.1:8765/api/health

Untuk mematikan ARETA tanpa menghentikan layanan, set `IMLASCAN_TAGGER=heuristic` di `.env`.

## Isi folder

| File | Fungsi |
|---|---|
| `areta_service.py` | Layanan: memuat ARETA sekali, lalu menyajikan API `/api/annotate` dan `/api/health` |
| `areta_client.py` | Klien kecil (pustaka standar saja), dipakai oleh `gec_model.py` |
| `lev_shim/` | Pengganti `python-Levenshtein 0.12.0` dalam Python murni (lihat catatan) |
| `requirements-areta-win.txt` | Paket teruji untuk Windows + Python 3.8 (tanpa torch) |
| `vendor/arabic-gec/` | Kode ARETA (CAMeL-Lab, MIT), commit `8c7fb84`. Tidak di-commit. |
| `_wintest.py` + `_wintest.b64` | Uji kesetaraan: hasil harus identik dengan rujukan Linux |

## Pasang ulang dari nol (Windows, tanpa WSL)

```powershell
cd C:\Users\Habib\gramatical
git clone --filter=blob:none --no-checkout https://github.com/CAMeL-Lab/arabic-gec.git areta_service\vendor\arabic-gec
cd areta_service\vendor\arabic-gec; git sparse-checkout set areta alignment; git checkout 8c7fb84f3ed84d1d30beb7080b02d0f6bfe1c5bf; cd ..\..\..
uv venv -p C:\Users\Habib\AppData\Local\Programs\Python\Python38\python.exe venv_areta
$env:VIRTUAL_ENV="$PWD\venv_areta"; uv pip install --no-deps -r areta_service\requirements-areta-win.txt
.\venv_areta\Scripts\camel_data.exe -i light      # kamus morfologi, ~600 MB
```

Cek hasilnya: jalankan layanan, lalu `venv311\Scripts\python.exe areta_service\_wintest.py`. Output harus `COCOK dengan rujukan Linux: True`.

## Catatan teknis (kenapa ada tambalan)

ARETA ditulis untuk Linux + Python 3.7. Agar hasilnya **identik** di Windows, ada empat tambalan. Semuanya dipasang dari luar, tanpa mengubah kode ARETA:

1. **`python-Levenshtein 0.12.0`** butuh compiler C di Windows. Versi baru punya wheel, tapi mengubah ~0,2% label (memilih jalur edit berbeda saat ada seri). Solusinya `lev_shim/`, salinan algoritma C 0.12.0 dalam Python murni. Sudah diverifikasi identik pada 222.103 pasangan kata.
2. **Encoding:** ARETA membuka file tanpa `encoding=`. Di Windows itu menjadi cp1252, sehingga huruf Arab gagal ditulis. Solusinya, layanan selalu jalan dengan `python -X utf8`.
3. **Akhir baris:** file tulisan ARETA di Windows berakhir `\r\n`, sehingga baris pemisah tak dikenali dan label multi-kata berubah (UNK/S → X). Solusinya, alignment ditulis sendiri dengan `\n`, dan `open()` di `aligner/align_text_api.py` ditambal.
4. **Folder sementara:** Python 3.8 di Windows gagal membersihkan `TemporaryDirectory` yang masih dipegang ARETA. Solusinya memakai folder kerja tetap (`%TEMP%\areta_service_work`).

Hasil verifikasi: 4.956 label (30 esai pelajar L2 QALB-2015 + 8 contoh) **identik** dengan rujukan Linux. Di Linux, 24.190 label pada 158 esai QALB-2015 L2 juga identik dengan ARETA asli (Levenshtein C 0.12.0).

## Rujukan
- Belkebir & Habash (2021), *Automatic Error Type Annotation for Arabic*, CoNLL. https://aclanthology.org/2021.conll-1.47
- Alhafni dkk. (2023), *Advancements in Arabic GED and GEC*, EMNLP. https://github.com/CAMeL-Lab/arabic-gec

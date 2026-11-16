@echo off
REM Menjalankan layanan ARETA (tagger jenis kesalahan) untuk ImlaScan.
REM Biarkan jendela ini terbuka selama ImlaScan dipakai. Tutup / Ctrl+C untuk berhenti.
cd /d "%~dp0"
set PYTHONIOENCODING=utf-8
REM Kamus CAMeL disimpan di folder proyek (bukan %APPDATA%, yang bisa tervirtualisasi oleh aplikasi MSIX).
set CAMELTOOLS_DATA=%~dp0camel_data
"..\venv_areta\Scripts\python.exe" areta_service.py --repo vendor\arabic-gec
pause

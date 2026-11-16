@echo off
REM Menjalankan ImlaScan lengkap: layanan ARETA (jendela terpisah) + server web.
REM Tutup kedua jendela (atau Ctrl+C) untuk berhenti.
cd /d "%~dp0"
echo [1/2] Menyalakan layanan ARETA di jendela terpisah...
start "ARETA - ImlaScan" cmd /k "areta_service\jalankan_areta.bat"
echo       Menunggu ARETA siap (maks 60 detik)...
set /a n=0
:tunggu
"venv311\Scripts\python.exe" -c "import sys; sys.path.insert(0,'areta_service'); import areta_client; sys.exit(0 if areta_client.is_up() else 1)" >nul 2>&1
if not errorlevel 1 goto siap
set /a n+=1
if %n% geq 30 goto lewat
timeout /t 2 /nobreak >nul
goto tunggu
:lewat
echo       PERINGATAN: ARETA belum merespons. ImlaScan tetap jalan memakai tagger heuristik.
goto web
:siap
echo       ARETA siap.
:web
echo [2/2] Menyalakan server web. Buka http://127.0.0.1:8000
"venv311\Scripts\python.exe" manage.py runserver
pause

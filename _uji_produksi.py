# -*- coding: utf-8 -*-
"""Cek konfigurasi produksi tanpa mengubah .env: jalankan Django dengan DJANGO_DEBUG=False
dan pastikan pengaman aktif. Jalankan: venv311\\Scripts\\python.exe -X utf8 _uji_produksi.py"""
import os, subprocess, sys
ROOT = os.path.dirname(os.path.abspath(__file__))
PY = sys.executable
def jalan(env_tambah, kode):
    env = dict(os.environ, DJANGO_SETTINGS_MODULE="gramatical.settings", PYTHONIOENCODING="utf-8", **env_tambah)
    r = subprocess.run([PY, "-X", "utf8", "-c", kode], cwd=ROOT, env=env, capture_output=True, text=True, encoding="utf-8")
    return r.returncode, (r.stdout + r.stderr).strip()
PROD = {"DJANGO_DEBUG": "False", "DJANGO_SECRET_KEY": "x" * 60 + "Ab1!", "DJANGO_ALLOWED_HOSTS": "imlascan.example.com",
        "IMLASCAN_BEHIND_PROXY": "True", "DJANGO_CSRF_TRUSTED_ORIGINS": "https://imlascan.example.com",
        "DJANGO_HSTS_SECONDS": "31536000", "DJANGO_ENABLE_ADMIN": "False"}
ok = True
def cek(nama, syarat, info=""):
    global ok; ok &= bool(syarat)
    print(("LULUS " if syarat else "GAGAL ") + nama + (" :: " + info if info else ""), flush=True)

kode, out = jalan({"DJANGO_DEBUG": "False", "DJANGO_SECRET_KEY": "django-insecure-contoh", "DJANGO_ALLOWED_HOSTS": "x.com"},
                  "import django; django.setup()")
cek("DEBUG=False dengan SECRET_KEY bawaan/tidak aman -> server menolak jalan", kode != 0 and "DJANGO_SECRET_KEY" in out)
kode, out = jalan(dict(PROD, DJANGO_ALLOWED_HOSTS=""), "import django; django.setup()")
cek("DEBUG=False tanpa ALLOWED_HOSTS -> server menolak jalan", kode != 0 and "DJANGO_ALLOWED_HOSTS" in out)
kode, out = jalan(PROD, "import django; django.setup(); from django.core.management import call_command; call_command('check', deploy=True)")
import re
ids = sorted(set(re.findall(r"security\.W\d+", out)))
# Yang boleh tersisa (keputusan sadar, bukan lubang): W008 redirect HTTP->HTTPS dikerjakan nginx;
# W005/W021 HSTS subdomain & preload sengaja tidak dinyalakan (sulit dibatalkan).
boleh = {"security.W005", "security.W008", "security.W021"}
cek("manage.py check --deploy: tidak ada peringatan di luar yang disengaja", kode == 0 and set(ids) <= boleh, "tersisa: " + (", ".join(ids) or "tidak ada"))
kode, out = jalan(PROD, """
import django; django.setup()
from django.test import Client
c = Client(HTTP_HOST='imlascan.example.com', HTTP_X_FORWARDED_PROTO='https')
r = c.get('/'); a = c.get('/admin/'); p = Client(HTTP_HOST='imlascan.example.com', HTTP_X_FORWARDED_PROTO='https', enforce_csrf_checks=True).post('/', {'teks_arab': 'x'})
b = Client(HTTP_HOST='jahat.example.org').get('/')
print(r.status_code, a.status_code, p.status_code, b.status_code, r.headers.get('X-Frame-Options'), r.headers.get('X-Content-Type-Options'), 'Secure' in str(r.cookies.get('csrftoken').output()) if r.cookies.get('csrftoken') else 'tanpa-cookie')
""")
cek("Produksi: beranda 200, /admin/ 404, POST tanpa CSRF 403, host asing 400, header aman", out.split("\n")[-1].startswith("200 404 403 400 DENY nosniff True"), out.split("\n")[-1])
print("\nRINGKASAN PRODUKSI:", "SEMUA LULUS" if ok else "ADA YANG GAGAL")

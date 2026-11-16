# -*- coding: utf-8 -*-
"""Uji paket zip: ekstrak ke folder sementara, lalu jalankan aplikasi DARI HASIL EKSTRAK
(tanpa .env, memakai model & tessdata di dalam paket). Syarat: layanan ARETA aktif.
Jalankan: venv311\\Scripts\\python.exe -X utf8 _uji_paket.py"""
import glob, hashlib, os, shutil, subprocess, sys, tempfile, time, zipfile
ROOT = os.path.dirname(os.path.abspath(__file__))
zips = sorted(glob.glob(os.path.join(ROOT, "_paket", "ImlaScan_siap_pasang_*.zip")))
assert zips, "zip belum dibuat (jalankan _buat_paket.py)"
ZIP = zips[-1]; ok = True
def cek(nama, syarat, info=""):
    global ok; ok &= bool(syarat)
    print(("LULUS " if syarat else "GAGAL ") + nama + (" :: " + info if info else ""), flush=True)
def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""): h.update(b)
    return h.hexdigest()
tmp = tempfile.mkdtemp(prefix="imlascan_paket_")
try:
    t0 = time.time()
    with zipfile.ZipFile(ZIP) as z: z.extractall(tmp)
    app = os.path.join(tmp, "ImlaScan")
    cek("Zip terekstrak utuh", os.path.isfile(os.path.join(app, "manage.py")), f"{os.path.basename(ZIP)}, {time.time()-t0:.0f} dtk")
    cek("Tidak ada .env, data siswa, atau hitungan kuota di dalam paket",
        not any(os.path.exists(os.path.join(app, x)) for x in (".env", "data_siap", "data_mentah", ".imlascan_kuota.json")))
    sama = all(sha(os.path.join(app, f)) == sha(os.path.join(ROOT, f)) for f in
               ("models/D_tibyan_v2/model.safetensors", "models/D_tibyan_v2/tokenizer.json", "tessdata/ara.traineddata"))
    cek("Model, tokenizer, dan tessdata identik dengan aslinya (SHA-256)", sama)
    r = subprocess.run(["git", "ls-files"], cwd=ROOT, capture_output=True, text=True, encoding="utf-8")
    beda = [f for f in r.stdout.split("\n") if f and (not os.path.isfile(os.path.join(app, f)) or
            open(os.path.join(app, f), "rb").read() != open(os.path.join(ROOT, f), "rb").read())]
    cek("Semua berkas kode di paket sama dengan versi terakhir di git", not beda, ("berbeda: " + ", ".join(beda[:5])) if beda else "")
    kode = r'''
import os, io, sys, json
sys.path.insert(0, os.getcwd())
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "gramatical.settings")
import django; django.setup()
from django.test.utils import setup_test_environment; setup_test_environment()
from django.conf import settings
settings.ALLOWED_HOSTS = list(settings.ALLOWED_HOSTS) + ["testserver"]
from django.test import Client
from home import gec_model
from PIL import Image
c = Client()
g = c.get("/"); h = g.content.decode("utf-8")
r = c.post("/", {"teks_arab": "ذهب احمد الي المدرسه فى مدينه جميله."})
ha = r.context["hasil_analisis"]
b = io.BytesIO(); Image.new("L", (600, 300), 255).save(b, format="PNG"); f = io.BytesIO(b.getvalue()); f.name = "kosong.png"
t = c.post("/", {"dokumen_arab": f, "ocr_engine": "tesseract"})
print(json.dumps({"form": g.status_code, "tombol": h.count('type="radio" name="ocr_engine"'), "kuota": "Sisa kuota hari ini: 30 halaman" in h,
    "tagger": ha["tagger"], "n": ha["total_errors"], "model": os.path.normcase(os.environ["IMLASCAN_MODEL_PATH"]) in os.path.normcase(str(gec_model._MODEL is not None and os.environ["IMLASCAN_MODEL_PATH"])),
    "tess": t.context.get("error_message", "")[:80]}, ensure_ascii=False))
'''
    env = {k: v for k, v in os.environ.items() if not k.startswith(("IMLASCAN_", "DJANGO_"))}
    env.update(IMLASCAN_MODEL_PATH=os.path.join(app, "models", "D_tibyan_v2"), PYTHONIOENCODING="utf-8",
               IMLASCAN_KUOTA_FILE=os.path.join(tmp, "kuota.json"))
    r = subprocess.run([sys.executable, "-X", "utf8", "-c", kode], cwd=app, env=env, capture_output=True, text=True, encoding="utf-8")
    baris = [l for l in r.stdout.split("\n") if l.startswith("{")]
    import json
    d = json.loads(baris[-1]) if baris else {}
    cek("Aplikasi dari hasil ekstrak menyala tanpa .env (form 200, 3 tombol OCR, kuota 30)", d.get("form") == 200 and d.get("tombol") == 3 and d.get("kuota"), str({k: d.get(k) for k in ("form", "tombol", "kuota")}) if d else r.stderr[-900:])
    cek("Model di dalam paket termuat dan mengoreksi (AraT5 + areta, 5 kesalahan)", d.get("tagger") == "AraT5 + areta" and d.get("n") == 5 and d.get("model"), f"{d.get('tagger')}, {d.get('n')} kesalahan")
    cek("Tesseract memakai tessdata di dalam paket", "tidak menghasilkan teks" in d.get("tess", ""), d.get("tess", ""))
finally:
    shutil.rmtree(tmp, ignore_errors=True)
print("\nRINGKASAN PAKET:", "SEMUA LULUS" if ok else "ADA YANG GAGAL")

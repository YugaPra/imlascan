# -*- coding: utf-8 -*-
"""Uji beban ImlaScan lewat HTTP sungguhan dalam mode produksi (DJANGO_DEBUG=False).
Menyalakan server sendiri di port 8001, lalu mengirim permintaan bersamaan.
Syarat: layanan ARETA aktif. Jalankan: venv311\Scripts\python.exe -X utf8 _uji_beban.py"""
import os, re, subprocess, sys, tempfile, threading, time, statistics
import requests
ROOT = os.path.dirname(os.path.abspath(__file__)); URL = "http://127.0.0.1:8001/"
IMG = os.path.join(ROOT, "data_siap", "v2", "verifikasi_kembali", "gambar", "P15_E02_H1.jpg")
TEKS = "ذهب احمد الي المدرسه فى مدينه جميله. و كانو الطلاب يدرسون اللغه العربيه في المدرسه كل يوم."
ok = True
def cek(nama, syarat, info=""):
    global ok; ok &= bool(syarat)
    print(("LULUS " if syarat else "GAGAL ") + nama + (" :: " + info if info else ""), flush=True)

def server(extra):
    env = dict(os.environ, DJANGO_DEBUG="False", DJANGO_ALLOWED_HOSTS="127.0.0.1", DJANGO_HTTPS="False",
               DJANGO_SECRET_KEY="uji-beban-" + "k" * 50, IMLASCAN_PRELOAD="True", PYTHONIOENCODING="utf-8",
               IMLASCAN_KUOTA_FILE=os.path.join(tempfile.mkdtemp(), "k.json"), **extra)
    log = open(os.path.join(ROOT, "_uji_beban_server.log"), "ab")
    p = subprocess.Popen([sys.executable, "-X", "utf8", "manage.py", "runserver", "--noreload", "127.0.0.1:8001"],
                         cwd=ROOT, env=env, stdout=log, stderr=log)
    for _ in range(120):
        try:
            if requests.get(URL, timeout=2).status_code == 200: break
        except requests.RequestException: time.sleep(0.5)
    return p

def matikan(p):
    try:
        import psutil
        for x in psutil.Process(p.pid).children(recursive=True): x.kill()
    except Exception:
        pass
    p.terminate(); p.wait(timeout=20)

def kirim(data=None, berkas=None):
    s = requests.Session(); g = s.get(URL, timeout=30)
    tok = re.search(r'name="csrfmiddlewaretoken" value="([^"]+)"', g.text).group(1)
    t0 = time.time()
    files = {"dokumen_arab": open(berkas, "rb")} if berkas else None
    r = s.post(URL, data={"csrfmiddlewaretoken": tok, **(data or {})}, files=files, timeout=300)
    m = re.search(r'Kesalahan Ditemukan:\s*<span[^>]*>(\d+)</span>', r.text)
    return r.status_code, (int(m.group(1)) if m else None), time.time() - t0, r.text

def bersamaan(n, **kw):
    out = [None] * n
    def kerja(i): out[i] = kirim(**kw)
    th = [threading.Thread(target=kerja, args=(i,)) for i in range(n)]
    t0 = time.time(); [t.start() for t in th]; [t.join() for t in th]
    return out, time.time() - t0

def rss(pid):
    try:
        import psutil                       # python.exe di venv Windows hanya peluncur -> jumlahkan anak prosesnya
        pr = psutil.Process(pid)
        return sum(x.memory_info().rss for x in [pr] + pr.children(recursive=True)) / 1e6
    except Exception:
        return None

# ---------- A: beban normal (pembatas laju dimatikan) ----------
p = server({"IMLASCAN_RATE_LIMIT": "", "IMLASCAN_RATE_LIMIT_OCR": ""})
try:
    kode, n0, dt0, html = kirim({"teks_arab": TEKS})            # pemanasan (model sudah dimuat di awal)
    cek("Mode produksi lewat HTTP: analisis teks jalan, mesin AraT5 + areta", kode == 200 and n0 and "AraT5 + areta" in html, f"{n0} kesalahan, {dt0:.1f} dtk")
    m0 = rss(p.pid)
    out, dt = bersamaan(16, data={"teks_arab": TEKS})
    cek("16 analisis teks bersamaan: semua 200 dan hasil identik", all(o[0] == 200 and o[1] == n0 for o in out),
        f"total {dt:.1f} dtk, paling lambat {max(o[2] for o in out):.1f} dtk")
    lat = [kirim({"teks_arab": TEKS})[2] for _ in range(40)]
    m1 = rss(p.pid)
    cek("40 analisis berurutan stabil", max(lat) < 10, f"median {statistics.median(lat):.2f} dtk, maks {max(lat):.2f} dtk")
    if m0 and m1:
        cek("Memori proses tidak membengkak setelah 56 analisis", m1 - m0 < 300, f"{m0:.0f} MB -> {m1:.0f} MB")
    else:
        print("LEWAT pengukuran memori (psutil tidak terpasang)")
    out, dt = bersamaan(6, data={"ocr_engine": "tesseract"}, berkas=IMG)
    cek("6 unggahan gambar bersamaan (Tesseract): semua 200 dan hasil identik",
        all(o[0] == 200 and o[1] is not None for o in out) and len({o[1] for o in out}) == 1, f"total {dt:.1f} dtk, {out[0][1]} kesalahan")
    out, dt = bersamaan(40)                                       # 40 GET + POST kosong
    cek("40 permintaan kosong bersamaan: semua 200 dengan pesan, tanpa error server", all(o[0] == 200 and "Mohon masukkan teks" in o[3] for o in out))
    r = requests.post(URL, data={"teks_arab": TEKS}, timeout=30)
    cek("POST tanpa token CSRF ditolak (403)", r.status_code == 403)
    r = requests.get(URL + "admin/", timeout=30); r2 = requests.get(URL + "tidak-ada", timeout=30)
    cek("/admin/ dan alamat tak dikenal -> 404 tanpa halaman debug", r.status_code == 404 and r2.status_code == 404 and "Traceback" not in r2.text and "DEBUG = True" not in r2.text)
finally:
    matikan(p)

# ---------- B: pembatas laju di bawah serbuan bersamaan ----------
p = server({"IMLASCAN_RATE_LIMIT": "5/600", "IMLASCAN_RATE_LIMIT_OCR": "5/600"})
try:
    out, dt = bersamaan(25, data={"teks_arab": TEKS})
    lolos = sum(o[1] is not None for o in out); tolak = sum("Terlalu banyak permintaan" in o[3] for o in out)
    cek("Pembatas laju 5/10 menit di bawah 25 permintaan bersamaan", lolos == 5 and tolak == 20, f"lolos {lolos}, ditolak {tolak}")
finally:
    matikan(p)
print("\nRINGKASAN BEBAN:", "SEMUA LULUS" if ok else "ADA YANG GAGAL")

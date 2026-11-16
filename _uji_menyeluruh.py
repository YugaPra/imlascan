# -*- coding: utf-8 -*-
"""Uji menyeluruh ImlaScan: pilihan OCR (Gemini/Tesseract/keduanya) + ARETA + jalur error.
Jalankan:  venv311\\Scripts\\python.exe -X utf8 _uji_menyeluruh.py
Syarat: layanan ARETA aktif (areta_service\\jalankan_areta.bat). Hanya membaca; menulis _uji_menyeluruh_*.html/.json
"""
import os, sys, io, json, time, traceback
ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "gramatical.settings")
from dotenv import load_dotenv; load_dotenv(os.path.join(ROOT, ".env"))
import tempfile
# Hitungan kuota uji disimpan di berkas sementara supaya tidak mengubah kuota aplikasi yang asli.
os.environ["IMLASCAN_KUOTA_FILE"] = os.path.join(tempfile.mkdtemp(), "kuota_uji.json")
import django; django.setup()
from django.test import Client
from django.test.utils import setup_test_environment; setup_test_environment()
from django.conf import settings
settings.ALLOWED_HOSTS = list(settings.ALLOWED_HOSTS) + ["testserver"]
settings.IMLASCAN_RATE_LIMIT = settings.IMLASCAN_RATE_LIMIT_OCR = ""   # dimatikan; diuji khusus di bawah
from home import ocr, gec_model, views, kuota
sys.path.insert(0, os.path.join(ROOT, "areta_service"))
import areta_client

IMG = os.path.join(ROOT, "data_siap", "v2", "verifikasi_kembali", "gambar", "P15_E02_H1.jpg")
PDF = os.path.join(ROOT, "sample.pdf")
c = Client()
HASIL = []

def uji(nama, fn):
    t0 = time.time()
    try:
        info = fn() or ""
        HASIL.append({"uji": nama, "lulus": True, "info": info, "dtk": round(time.time() - t0, 1)})
    except Exception as e:
        HASIL.append({"uji": nama, "lulus": False, "info": f"{type(e).__name__}: {e}",
                      "dtk": round(time.time() - t0, 1), "trace": traceback.format_exc()[-600:]})
    r = HASIL[-1]
    print(("LULUS " if r["lulus"] else "GAGAL ") + f"[{r['dtk']:>5} dtk] {nama} :: {r['info']}", flush=True)

def simpan(nama, resp):
    with open(os.path.join(ROOT, f"_uji_menyeluruh_{nama}.html"), "w", encoding="utf-8") as f:
        f.write(resp.content.decode("utf-8"))

def post_img(engine=None, path=IMG):
    data = {}
    if engine is not None:
        data["ocr_engine"] = engine
    with open(path, "rb") as f:
        data["dokumen_arab"] = f
        return c.post("/", data)

# 1 ARETA
def t_areta():
    assert areta_client.is_up(), "layanan ARETA tidak aktif di " + areta_client.ARETA_URL
    toks = areta_client.annotate([("ذهب احمد الي المدرسه", "ذهب أحمد إلى المدرسة")])[0]
    tags = {t["src"]: t["tag"] for t in toks}
    assert tags.get("احمد") == "OH", tags
    assert tags.get("المدرسه") == "OT", tags
    import urllib.request, urllib.error
    try:
        urllib.request.urlopen(areta_client.ARETA_URL + "/", timeout=5); halaman = True
    except urllib.error.HTTPError as e:
        halaman = e.code != 404
    assert not halaman, "halaman demo ARETA masih tersaji"
    return json.dumps(tags, ensure_ascii=False)
uji("ARETA aktif & memberi tag benar (OH, OT)", t_areta)

# 2 form
def t_form():
    r = c.get("/"); h = r.content.decode("utf-8")
    assert r.status_code == 200
    for v in ("gemini", "tesseract", "keduanya"):
        assert f'name="ocr_engine" value="{v}"' in h, v
    assert h.count('type="radio" name="ocr_engine"') == 3
    assert 'value="gemini" class="peer sr-only" checked' in h
    simpan("form", r)
    return "3 tombol, default Gemini"
uji("Form memuat 3 tombol pilihan OCR", t_form)

# 3 tesseract saja
def t_tess():
    r = post_img("tesseract"); ctx = r.context; ob = ctx["ocr_bandingan"]
    assert r.status_code == 200 and not ctx.get("error_message"), ctx.get("error_message")
    assert ob["tunggal"] and [k["engine"] for k in ob["kolom"]] == ["tesseract"]
    assert ob["dipakai"] == "Tesseract OCR"
    ha = ctx["hasil_analisis"]; assert ha["tagger"] == "AraT5 + areta", ha["tagger"]
    h = r.content.decode("utf-8")
    assert "Mesin OCR: Tesseract OCR" in h and "Perbandingan OCR" not in h and "Gemini OCR" not in h.split('id="hasil-analisis"')[1].split("Bagaimana ImlaScan")[0]
    simpan("tesseract", r)
    return f"{ob['kolom'][0]['words']} kata, {ob['kolom'][0]['seconds']} dtk, {ha['total_errors']} error, tagger {ha['tagger']}"
uji("Pilih Tesseract -> hanya Tesseract jalan, tag dari ARETA", t_tess)

# 4 gemini saja
def t_gem():
    r = post_img("gemini"); ctx = r.context
    assert not ctx.get("error_message"), ctx.get("error_message")
    ob = ctx["ocr_bandingan"]
    assert ob["tunggal"] and [k["engine"] for k in ob["kolom"]] == ["gemini"]
    ha = ctx["hasil_analisis"]; assert ha["tagger"] == "AraT5 + areta", ha["tagger"]
    simpan("gemini", r)
    return f"{ob['kolom'][0]['words']} kata, {ob['kolom'][0]['seconds']} dtk, {ha['total_errors']} error, tagger {ha['tagger']}"
uji("Pilih Gemini -> hanya Gemini jalan, tag dari ARETA", t_gem)

# 5 keduanya + analisis ulang
def t_dua():
    r = post_img("keduanya"); ctx = r.context
    assert not ctx.get("error_message"), ctx.get("error_message")
    ob = ctx["ocr_bandingan"]
    assert not ob["tunggal"] and [k["engine"] for k in ob["kolom"]] == ["gemini", "tesseract"]
    gem = ob["kolom"][0]
    if gem["error"]:      # Gemini (layanan luar) tidak merespons: yang diuji lalu adalah peralihannya
        assert ob["dipakai"] == "Tesseract OCR" and ctx.get("info_message") and "hasil_analisis" in ctx
        return f"CATATAN: Gemini tidak merespons ({gem['error'][:60]}; {gem['seconds']} dtk) -> otomatis Tesseract, panel tetap tampil"
    assert ob["kemiripan"] is not None
    simpan("keduanya", r)
    tes = [k for k in ob["kolom"] if k["engine"] == "tesseract"][0]
    r2 = c.post("/", {"teks_arab": tes["text"], "sumber_ocr": "tesseract", "ocr_json": ob["json"]})
    assert r2.context["ocr_bandingan"]["dipakai"] == "Tesseract OCR"
    assert not r2.context["ocr_bandingan"]["tunggal"]
    simpan("keduanya_ulang", r2)
    return f"kemiripan {ob['kemiripan']} %, dipakai {ob['dipakai']}; analisis ulang Tesseract: {r2.context['hasil_analisis']['total_errors']} error"
uji("Pilih keduanya -> panel pembanding + tombol analisis ulang", t_dua)

# 6-7 nilai default / tidak sah -> gemini (tanpa memanggil API: extract_compare diganti sementara)
def t_default():
    asli = ocr.extract_compare; tangkap = []
    def palsu(b, n, engines=ocr.COMPARE_ENGINES):
        tangkap.append(tuple(engines))
        return [{"engine": e, "label": ocr.ENGINE_LABELS[e], "text": "ذهب احمد الي المدرسه", "error": "", "seconds": 0.0} for e in engines]
    ocr.extract_compare = palsu
    try:
        post_img(None); post_img("ngawur"); post_img("KEDUANYA"); post_img(" Tesseract ")
    finally:
        ocr.extract_compare = asli
    assert tangkap == [("gemini",), ("gemini",), ("gemini", "tesseract"), ("tesseract",)], tangkap
    return str(tangkap)
uji("Tanpa pilihan / nilai tak sah -> Gemini; huruf besar & spasi ditoleransi", t_default)

# 8 teks tempel
def t_teks():
    r = c.post("/", {"teks_arab": "ذهب احمد الي المدرسه فى مدينه جميله."}); ctx = r.context
    ha = ctx["hasil_analisis"]
    assert "ocr_bandingan" not in ctx
    assert ha["tagger"] == "AraT5 + areta", ha["tagger"]
    assert ha["total_errors"] >= 1
    kode = [e["error_type"].split("·")[0].strip() for e in ha["errors"]]
    for k in kode:
        assert all(p in views.SCOPE_TAGS for p in k.split("+")), k
    simpan("teks", r)
    return f"{ha['total_errors']} error: " + ", ".join(f"{e['word']}→{e['suggestion']} [{k}]" for e, k in zip(ha["errors"], kode))
uji("Tempel teks -> AraT5 + ARETA, semua tag di dalam cakupan", t_teks)

# 9 .txt
def t_txt():
    f = io.BytesIO("ذهب احمد الي المدرسه.".encode("utf-8")); f.name = "contoh.txt"
    r = c.post("/", {"dokumen_arab": f, "ocr_engine": "gemini"})
    assert "ocr_bandingan" not in r.context and r.context["hasil_analisis"]["extracted_text"].startswith("ذهب")
    return f"{r.context['hasil_analisis']['total_errors']} error, tanpa OCR"
uji("Unggah .txt -> tanpa OCR", t_txt)

# 10 PDF + tesseract
def t_pdf():
    r = post_img("tesseract", PDF); ctx = r.context
    assert not ctx.get("error_message"), ctx.get("error_message")
    k = ctx["ocr_bandingan"]["kolom"][0]
    return f"{k['words']} kata, {k['seconds']} dtk, {ctx['hasil_analisis']['total_errors']} error"
uji("Unggah PDF (sample.pdf) dengan Tesseract", t_pdf)

# 11-12 jalur error
def t_kosong():
    r = c.post("/", {"teks_arab": "   "})
    assert "Mohon masukkan teks" in r.context["error_message"] and "hasil_analisis" not in r.context
uji("Kirim kosong -> pesan error, tanpa hasil", t_kosong)

def t_format():
    f = io.BytesIO(b"abc"); f.name = "esai.docx"
    r = c.post("/", {"dokumen_arab": f})
    assert "Format tidak didukung" in r.context["error_message"]
uji("Format tak didukung (.docx) -> pesan error", t_format)

def t_rusak():
    f = io.BytesIO(b"bukan gambar"); f.name = "rusak.png"
    r = c.post("/", {"dokumen_arab": f, "ocr_engine": "tesseract"})
    assert r.status_code == 200 and r.context.get("error_message"), "tidak ada pesan error"
    return r.context["error_message"][:90]
uji("Gambar rusak -> pesan error, bukan crash", t_rusak)

# 13 mesin belum dikonfigurasi
def t_tak_konfig():
    lama = os.environ.get("IMLASCAN_TESSERACT_CMD")
    os.environ["IMLASCAN_TESSERACT_CMD"] = r"C:\tidak_ada\tesseract.exe"
    try:
        r = post_img("tesseract")
        assert "Tesseract OCR" in r.context["error_message"] and "belum dikonfigurasi" in r.context["error_message"]
    finally:
        if lama is None: os.environ.pop("IMLASCAN_TESSERACT_CMD")
        else: os.environ["IMLASCAN_TESSERACT_CMD"] = lama
    return r.context["error_message"]
uji("Tesseract tak terpasang -> pesan jelas", t_tak_konfig)

def t_gemini_mati():
    lama = os.environ.get("IMLASCAN_GEMINI_KEY", "")
    os.environ["IMLASCAN_GEMINI_KEY"] = ""
    try:
        r1 = post_img("gemini")
        assert "Gemini OCR" in r1.context["error_message"]
        r2 = post_img("keduanya"); ob = r2.context["ocr_bandingan"]
        assert ob["dipakai"] == "Tesseract OCR" and ob["kemiripan"] is None
    finally:
        os.environ["IMLASCAN_GEMINI_KEY"] = lama
    return "Gemini saja: pesan error; keduanya: otomatis pakai Tesseract"
uji("Key Gemini kosong -> pesan jelas / fallback ke Tesseract", t_gemini_mati)

# 14 XSS
def t_xss():
    r = c.post("/", {"teks_arab": '<script>alert(1)</script> <img src=x onerror=alert(2)> ذهب احمد الي المدرسه'})
    h = r.content.decode("utf-8")
    assert "<script>alert(1)</script>" not in h and "<img src=x" not in h
    return "masukan HTML di-escape"
uji("Keamanan: masukan HTML/script tidak dieksekusi", t_xss)

# 15 ARETA mati -> heuristik
def t_fallback():
    lama = areta_client.ARETA_URL
    areta_client.ARETA_URL = "http://127.0.0.1:9"
    try:
        r = c.post("/", {"teks_arab": "ذهب احمد الي المدرسه فى مدينه جميله."})
        tg = r.context["hasil_analisis"]["tagger"]
        assert tg == "AraT5 + heuristic", tg
    finally:
        areta_client.ARETA_URL = lama
    r = c.post("/", {"teks_arab": "ذهب احمد الي المدرسه."})
    assert r.context["hasil_analisis"]["tagger"] == "AraT5 + areta"
    return "ARETA mati -> heuristic; hidup lagi -> areta"
uji("ARETA mati -> otomatis ke tagger heuristik (tidak crash)", t_fallback)

# 16 ARETA per kalimat: teks banyak kalimat tetap bertag & cepat
def t_per_kalimat():
    teks = " ".join(["ذهب احمد الي المدرسه فى مدينه جميله."] * 30)
    t0 = time.time(); r = c.post("/", {"teks_arab": teks}); dt = time.time() - t0
    ha = r.context["hasil_analisis"]
    assert ha["tagger"] == "AraT5 + areta", ha["tagger"]
    assert ha["total_errors"] >= 30 * 4, ha["total_errors"]
    return f"30 kalimat, {ha['total_errors']} error, {dt:.1f} dtk"
uji("ARETA per kalimat: 30 kalimat bertag konsisten", t_per_kalimat)

# 17-19 batas ukuran
def t_besar():
    lama = settings.IMLASCAN_MAX_UPLOAD_MB; settings.IMLASCAN_MAX_UPLOAD_MB = 0.05
    try:
        r = post_img("tesseract")
    finally:
        settings.IMLASCAN_MAX_UPLOAD_MB = lama
    assert "terlalu besar" in r.context["error_message"] and "hasil_analisis" not in r.context
    return r.context["error_message"]
uji("Berkas melebihi batas ukuran -> ditolak sebelum OCR", t_besar)

def t_halaman():
    lama = os.environ.get("IMLASCAN_MAX_PAGES"); os.environ["IMLASCAN_MAX_PAGES"] = "0"
    try:
        r = post_img("tesseract", PDF)
    finally:
        if lama is None: os.environ.pop("IMLASCAN_MAX_PAGES")
        else: os.environ["IMLASCAN_MAX_PAGES"] = lama
    assert "halaman" in r.context["error_message"] and "hasil_analisis" not in r.context
    return r.context["error_message"]
uji("PDF melebihi batas halaman -> ditolak", t_halaman)

def t_panjang():
    r = c.post("/", {"teks_arab": "ذهب احمد الي المدرسه. " * 1200})
    assert "terlalu panjang" in r.context["error_message"] and "hasil_analisis" not in r.context
    return r.context["error_message"]
uji("Teks melebihi batas karakter -> ditolak sebelum model", t_panjang)

# 20 pembatas laju
def t_laju():
    from django.core.cache import cache; cache.clear()
    settings.IMLASCAN_RATE_LIMIT = "3/600"
    try:
        kode = [bool(c.post("/", {"teks_arab": "ذهب احمد."}, REMOTE_ADDR="10.9.9.9").context.get("error_message")) for _ in range(5)]
        lain = c.post("/", {"teks_arab": "ذهب احمد."}, REMOTE_ADDR="10.9.9.10").context.get("error_message")
        pesan = c.post("/", {"teks_arab": "ذهب احمد."}, REMOTE_ADDR="10.9.9.9").context["error_message"]
    finally:
        settings.IMLASCAN_RATE_LIMIT = ""; cache.clear()
    assert kode == [False, False, False, True, True], kode
    assert not lain, "IP lain ikut terblokir"
    return pesan
uji("Pembatas laju: permintaan ke-4 dari IP yang sama ditolak, IP lain tidak", t_laju)

def t_laju_ocr():
    from django.core.cache import cache; cache.clear()
    settings.IMLASCAN_RATE_LIMIT_OCR = "1/600"
    asli = ocr.extract_compare; n = []
    def palsu(b, nm, engines=ocr.COMPARE_ENGINES):
        n.append(1)
        return [{"engine": e, "label": ocr.ENGINE_LABELS[e], "text": "ذهب احمد", "error": "", "seconds": 0.0} for e in engines]
    ocr.extract_compare = palsu
    try:
        r1 = post_img("tesseract"); r2 = post_img("tesseract")
        r3 = c.post("/", {"teks_arab": "ذهب احمد."})
    finally:
        ocr.extract_compare = asli; settings.IMLASCAN_RATE_LIMIT_OCR = ""; cache.clear()
    assert not r1.context.get("error_message") and "Terlalu banyak" in r2.context["error_message"] and len(n) == 1
    assert not r3.context.get("error_message"), "teks tempel ikut terblokir oleh batas OCR"
    return "OCR ke-2 ditolak tanpa memanggil mesin OCR; teks tempel tetap bisa"
uji("Pembatas laju OCR terpisah dari teks tempel", t_laju_ocr)

# 21 API key tidak bocor
def t_kunci():
    import requests
    key = os.environ.get("IMLASCAN_GEMINI_KEY", "")
    assert key, "key Gemini kosong"
    asli = requests.post; terekam = {}
    def palsu(url, **kw):
        terekam["url"] = url; terekam["headers"] = kw.get("headers", {})
        raise requests.exceptions.ConnectionError(f"Max retries exceeded with url: {url}")
    requests.post = palsu
    try:
        r = post_img("gemini")
    finally:
        requests.post = asli
    h = r.content.decode("utf-8")
    assert key not in terekam["url"] and terekam["headers"].get("x-goog-api-key") == key
    assert key not in h and key not in (r.context.get("error_message") or "")
    assert ocr._aman("abc " + key) == "abc ***"
    return "key dikirim lewat header; tidak muncul di URL maupun halaman error"
uji("Keamanan: API key Gemini tidak bocor ke URL / pesan error", t_kunci)

# 22-26 kuota global Gemini
def _reset_kuota(maks):
    os.environ["IMLASCAN_GEMINI_DAILY_MAX"] = str(maks)
    if os.path.exists(os.environ["IMLASCAN_KUOTA_FILE"]):
        os.replace(os.environ["IMLASCAN_KUOTA_FILE"], os.environ["IMLASCAN_KUOTA_FILE"] + ".lama")
    kuota.lupa()

def _gemini_palsu():
    asli = ocr._gemini_ocr; n = []
    def palsu(b, mime="image/png"):
        n.append(1); return "ذهب احمد الي المدرسه"
    ocr._gemini_ocr = palsu
    return asli, n

def t_kuota_habis():
    _reset_kuota(2); asli, n = _gemini_palsu()
    try:
        r1 = post_img("gemini"); r2 = post_img("gemini"); r3 = post_img("gemini")
        form = c.get("/").content.decode("utf-8")
    finally:
        ocr._gemini_ocr = asli; _reset_kuota(125)
    for r in (r1, r2):
        assert r.context["ocr_bandingan"]["dipakai"] == "Gemini OCR" and not r.context.get("info_message")
    ob = r3.context["ocr_bandingan"]
    assert len(n) == 2, f"Gemini terpanggil {len(n)}x"
    assert ob["dipakai"] == "Tesseract OCR" and ob["tunggal"] and "hasil_analisis" in r3.context
    assert "Kuota harian Gemini OCR (2 halaman" in r3.context["info_message"]
    assert "Kuota harian Gemini OCR" in r3.content.decode("utf-8")
    assert "Kuota hari ini habis" in form
    return "2 halaman pertama Gemini; ke-3 otomatis Tesseract + pemberitahuan; form menampilkan kuota habis"
uji("Kuota Gemini habis -> otomatis Tesseract dengan pemberitahuan", t_kuota_habis)

def t_kuota_keduanya():
    _reset_kuota(1); asli, n = _gemini_palsu()
    try:
        post_img("gemini"); r = post_img("keduanya")
    finally:
        ocr._gemini_ocr = asli; _reset_kuota(125)
    ob = r.context["ocr_bandingan"]; g = [k for k in ob["kolom"] if k["engine"] == "gemini"][0]
    assert len(n) == 1 and ob["dipakai"] == "Tesseract OCR" and "kuota" in g["error"] and r.context.get("info_message")
    return "kolom Gemini: " + g["error"]
uji("Kuota habis pada mode 'keduanya' -> Tesseract dipakai, kolom Gemini menjelaskan", t_kuota_keduanya)

def t_kuota_pdf():
    _reset_kuota(1); asli, n = _gemini_palsu()
    try:
        r = post_img("gemini", PDF); st = kuota.status()
    finally:
        ocr._gemini_ocr = asli; _reset_kuota(125)
    assert len(n) == 0 and st["terpakai"] == 0 and r.context["ocr_bandingan"]["dipakai"] == "Tesseract OCR"
    return "PDF 2 halaman, sisa 1 -> tidak dikirim sebagian; kuota tidak berkurang"
uji("PDF lebih banyak halaman daripada sisa kuota -> seluruhnya ke Tesseract", t_kuota_pdf)

def t_kuota_reset():
    _reset_kuota(1); asli, n = _gemini_palsu()
    try:
        post_img("gemini"); assert kuota.status()["sisa"] == 0
        d = json.load(open(os.environ["IMLASCAN_KUOTA_FILE"], encoding="utf-8"))
        d["hari"] = "2000-01-01"
        json.dump(d, open(os.environ["IMLASCAN_KUOTA_FILE"], "w", encoding="utf-8"))
        kuota.lupa(); st = kuota.status(); r = post_img("gemini")
    finally:
        ocr._gemini_ocr = asli; _reset_kuota(125)
    assert st["sisa"] == 1 and st["terpakai_bulan"] == 1 and r.context["ocr_bandingan"]["dipakai"] == "Gemini OCR"
    return "hari berganti -> kuota harian penuh lagi, hitungan bulanan tetap"
uji("Kuota harian kembali penuh saat hari berganti", t_kuota_reset)

def t_token():
    import requests
    _reset_kuota(125); asli = requests.post
    class R:
        status_code = 200
        def json(self):
            return {"candidates": [{"content": {"parts": [{"text": "ذهب احمد الي المدرسه"}]}}],
                    "usageMetadata": {"promptTokenCount": 1200, "candidatesTokenCount": 300, "thoughtsTokenCount": 900}}
    requests.post = lambda url, **kw: R()
    try:
        post_img("gemini"); post_img("gemini")
        d = json.load(open(os.environ["IMLASCAN_KUOTA_FILE"], encoding="utf-8"))["hari_ini"]
    finally:
        requests.post = asli; _reset_kuota(125)
    assert d == {"halaman": 2, "berhasil": 2, "token_masukan": 2400, "token_keluaran": 600, "token_berpikir": 1800}, d
    return str(d)
uji("Pemakaian token Gemini tercatat (masukan / keluaran / berpikir)", t_token)

# 27-34 ketahanan (durability) untuk pemakaian nyata
def t_kuota_thread():
    import threading
    _reset_kuota(50); ok = []; gagal = []
    def kerja():
        try: kuota.ambil(1); ok.append(1)
        except kuota.KuotaHabis: gagal.append(1)
    th = [threading.Thread(target=kerja) for _ in range(200)]
    [t.start() for t in th]; [t.join() for t in th]
    st = kuota.status(); _reset_kuota(50)
    assert len(ok) == 50 and len(gagal) == 150 and st["sisa"] == 0, (len(ok), len(gagal), st)
    return "200 permintaan bersamaan, batas 50 -> tepat 50 lolos"
uji("Kuota: 200 permintaan bersamaan tidak bisa melewati batas", t_kuota_thread)

def t_kuota_gagal_tulis():
    lama = os.environ["IMLASCAN_KUOTA_FILE"]
    os.environ["IMLASCAN_KUOTA_FILE"] = os.path.join(tempfile.gettempdir(), "folder_tidak_ada_xyz", "k.json")
    os.environ["IMLASCAN_GEMINI_DAILY_MAX"] = "2"; kuota.lupa()
    try:
        kuota.ambil(1); kuota.ambil(1)
        try: kuota.ambil(1); lolos = True
        except kuota.KuotaHabis: lolos = False
    finally:
        os.environ["IMLASCAN_KUOTA_FILE"] = lama; _reset_kuota(50)
    assert not lolos, "batas tidak ditegakkan saat berkas gagal ditulis"
    return "berkas kuota tak bisa ditulis -> batas tetap ditegakkan dari memori"
uji("Kuota tetap ditegakkan walau berkas penyimpanan gagal ditulis", t_kuota_gagal_tulis)

def _png(w, h):
    from PIL import Image
    b = io.BytesIO(); Image.new("L", (w, h), 255).save(b, format="PNG"); return b.getvalue()

def t_bom_piksel():
    f = io.BytesIO(_png(9000, 9000)); f.name = "besar.png"; ukuran = len(f.getvalue())
    t0 = time.time(); r = c.post("/", {"dokumen_arab": f, "ocr_engine": "tesseract"}); dt = time.time() - t0
    assert "terlalu besar" in r.context["error_message"] and "hasil_analisis" not in r.context and dt < 5
    return f"berkas {ukuran/1024:.0f} KB berisi 81 megapiksel ditolak dalam {dt:.1f} dtk"
uji("Gambar berpiksel raksasa (berkas kecil) ditolak sebelum diproses", t_bom_piksel)

def t_bmp():
    from PIL import Image
    b = io.BytesIO(); Image.new("RGB", (300, 120), "white").save(b, format="BMP"); f = io.BytesIO(b.getvalue()); f.name = "foto.bmp"
    _reset_kuota(50); asli = ocr._gemini_ocr; terima = {}
    def palsu(bts, mime="image/png"):
        terima["mime"] = mime; terima["png"] = bts[:8] == b"\x89PNG\r\n\x1a\n"; return "ذهب احمد الي المدرسه"
    ocr._gemini_ocr = palsu
    try:
        r = c.post("/", {"dokumen_arab": f, "ocr_engine": "gemini"})
    finally:
        ocr._gemini_ocr = asli; _reset_kuota(50)
    assert terima == {"mime": "image/png", "png": True} and "hasil_analisis" in r.context, terima
    return "BMP dikonversi ke PNG sebelum dikirim ke Gemini"
uji("Format yang tidak didukung Gemini (BMP) dikonversi otomatis", t_bmp)

def t_gemini_gagal():
    _reset_kuota(50); asli = ocr._gemini_ocr
    def palsu(b, mime="image/png"): raise RuntimeError("Gemini HTTP 503.")
    ocr._gemini_ocr = palsu
    try:
        r = post_img("gemini")
    finally:
        ocr._gemini_ocr = asli; _reset_kuota(50)
    assert r.context["ocr_bandingan"]["dipakai"] == "Tesseract OCR" and "hasil_analisis" in r.context
    assert "tidak dapat memproses" in r.context["info_message"]
    return "Gemini error -> otomatis Tesseract + pemberitahuan"
uji("Gemini gagal (mode Gemini saja) -> otomatis Tesseract, bukan halaman error", t_gemini_gagal)

def t_batas_waktu():
    _reset_kuota(50); asli = ocr._gemini_ocr; os.environ["IMLASCAN_OCR_DEADLINE"] = "1"
    def palsu(b, mime="image/png"): time.sleep(6); return "terlambat"
    ocr._gemini_ocr = palsu
    try:
        t0 = time.time(); r = post_img("gemini"); dt = time.time() - t0
    finally:
        ocr._gemini_ocr = asli; os.environ.pop("IMLASCAN_OCR_DEADLINE"); _reset_kuota(50)
    assert dt < 5.5, f"permintaan menggantung {dt:.1f} dtk"
    assert r.context["ocr_bandingan"]["dipakai"] == "Tesseract OCR" and r.context.get("info_message")
    return f"Gemini macet 6 dtk, batas 1 dtk -> pengguna dilayani Tesseract dalam {dt:.1f} dtk"
uji("Gemini macet -> permintaan tidak menggantung (batas waktu OCR)", t_batas_waktu)

def t_paralel():
    _reset_kuota(50); asli = ocr._gemini_ocr
    hal = ocr._to_images(open(PDF, "rb").read(), "sample.pdf"); peta = {hash(b): i for i, (b, m) in enumerate(hal)}
    def palsu(b, mime="image/png"): time.sleep(1.5); return f"halaman{peta[hash(b)]} ذهب احمد."
    ocr._gemini_ocr = palsu
    try:
        t0 = time.time(); r = ocr.extract_compare(open(PDF, "rb").read(), "sample.pdf", engines=("gemini",)); dt = time.time() - t0
    finally:
        ocr._gemini_ocr = asli; _reset_kuota(50)
    urut = [w for w in r[0]["text"].split() if w.startswith("halaman")]
    assert urut == [f"halaman{i}" for i in range(len(hal))], urut
    assert dt < 1.5 * len(hal) - 0.4, f"{dt:.1f} dtk (tidak paralel)"
    return f"{len(hal)} halaman @1,5 dtk selesai {dt:.1f} dtk, urutan halaman benar"
uji("Halaman PDF dikirim ke Gemini bersamaan, urutan tetap", t_paralel)

def t_pdf_rusak():
    f = io.BytesIO(b"%PDF-1.4 ini bukan pdf sungguhan"); f.name = "rusak.pdf"
    r = c.post("/", {"dokumen_arab": f, "ocr_engine": "tesseract"})
    assert r.status_code == 200 and r.context.get("error_message") and "hasil_analisis" not in r.context
    return r.context["error_message"]
uji("PDF rusak -> pesan jelas, bukan crash", t_pdf_rusak)

lulus = sum(r["lulus"] for r in HASIL)
print(f"\nRINGKASAN: {lulus}/{len(HASIL)} lulus", flush=True)
with open(os.path.join(ROOT, "_uji_menyeluruh_hasil.json"), "w", encoding="utf-8") as f:
    json.dump(HASIL, f, ensure_ascii=False, indent=1)

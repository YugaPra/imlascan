# -*- coding: utf-8 -*-
"""
home/ocr.py — OCR Arab untuk ImlaScan. Tiga provider:
    - "gemini" (default): Google AI Studio / Gemini API (multimodal LLM).
    - "tesseract"       : Tesseract OCR lokal (open-source, OCR murni) — PEMBANDING.
    - "vision"          : Google Cloud Vision (OCR murni).

Mode pembanding (extract_compare): satu berkas di-OCR oleh Gemini DAN Tesseract
sekaligus (paralel), hasil ditampilkan berdampingan di web.

Alur:
    PDF  -> render tiap halaman jadi PNG (PyMuPDF) -> OCR -> teks
    gambar (png/jpg/...) -> langsung OCR -> teks

Konfigurasi (.env):
    IMLASCAN_OCR_PROVIDER = gemini | vision          (default: gemini)
    IMLASCAN_GEMINI_KEY   = API key AI Studio (AIza...)   [provider gemini]
    IMLASCAN_GEMINI_MODEL = model, default gemini-2.0-flash
    IMLASCAN_GVISION_KEY  = API key Cloud Vision (AIza...) [provider vision]
    IMLASCAN_OCR_DPI      = resolusi render PDF (default 200)

CATATAN RISET (penting): Gemini adalah LLM yang cenderung "mengoreksi" ejaan
saat membaca. Untuk GEC kita WAJIB mempertahankan error asli, maka prompt di
bawah memerintahkan transkrip HARFIAH tanpa koreksi. Walau begitu, hasil Gemini
tetap perlu diperlakukan eksperimental. Vision (OCR murni) lebih aman untuk
evaluasi karena tak pernah mengoreksi.
"""

import os
import base64
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeout

from . import kuota

_BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ENGINE_LABELS = {"gemini": "Gemini OCR", "tesseract": "Tesseract OCR", "vision": "Google Vision"}
COMPARE_ENGINES = ("gemini", "tesseract")

VISION_URL = "https://vision.googleapis.com/v1/images:annotate"
GEMINI_URL = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff", ".gif")

# Prompt transkripsi harfiah (jangan koreksi) — kunci agar error imla' tetap utuh.
_GEMINI_PROMPT = (
    "You are a strict OCR engine. Transcribe the Arabic text in this image "
    "EXACTLY as written, character by character, INCLUDING any spelling/imla "
    "mistakes. Do NOT correct, normalize, or complete anything. Do NOT add "
    "diacritics (tashkil) that are not present. Preserve line breaks. Output "
    "ONLY the raw transcribed text, with no explanation and no markdown."
)


class TooManyPages(ValueError):
    """PDF melebihi IMLASCAN_MAX_PAGES (pesannya aman ditampilkan ke pengguna)."""


class BadImage(ValueError):
    """Berkas tidak bisa dibuka / terlalu besar (pesannya aman ditampilkan ke pengguna)."""


def _aman(pesan):
    """Buang API key dari pesan error sebelum ditampilkan / dicatat."""
    pesan = str(pesan)
    for k in (_gemini_key(), _vision_key()):
        if k:
            pesan = pesan.replace(k, "***")
    return pesan


def _provider():
    return os.environ.get("IMLASCAN_OCR_PROVIDER", "gemini").strip().lower()


def _gemini_key():
    return os.environ.get("IMLASCAN_GEMINI_KEY", "").strip()


def _vision_key():
    return os.environ.get("IMLASCAN_GVISION_KEY", "").strip()


def is_configured():
    """True bila key untuk provider terpilih sudah di-set."""
    return bool(_gemini_key()) if _provider() == "gemini" else bool(_vision_key())


# ---------------------------------------------------------------------------
# Provider: Gemini (AI Studio)
# ---------------------------------------------------------------------------
def _gemini_ocr(img_bytes, mime="image/png"):
    import requests
    import time
    key = _gemini_key()
    model = os.environ.get("IMLASCAN_GEMINI_MODEL", "gemini-flash-latest").strip()
    url = GEMINI_URL.format(model=model)
    payload = {
        "contents": [{
            "parts": [
                {"text": _GEMINI_PROMPT},
                {"inline_data": {"mime_type": mime,
                                 "data": base64.b64encode(img_bytes).decode("ascii")}},
            ]
        }],
        "generationConfig": {"temperature": 0},
    }
    # Auto-retry untuk error sementara (429/500/502/503) dengan backoff.
    # Key dikirim lewat HEADER, bukan ?key= di URL: pesan error `requests` memuat URL lengkap,
    # sehingga key di URL bisa ikut tampil ke pengguna / tercatat di log.
    timeout = float(os.environ.get("IMLASCAN_GEMINI_TIMEOUT", "100"))
    tries = max(1, int(os.environ.get("IMLASCAN_GEMINI_TRIES", "3")))
    r = None
    for attempt in range(tries):
        r = requests.post(url, headers={"x-goog-api-key": key}, json=payload, timeout=timeout)
        if r.status_code == 200 or r.status_code not in (429, 500, 502, 503):
            break
        if attempt < tries - 1:
            time.sleep(2 * (attempt + 1))  # 2s, 4s
    if r.status_code != 200:
        detail = ""
        try:
            detail = r.json().get("error", {}).get("message", "")
        except Exception:
            pass
        hint = {
            400: "Permintaan/format tidak valid.",
            401: "API key tidak valid — key AI Studio diawali 'AIza'.",
            403: "Key ditolak / API belum aktif untuk project ini.",
            404: f"Model '{model}' tidak ditemukan — coba ganti IMLASCAN_GEMINI_MODEL.",
            429: "Kuota/rate limit terlampaui.",
        }.get(r.status_code, "")
        raise RuntimeError(f"Gemini HTTP {r.status_code}. {hint} {detail}".strip())
    data = r.json()
    # Catat pemakaian token riil (dasar hitung biaya): masukan, keluaran, dan "berpikir".
    um = data.get("usageMetadata") or {}
    kuota.catat_token(um.get("promptTokenCount"), um.get("candidatesTokenCount"),
                      um.get("thoughtsTokenCount"))
    try:
        parts = data["candidates"][0]["content"]["parts"]
        return "".join(p.get("text", "") for p in parts).strip()
    except (KeyError, IndexError):
        # bisa terjadi bila konten diblokir safety filter
        reason = data.get("candidates", [{}])[0].get("finishReason", "")
        raise RuntimeError(f"Gemini tak mengembalikan teks (finishReason={reason}).")


# ---------------------------------------------------------------------------
# Provider: Google Cloud Vision
# ---------------------------------------------------------------------------
def _vision_ocr(img_bytes):
    import requests
    key = _vision_key()
    payload = {
        "requests": [{
            "image": {"content": base64.b64encode(img_bytes).decode("ascii")},
            "features": [{"type": "DOCUMENT_TEXT_DETECTION"}],
            "imageContext": {"languageHints": ["ar"]},
        }]
    }
    r = requests.post(VISION_URL, headers={"x-goog-api-key": key}, json=payload, timeout=90)
    if r.status_code != 200:
        detail = ""
        try:
            detail = r.json().get("error", {}).get("message", "")
        except Exception:
            pass
        hint = {
            401: "API key tidak valid — key Google Cloud diawali 'AIza'.",
            403: "Cloud Vision API belum di-enable, billing mati, atau key dibatasi.",
        }.get(r.status_code, "")
        raise RuntimeError(f"Google Vision HTTP {r.status_code}. {hint} {detail}".strip())
    resp = r.json()["responses"][0]
    if resp.get("error"):
        raise RuntimeError(resp["error"].get("message", "Google Vision error"))
    fta = resp.get("fullTextAnnotation")
    return (fta["text"].strip() if fta else "")


# ---------------------------------------------------------------------------
# Provider: Tesseract (lokal, open-source)
# ---------------------------------------------------------------------------
def _tesseract_cmd():
    return os.environ.get("IMLASCAN_TESSERACT_CMD",
                          r"C:\Program Files\Tesseract-OCR\tesseract.exe").strip()


def _tessdata_dir():
    return os.environ.get("IMLASCAN_TESSDATA_DIR",
                          os.path.join(_BASE_DIR, "tessdata")).strip()


def _tesseract_lang():
    return os.environ.get("IMLASCAN_TESSERACT_LANG", "ara").strip()


def tesseract_configured():
    """True bila tesseract.exe dan berkas <lang>.traineddata ditemukan."""
    try:
        import pytesseract  # noqa: F401
    except ImportError:
        return False
    data = os.path.join(_tessdata_dir(), f"{_tesseract_lang()}.traineddata")
    return os.path.isfile(_tesseract_cmd()) and os.path.isfile(data)


def _tesseract_ocr(img_bytes):
    import io
    import pytesseract
    from PIL import Image
    if not tesseract_configured():
        raise RuntimeError(
            "Tesseract belum siap: pastikan tesseract.exe terpasang dan "
            f"{_tesseract_lang()}.traineddata ada di {_tessdata_dir()}.")
    pytesseract.pytesseract.tesseract_cmd = _tesseract_cmd()
    psm = os.environ.get("IMLASCAN_TESSERACT_PSM", "6").strip()
    img = Image.open(io.BytesIO(img_bytes)).convert("L")   # grayscale
    # Folder model lewat env (argumen --tessdata-dir ber-kutip gagal di Windows)
    os.environ["TESSDATA_PREFIX"] = _tessdata_dir()
    cfg = f"--psm {psm}"
    return pytesseract.image_to_string(img, lang=_tesseract_lang(), config=cfg).strip()


def is_engine_configured(engine):
    if engine == "gemini":
        return bool(_gemini_key())
    if engine == "vision":
        return bool(_vision_key())
    if engine == "tesseract":
        return tesseract_configured()
    return False


def _ocr_image_bytes(img_bytes, mime="image/png", engine=None):
    """Dispatch ke provider (engine eksplisit, atau IMLASCAN_OCR_PROVIDER)."""
    engine = (engine or _provider()).lower()
    if engine == "gemini":
        return _gemini_ocr(img_bytes, mime=mime)
    if engine == "tesseract":
        return _tesseract_ocr(img_bytes)
    return _vision_ocr(img_bytes)


# Gemini hanya menerima PNG/JPEG/WEBP (HEIC/HEIF); format lain dikonversi ke PNG lebih dulu.
_MIME_GEMINI = {"jpg": "image/jpeg", "jpeg": "image/jpeg", "png": "image/png", "webp": "image/webp"}


def _max_pixels():
    return int(float(os.environ.get("IMLASCAN_MAX_MEGAPIXELS", "40")) * 1_000_000)


def _siapkan_gambar(file_bytes, ext):
    """Validasi gambar unggahan -> (bytes, mime). Menolak berkas yang bukan gambar dan gambar
    berukuran piksel sangat besar (berkas kecil bisa mengembang jadi ratusan MB di memori)."""
    import io
    from PIL import Image
    try:
        with Image.open(io.BytesIO(file_bytes)) as im:
            w, h = im.size
            if w * h > _max_pixels():
                raise BadImage(f"Gambar terlalu besar ({w}×{h} piksel). "
                               f"Batasnya {_max_pixels() // 1_000_000} megapiksel.")
            if ext in _MIME_GEMINI:
                im.verify()                      # cek integritas tanpa mendekode penuh
                return file_bytes, _MIME_GEMINI[ext]
            buf = io.BytesIO()                   # bmp/tif/gif -> PNG (halaman pertama)
            im.convert("RGB").save(buf, format="PNG")
            return buf.getvalue(), "image/png"
    except BadImage:
        raise
    except Exception:
        raise BadImage("Berkas gambar tidak dapat dibuka (rusak atau bukan gambar).") from None


def _to_images(file_bytes, filename):
    """PDF -> list (png_bytes, mime) per halaman; gambar -> [(bytes, mime)]."""
    name = (filename or "").lower()
    if name.endswith(".pdf"):
        import fitz  # PyMuPDF
        dpi = int(os.environ.get("IMLASCAN_OCR_DPI", "200"))
        out = []
        try:
            doc = fitz.open(stream=file_bytes, filetype="pdf")
        except Exception:
            raise BadImage("Berkas PDF tidak dapat dibuka (rusak atau bukan PDF).") from None
        try:
            if doc.needs_pass:
                raise BadImage("PDF ini dilindungi kata sandi.")
            batas = int(os.environ.get("IMLASCAN_MAX_PAGES", "5"))
            if doc.page_count > batas:
                raise TooManyPages(f"PDF berisi {doc.page_count} halaman; batasnya {batas} halaman per unggahan.")
            if doc.page_count == 0:
                raise BadImage("PDF tidak berisi halaman.")
            for page in doc:
                # Halaman berukuran fisik sangat besar: turunkan DPI agar sisi terpanjang <= 3000 px
                sisi = max(page.rect.width, page.rect.height) or 1.0
                dpi_hal = min(dpi, max(36, int(3000 * 72 / sisi)))
                out.append((page.get_pixmap(dpi=dpi_hal).tobytes("png"), "image/png"))
        finally:
            doc.close()
        return out
    if name.endswith(IMAGE_EXTS):
        return [_siapkan_gambar(file_bytes, name.rsplit(".", 1)[-1])]
    raise ValueError("Format tidak didukung untuk OCR (pakai PDF atau gambar).")


def _run_engine(images, engine):
    engine = (engine or _provider()).lower()
    if engine == "gemini":
        kuota.ambil(len(images))      # melempar kuota.KuotaHabis bila kuota global tak cukup
        if len(images) > 1:
            # Halaman PDF dikirim ke Gemini bersamaan (tiap halaman 30-90 dtk bila berurutan);
            # urutan hasil tetap urutan halaman.
            with ThreadPoolExecutor(max_workers=min(4, len(images))) as pool:
                parts = list(pool.map(lambda bm: _ocr_image_bytes(bm[0], mime=bm[1], engine=engine), images))
            return "\n".join(p for p in parts if p).strip()
    parts = [_ocr_image_bytes(b, mime=m, engine=engine) for b, m in images]
    return "\n".join(p for p in parts if p).strip()


def extract_text(file_bytes, filename, engine=None):
    """
    Ekstrak teks dari PDF/gambar dengan satu engine (default: IMLASCAN_OCR_PROVIDER).
    Return str (bisa kosong). Melempar Exception bila gagal (ditangani view).
    """
    return _run_engine(_to_images(file_bytes, filename), engine)


def extract_compare(file_bytes, filename, engines=COMPARE_ENGINES):
    """
    OCR satu berkas dengan beberapa engine sekaligus (paralel) untuk perbandingan.
    Return list[dict]: {"engine","label","text","error","seconds"} — urutan = engines.
    Kegagalan satu engine tidak menggagalkan engine lain.
    """
    images = _to_images(file_bytes, filename)

    def job(engine):
        res = {"engine": engine, "label": ENGINE_LABELS.get(engine, engine),
               "text": "", "error": "", "seconds": 0.0, "kuota_habis": False}
        if not is_engine_configured(engine):
            res["error"] = f"{res['label']} belum dikonfigurasi."
            return res
        t0 = time.time()
        try:
            res["text"] = _run_engine(images, engine)
        except kuota.KuotaHabis as exc:
            res["error"], res["kuota_habis"] = str(exc), True
        except Exception as exc:          # noqa: BLE001
            res["error"] = _aman(exc)
        res["seconds"] = round(time.time() - t0, 1)
        return res

    # Batas waktu total: permintaan web tidak boleh menggantung melebihi timeout server.
    # Thread yang terlambat dibiarkan selesai sendiri di latar (tidak bisa dihentikan paksa).
    batas = float(os.environ.get("IMLASCAN_OCR_DEADLINE", "150"))
    pool = ThreadPoolExecutor(max_workers=len(engines))
    t0 = time.time()
    try:
        futs = [pool.submit(job, e) for e in engines]
        hasil = []
        for e, f in zip(engines, futs):
            try:
                hasil.append(f.result(timeout=max(0.1, batas - (time.time() - t0))))
            except FutureTimeout:
                hasil.append({"engine": e, "label": ENGINE_LABELS.get(e, e), "text": "",
                              "error": f"melebihi batas waktu {batas:g} detik",
                              "seconds": round(time.time() - t0, 1), "kuota_habis": False})
        return hasil
    finally:
        pool.shutdown(wait=False)

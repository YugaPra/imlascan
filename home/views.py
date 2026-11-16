# -*- coding: utf-8 -*-
"""
home/views.py — Pipeline GEC Arab (ImlaScan)

SCOPE (arahan Bu Mia): HANYA Ortografi (Imla') + Morfologi.
Sintaksis / I'rab / Nahwu SENGAJA di luar cakupan → difilter keluar.

Arsitektur pipeline:
    teks  ->  run_gec_model()  ->  filter scope  ->  format kontrak template
              (LUBANG COLOKAN)     (buang non                (index.html
               model AraT5          ortho/morf)               tidak berubah)

Sumber tabel tag: Belkebir & Habash (2021), "Automatic Error Type Annotation
for Arabic" (ARETA), Table 1 — taksonomi ALC.
https://aclanthology.org/2021.conll-1.47/
"""

from django.conf import settings
from django.core.cache import cache
from django.shortcuts import render
from django.utils.html import escape
from django.utils.safestring import mark_safe
import difflib
import json
import logging
import re
import time

log = logging.getLogger("imlascan")

# Modul integrasi model AraT5 (lokal). Aman diimpor: torch dimuat lazy di dalamnya,
# dan bila belum di-setup, analyze_with_model() mengembalikan None (fallback lexicon).
from . import gec_model
from . import ocr  # OCR Arab: Gemini (utama) + Tesseract (pembanding)
from . import kuota  # kuota global halaman Gemini per hari/bulan


# ===========================================================================
# 1) TABEL TAG ARETA — hanya 13 tag yang MASUK scope (Ortografi + Morfologi)
#    'short' = teks badge singkat; 'rule' = penjelasan kaidah (bahasa Indonesia)
#    Definisi & contoh diambil persis dari Tabel 1 paper ARETA.
# ===========================================================================
TAG_INFO = {
    # --- Ortografi / Imla' (11 tag) ---
    "OA": {"kelas": "Ortografi", "short": "Alif/Ya/Alif-Maqsurah",
           "rule": "Kekeliruan antara Alif (ا), Ya (ي), dan Alif Maqsurah (ى) "
                   "di akhir kata. Contoh: علی → على."},
    "OC": {"kelas": "Ortografi", "short": "Urutan huruf terbalik",
           "rule": "Urutan huruf dalam kata tertukar. Contoh: تبرینا → تربینا."},
    "OD": {"kelas": "Ortografi", "short": "Kelebihan huruf",
           "rule": "Terdapat huruf tambahan yang seharusnya tidak ada. "
                   "Contoh: يعدوم → يدوم."},
    "OG": {"kelas": "Ortografi", "short": "Pemanjangan vokal pendek",
           "rule": "Vokal pendek ditulis memanjang (menambah huruf mad). "
                   "Contoh: نقيمو → نقيم."},
    "OH": {"kelas": "Ortografi", "short": "Hamzah",
           "rule": "Penulisan Hamzah keliru (Qath'/Washl atau dudukan Hamzah). "
                   "Contoh: اكثر → أكثر."},
    "OM": {"kelas": "Ortografi", "short": "Kekurangan huruf",
           "rule": "Ada huruf yang hilang / tidak ditulis. Contoh: سالين → سائلين."},
    "ON": {"kelas": "Ortografi", "short": "Nun/Tanwin",
           "rule": "Kekeliruan antara Nun (ن) dan Tanwin. Contoh: ثوبن → ثوبٌ."},
    "OR": {"kelas": "Ortografi", "short": "Penggantian huruf",
           "rule": "Satu atau lebih huruf tertukar dengan huruf lain. "
                   "Contoh: مصلنا → وصلنا."},
    "OS": {"kelas": "Ortografi", "short": "Pemendekan vokal panjang",
           "rule": "Vokal panjang (mad) ditulis pendek. Contoh: أوقت → أوقات."},
    "OT": {"kelas": "Ortografi", "short": "Ta' Marbuthah/Ha",
           "rule": "Kekeliruan antara Ha (ه) dan Ta' Marbuthah (ة). "
                   "Contoh: مشاركه → مشاركة."},
    "OW": {"kelas": "Ortografi", "short": "Alif Fariqah",
           "rule": "Alif Fariqah setelah Waw jamak hilang/keliru. "
                   "Contoh: وكانو → وكانوا."},

    # --- Morfologi (2 tag) ---
    "MI": {"kelas": "Morfologi", "short": "Infleksi kata",
           "rule": "Bentuk infleksi / pola kata (wazan) keliru. "
                   "Contoh: معروف → عارف."},
    "MT": {"kelas": "Morfologi", "short": "Kala verba",
           "rule": "Bentuk kala (tense) kata kerja keliru. "
                   "Contoh: تفرحني → أفرحتني."},
}

# Tag yang MASUK scope. Catatan: MG (merge) & SP (split) menunggu konfirmasi
# Bu Mia → sengaja TIDAK dimasukkan (default di luar scope). Untuk mengaktifkan,
# tambahkan definisinya ke TAG_INFO lalu masukkan kodenya ke set ini.
SCOPE_TAGS = set(TAG_INFO.keys())  # 13 tag ortografi + morfologi


def _in_scope(tag):
    """True bila SEMUA sub-tag (mis. 'OH+OA') ada di dalam scope."""
    return all(part in SCOPE_TAGS for part in tag.split("+"))


def _scope_part(tag):
    """Ambil hanya sub-tag yang masuk scope. 'OH+XG' -> 'OH'; 'XG' -> '' (dibuang).
    Dipakai sejak ARETA (opsi B): ARETA juga memberi tag sintaksis, sehingga kata
    dengan kesalahan hamzah + gender tetap ditampilkan sebagai kesalahan hamzah."""
    return "+".join(p for p in tag.split("+") if p in SCOPE_TAGS)


def _badge(tag):
    """Teks badge singkat, mis. 'OT · Ta' Marbuthah/Ha' atau gabungan."""
    parts = tag.split("+")
    shorts = [TAG_INFO[p]["short"] for p in parts if p in TAG_INFO]
    return f"{tag} · " + " + ".join(shorts) if shorts else tag


def _rule_text(tag):
    """Penjelasan kaidah gabungan (bahasa Indonesia)."""
    parts = tag.split("+")
    rules = [TAG_INFO[p]["rule"] for p in parts if p in TAG_INFO]
    return " ".join(rules) if rules else "Kesalahan ortografi."


# ===========================================================================
# 2) MESIN SEMENTARA (rule-based lexicon)  —  PLACEHOLDER
#    Ganti seluruh isi run_gec_model() dengan inferensi model AraT5 (Langkah 2).
#    Lexicon kurasi: aman & tidak menimbulkan false-positive. Setiap entri diberi
#    kode tag ARETA yang benar.
# ===========================================================================
_RULE_LEXICON = {
    # 6 kata kalimat demo (target verifikasi: 6 error)
    "احمد":    ("أحمد",    "OH"),      # tambah Hamzah Qath'
    "الي":     ("إلى",     "OH+OA"),   # Hamzah + Ya→Alif Maqsurah
    "فى":      ("في",      "OA"),      # Alif Maqsurah → Ya
    "المدرسه": ("المدرسة", "OT"),      # Ha → Ta' Marbuthah
    "مدينه":   ("مدينة",   "OT"),
    "جميله":   ("جميلة",   "OT"),

    # perluasan ortografi (umum, aman)
    "انا":     ("أنا",     "OH"),
    "انت":     ("أنت",     "OH"),
    "اكثر":    ("أكثر",    "OH"),
    "اقل":     ("أقل",     "OH"),
    "لاكن":    ("لكن",     "OD"),      # kelebihan Alif
    "هاذا":    ("هذا",     "OD"),
    "الذى":    ("الذي",    "OA"),
    "التى":    ("التي",    "OA"),
    "هذة":     ("هذه",     "OT"),

    # contoh morfologi (slot demonstrasi tag M*)
    "تفرحني":  ("أفرحتني", "MT"),
}


_LAST_SOURCE = None  # sumber hasil terakhir, untuk ditampilkan/di-debug


def run_gec_model(text):
    """
    ===================== TITIK MASUK MESIN GEC (Langkah 2) =====================
    KONTRAK (dipakai oleh analyze_imla; jangan diubah):
        INPUT  : text (str)  — teks Arab mentah
        OUTPUT : list[dict]  — tiap item: {
                     "word":       <kata salah pada teks sumber> (str),
                     "suggestion": <koreksi> (str),
                     "tag":        <kode ARETA, boleh gabungan 'OH+OA'> (str),
                 }

    Strategi:
      1. Coba model AraT5 lokal (home/gec_model.py). Aktif bila env var
         IMLASCAN_MODEL_PATH menunjuk folder model yang valid & torch terpasang.
      2. Bila model belum di-setup / gagal → fallback ke MESIN SEMENTARA
         (lexicon rule-based) agar aplikasi tetap berfungsi.
    Filter scope & pemformatan ditangani analyze_imla() — tak perlu diubah.
    ===========================================================================
    """
    global _LAST_SOURCE
    # 1) Coba model AraT5 asli (tag dari ARETA bila layanannya aktif, else heuristik)
    model_result = gec_model.analyze_with_model(text)
    if model_result is not None:
        _LAST_SOURCE = f"AraT5 + {gec_model.LAST_TAGGER}"
        return model_result

    # 2) Fallback: mesin sementara (lexicon)
    _LAST_SOURCE = "lexicon"
    raw = []
    for word in text.split():
        clean = re.sub(r"[^\w]", "", word, flags=re.UNICODE)  # buang tanda baca
        if clean in _RULE_LEXICON:
            suggestion, tag = _RULE_LEXICON[clean]
            raw.append({"word": clean, "suggestion": suggestion, "tag": tag})
    return raw


# ===========================================================================
# 3) ORKESTRATOR — memanggil mesin, memfilter scope, memformat sesuai template
# ===========================================================================
def analyze_imla(text):
    """
    Menjalankan pipeline GEC. Bentuk return SENGAJA dipertahankan sama seperti
    versi lama agar index.html tidak perlu diubah:
        { "extracted_text": str, "total_errors": int, "errors": [ {...} ] }
    dengan tiap error = {"word","error_type","rule","suggestion"}.
    """
    raw_errors = run_gec_model(text)

    errors_found = []
    for e in raw_errors:
        tag = _scope_part(e["tag"])   # FILTER SCOPE: simpan hanya sub-tag ortografi/morfologi
        if not tag:                   # seluruhnya di luar cakupan (mis. XG, PM, SP) -> buang
            continue
        errors_found.append({
            "word": e["word"],
            "error_type": _badge(tag),     # badge: "OT · Ta' Marbuthah/Ha"
            "rule": _rule_text(tag),       # penjelasan kaidah (Indonesia)
            "suggestion": e["suggestion"],
        })

    return {
        "extracted_text": text,
        "total_errors": len(errors_found),
        "errors": errors_found,
        "tagger": _LAST_SOURCE,   # info: "AraT5 + areta" | "AraT5 + heuristic" | "lexicon"
    }



# ===========================================================================
# 3b) PEMBANDING OCR — Gemini vs Tesseract
#     Kemiripan = kesepakatan antar-OCR, BUKAN akurasi (tanpa teks acuan manusia).
# ===========================================================================
_HARAKAT = re.compile(r"[\u064B-\u0652\u0670\u0640]")


def _norm_ocr(t):
    return re.sub(r"\s+", " ", _HARAKAT.sub("", t or "")).strip()


def _diff_html(a, b):
    """Sorot kata yang berbeda antara dua teks (level kata). Return (html_a, html_b)."""
    wa, wb = _norm_ocr(a).split(), _norm_ocr(b).split()
    out_a, out_b = [], []
    for op, i1, i2, j1, j2 in difflib.SequenceMatcher(None, wa, wb, autojunk=False).get_opcodes():
        sa, sb = " ".join(wa[i1:i2]), " ".join(wb[j1:j2])
        if op == "equal":
            out_a.append(escape(sa)); out_b.append(escape(sb))
        else:
            if sa:
                out_a.append(f'<mark class="bg-tertiary-fixed text-on-tertiary-fixed rounded px-1">{escape(sa)}</mark>')
            if sb:
                out_b.append(f'<mark class="bg-tertiary-fixed text-on-tertiary-fixed rounded px-1">{escape(sb)}</mark>')
    return mark_safe(" ".join(out_a)), mark_safe(" ".join(out_b))


def build_ocr_comparison(results, used_engine):
    """results = list dict dari ocr.extract_compare (atau dari JSON form)."""
    by = {r["engine"]: r for r in results}
    cols = []
    for r in results:
        txt = r.get("text", "")
        cols.append({**r, "words": len(_norm_ocr(txt).split()) if txt else 0,
                     "chars": len(_norm_ocr(txt)), "html": escape(txt),
                     "used": r["engine"] == used_engine})
    sim = None
    a, b = by.get("gemini", {}).get("text", ""), by.get("tesseract", {}).get("text", "")
    if a and b:
        sim = round(100 * difflib.SequenceMatcher(None, _norm_ocr(a), _norm_ocr(b), autojunk=False).ratio(), 1)
        ha, hb = _diff_html(a, b)
        for c in cols:
            c["html"] = ha if c["engine"] == "gemini" else hb if c["engine"] == "tesseract" else c["html"]
    return {
        "kolom": cols,
        "tunggal": len(cols) == 1,   # True = hanya satu mesin OCR yang dijalankan
        "kemiripan": sim,
        "dipakai": ocr.ENGINE_LABELS.get(used_engine, used_engine),
        "json": json.dumps([{k: r.get(k, "") for k in ("engine", "label", "text", "error", "seconds")}
                            for r in results], ensure_ascii=False),
    }

# ===========================================================================
# 3c) PENGAMAN: pembatas laju per alamat IP + batas ukuran masukan
# ===========================================================================
def _client_ip(request):
    """Alamat IP pengunjung. Di belakang nginx (IMLASCAN_BEHIND_PROXY=True) dipakai entri
    TERAKHIR X-Forwarded-For, yaitu yang ditambahkan nginx sendiri (entri awal bisa dipalsukan)."""
    if getattr(settings, "BEHIND_PROXY", False):
        xff = request.META.get("HTTP_X_FORWARDED_FOR", "")
        if xff:
            return xff.split(",")[-1].strip()
    return request.META.get("REMOTE_ADDR", "") or "?"


def _lewat_batas(request, nama, aturan):
    """aturan = "jumlah/detik" (mis. "10/600"). Return sisa detik bila batas terlampaui, else 0.
    Memakai cache Django (memori proses) -> cukup untuk satu server / satu worker."""
    try:
        jumlah, detik = (int(x) for x in str(aturan).split("/"))
    except ValueError:
        return 0            # kosong / format salah = pembatas mati
    if jumlah <= 0 or detik <= 0:
        return 0
    jendela = int(time.time() // detik)
    kunci = f"imlascan:rl:{nama}:{_client_ip(request)}:{jendela}"
    if cache.add(kunci, 1, timeout=detik):
        return 0
    try:
        n = cache.incr(kunci)
    except ValueError:      # kunci kedaluwarsa di antara add & incr
        cache.set(kunci, 1, timeout=detik)
        return 0
    return 0 if n <= jumlah else max(1, int((jendela + 1) * detik - time.time()))


def _pesan_tunggu(sisa):
    return (f"Terlalu banyak permintaan dari alamat Anda. Coba lagi dalam "
            f"{max(1, round(sisa / 60))} menit.")


# ===========================================================================
# 4) VIEW UTAMA
# ===========================================================================
def index(request):
    """Halaman beranda: GET menampilkan form, POST memproses teks/berkas."""
    context = {}

    if request.method == "POST":
        teks_input = request.POST.get("teks_arab", "").strip()
        dokumen_upload = request.FILES.get("dokumen_arab")

        extracted_text = ""

        sisa = _lewat_batas(request, "semua", settings.IMLASCAN_RATE_LIMIT)
        maks_byte = int(settings.IMLASCAN_MAX_UPLOAD_MB * 1024 * 1024)
        if sisa:
            context["error_message"] = _pesan_tunggu(sisa)
        elif dokumen_upload and dokumen_upload.size > maks_byte:
            context["error_message"] = (
                f"Berkas terlalu besar ({dokumen_upload.size / 1048576:.1f} MB). "
                f"Batasnya {settings.IMLASCAN_MAX_UPLOAD_MB:g} MB.")
        elif dokumen_upload:
            nama = dokumen_upload.name.lower()
            if nama.endswith(".txt"):
                try:
                    extracted_text = dokumen_upload.read().decode(
                        "utf-8", errors="ignore").strip()
                except Exception:
                    context["error_message"] = "Gagal membaca berkas .txt."
            elif nama.endswith(".pdf") or nama.endswith(ocr.IMAGE_EXTS):
                # OCR Arab: mesin dipilih pengguna lewat tombol di form.
                #   gemini | tesseract  -> hanya mesin itu yang dijalankan (hemat waktu & kuota)
                #   keduanya            -> Gemini + Tesseract paralel sebagai pembanding
                pilihan = request.POST.get("ocr_engine", "gemini").strip().lower()
                if pilihan == "keduanya":
                    engines = ocr.COMPARE_ENGINES
                elif pilihan in ocr.COMPARE_ENGINES:
                    engines = (pilihan,)
                else:
                    engines = ("gemini",)
                sisa = _lewat_batas(request, "ocr", settings.IMLASCAN_RATE_LIMIT_OCR)
                if sisa:
                    context["error_message"] = _pesan_tunggu(sisa)
                elif not any(ocr.is_engine_configured(e) for e in engines):
                    petunjuk = {
                        "gemini": "isi IMLASCAN_GEMINI_KEY di .env",
                        "tesseract": "pasang Tesseract + tessdata/ara.traineddata",
                    }
                    context["error_message"] = "OCR belum dikonfigurasi: " + " dan/atau ".join(
                        f"{ocr.ENGINE_LABELS[e]} ({petunjuk[e]})" for e in engines) + "."
                else:
                    try:
                        isi_berkas = dokumen_upload.read()
                        hasil_ocr = ocr.extract_compare(
                            isi_berkas, dokumen_upload.name, engines=engines)
                        # Gemini tidak memberi teks (kuota global habis, gagal, atau terlalu lama)
                        # -> jangan gagal: baca dengan Tesseract dan beri tahu pengguna.
                        g = next((r for r in hasil_ocr if r["engine"] == "gemini"), None)
                        if g is not None and not g["text"]:
                            if g.get("kuota_habis"):
                                sebab = (f"Kuota harian Gemini OCR ({kuota.status()['maks']} halaman untuk "
                                         f"semua pengguna) tidak mencukupi")
                                akhir = " Kuota Gemini kembali penuh besok."
                            else:
                                sebab, akhir = "Gemini OCR sedang tidak dapat memproses berkas ini", ""
                            if "tesseract" not in engines and ocr.is_engine_configured("tesseract"):
                                hasil_ocr = ocr.extract_compare(
                                    isi_berkas, dokumen_upload.name, engines=("tesseract",))
                            if any(r["engine"] == "tesseract" and r["text"] for r in hasil_ocr):
                                context["info_message"] = (
                                    f"{sebab}, jadi berkas ini dibaca dengan Tesseract OCR. Tesseract "
                                    f"dilatih untuk teks cetak; pada tulisan tangan hasilnya lebih kasar."
                                    f"{akhir}")
                        if not settings.DEBUG:      # rincian error mesin OCR hanya ke log server
                            for r in hasil_ocr:
                                if r["error"] and not r.get("kuota_habis"):
                                    log.warning("OCR %s gagal: %s", r["engine"], r["error"])
                                    r["error"] = "tidak dapat memproses berkas ini"
                        # Teks untuk dianalisis: mesin pertama yang menghasilkan teks
                        # (mode keduanya: Gemini bila ada, kalau gagal pakai Tesseract)
                        pilih = next((r for r in hasil_ocr if r["text"]), None)
                        if pilih:
                            extracted_text = pilih["text"]
                            context["ocr_bandingan"] = build_ocr_comparison(
                                hasil_ocr, pilih["engine"])
                        else:
                            errs = "; ".join(f"{r['label']}: {r['error'] or 'teks kosong'}"
                                             for r in hasil_ocr)
                            context["error_message"] = f"OCR tidak menghasilkan teks ({errs})."
                    except (ocr.TooManyPages, ocr.BadImage) as exc:
                        context["error_message"] = str(exc)
                    except Exception as exc:
                        # Rincian error hanya ke log server; pengguna cukup pesan umum
                        # (rincian bisa memuat path/konfigurasi server).
                        log.exception("OCR gagal")
                        context["error_message"] = (
                            f"OCR gagal: {ocr._aman(exc)}" if settings.DEBUG
                            else "OCR gagal memproses berkas ini. Coba berkas lain atau ulangi sebentar lagi.")
            else:
                context["error_message"] = (
                    "Format tidak didukung. Gunakan tab 'Salin Teks', berkas "
                    ".txt, PDF, atau gambar.")
        elif teks_input:
            extracted_text = teks_input
            # Analisis ulang dengan teks OCR lain → bangun ulang tabel pembanding tanpa OCR ulang
            ocr_json = request.POST.get("ocr_json", "")
            sumber = request.POST.get("sumber_ocr", "")
            if ocr_json and sumber:
                try:
                    context["ocr_bandingan"] = build_ocr_comparison(json.loads(ocr_json), sumber)
                except (ValueError, KeyError, TypeError):
                    pass

        maks_teks = settings.IMLASCAN_MAX_TEXT_CHARS
        if extracted_text and len(extracted_text) > maks_teks and "error_message" not in context:
            context.pop("ocr_bandingan", None)
            context["error_message"] = (
                f"Teks terlalu panjang ({len(extracted_text):,} karakter). "
                f"Batasnya {maks_teks:,} karakter per analisis.").replace(",", ".")
        elif extracted_text and "error_message" not in context:
            context["hasil_analisis"] = analyze_imla(extracted_text)
        elif "error_message" not in context:
            context["error_message"] = (
                "Mohon masukkan teks atau unggah dokumen terlebih dahulu "
                "sebelum memeriksa.")

    if "hasil_analisis" not in context:      # form ditampilkan -> sertakan sisa kuota Gemini
        context["kuota_gemini"] = kuota.status()
    return render(request, "home/index.html", context)

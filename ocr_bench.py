#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ocr_bench.py — Uji banding OCR tulisan tangan Arab (Arabic GEC / ImlaScan)

Tujuan: menjalankan beberapa OCR pada esai yang SAMA, menyimpan hasilnya
berdampingan, lalu membuat laporan:
  * TANPA ground truth : kesepakatan antar-OCR, profil ortografi (ة/ه, ى/ي, hamzah),
                         bendera masalah (HTML, pengulangan, harakat, huruf Latin),
                         dan lembar review untuk ahli bahasa Arab.
  * DENGAN ground truth: CER / WER terhadap transkripsi verbatim ahli.
  * DENGAN daftar error : EPR (Error Preservation Rate) — berapa % kesalahan asli
                         pelajar yang tetap dipertahankan OCR.

Perintah:
  python ocr_bench.py init                      # buat folder & template
  python ocr_bench.py list                      # daftar OCR yang tersedia
  python ocr_bench.py run --providers all       # jalankan OCR + buat laporan
  python ocr_bench.py report                    # buat ulang laporan saja
  python ocr_bench.py segment                   # hanya potong baris (cek visual)

Lihat CARA_PAKAI.md untuk panduan lengkap.
"""
from __future__ import annotations

import argparse
import csv
import gc
import html
import itertools
import json
import os
import re
import sys
import time
import traceback
from collections import Counter, defaultdict
from difflib import SequenceMatcher
from pathlib import Path

try:  # Windows console: agar huruf Arab tidak bikin crash saat print
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

try:
    from dotenv import load_dotenv  # opsional: baca API key dari file .env
    load_dotenv()
except Exception:
    pass

IMG_EXT = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"}

# =============================================================================
# 1. PROMPT
# =============================================================================
# Prompt "mesin OCR" untuk VLM umum (Gemini, Qwen2.5-VL, AIN). Framing ini
# sengaja melarang koreksi — lihat learnings: framing "asisten bahasa" memicu
# normalisasi diam-diam.
FAITHFUL_PROMPT = (
    "You are a literal OCR engine, not a language assistant. Transcribe the "
    "handwritten Arabic text in this image EXACTLY as written, character by character.\n"
    "Rules:\n"
    "- Do NOT correct spelling, grammar, hamza, taa marbuta (ة/ه), alif maqsura (ى/ي) "
    "or any other mistake. Mistakes must stay exactly as the writer wrote them.\n"
    "- Do NOT add diacritics (harakat) that are not visible in the image.\n"
    "- Do NOT add, remove, translate, reorder or summarize anything.\n"
    "- Keep the original line breaks. If a word is illegible, write [؟].\n"
    "- Output plain text only: no markdown, no HTML, no commentary."
)
# Prompt resmi dari model card masing-masing (model dilatih dengan prompt ini).
QARI_PROMPT = (
    "Below is the image of one page of a document, as well as some raw textual "
    "content that was previously extracted for it. Just return the plain text "
    "representation of this document as if you were reading it naturally. Do not hallucinate."
)
AMIDDA_PROMPT = "Transcribe the handwritten text in this line image. Output ONLY the transcription."


# =============================================================================
# 2. UTIL TEKS ARAB
# =============================================================================
HARAKAT_RE = re.compile(r"[ؐ-ًؚ-ٰٟۖ-ۭ]")
TATWEEL = "ـ"
HTML_TAG_RE = re.compile(r"<[^>]+>")
LATIN_RE = re.compile(r"[A-Za-z]")
ARABIC_LETTER_RE = re.compile(r"[ء-يٱ-ۓ]")


def clean_output(text: str) -> str:
    """Bersihkan pembungkus format (HTML/markdown) TANPA menyentuh huruf Arab."""
    t = text.replace("\r\n", "\n")
    t = re.sub(r"```[a-zA-Z]*", "", t)                 # code fence
    t = re.sub(r"<br\s*/?>", "\n", t, flags=re.I)
    t = HTML_TAG_RE.sub("", t)                          # <h3><u><b> dsb. (QARI v0.3)
    t = re.sub(r"!\[[^\]]*\]\([^)]*\)", "", t)          # gambar markdown (Mistral)
    t = re.sub(r"^\s{0,3}#{1,6}\s*", "", t, flags=re.M)  # heading markdown
    t = re.sub(r"(\*\*|__|\*|`)", "", t)                # bold/italic/code
    t = re.sub(r"^\s*[-|:]{3,}\s*$", "", t, flags=re.M)  # garis tabel markdown
    t = t.replace("|", " ")
    t = re.sub(r"[ \t]+", " ", t)
    t = re.sub(r"\n{3,}", "\n\n", t)
    return t.strip()


def norm_for_scoring(text: str, keep_harakat: bool = False) -> str:
    """Normalisasi RINGAN untuk skor. Huruf TIDAK dinormalisasi (ة/ه, ى/ي, hamzah
    tetap beda) karena justru itu yang mau diukur."""
    t = text.replace(TATWEEL, "")
    if not keep_harakat:
        t = HARAKAT_RE.sub("", t)
    t = re.sub(r"\s+", " ", t)
    return t.strip()


def words(text: str, keep_harakat: bool = False) -> list[str]:
    return norm_for_scoring(text, keep_harakat).split()


def levenshtein(a, b) -> int:
    """Jarak edit (bisa untuk string atau list kata)."""
    try:
        from rapidfuzz.distance import Levenshtein as L
        return L.distance(a, b)
    except Exception:
        pass
    if len(a) < len(b):
        a, b = b, a
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def cer(hyp: str, ref: str) -> float:
    ref_n = norm_for_scoring(ref)
    return levenshtein(norm_for_scoring(hyp), ref_n) / max(1, len(ref_n))


def wer(hyp: str, ref: str) -> float:
    rw = words(ref)
    return levenshtein(words(hyp), rw) / max(1, len(rw))


def repetition_score(text: str, n: int = 4) -> float:
    """Porsi n-gram kata yang berulang (0 = tidak ada, mendekati 1 = loop)."""
    w = words(text)
    grams = [tuple(w[i:i + n]) for i in range(len(w) - n + 1)]
    if not grams:
        return 0.0
    return 1 - len(set(grams)) / len(grams)


# Kelompok huruf yang sering tertukar oleh pelajar (sesuai kelas ARETA)
CONFUSION = [
    ("OT", set("ةهت")),        # Ha / Ta / Ta-marbuta
    ("OA", set("ىيا")),        # Alif / Ya / Alif-maqsura
    ("OH", set("أإآاءؤئوي")),  # Hamzah
]


def ortho_category(a: str, b: str) -> str | None:
    """Jika dua kata HANYA berbeda pada huruf rawan (ة/ه, ى/ي, hamzah),
    kembalikan kategorinya (mis. 'OT' atau 'OT+OH'). Selain itu None."""
    a, b = norm_for_scoring(a), norm_for_scoring(b)
    if a == b or not a or not b:
        return None
    cats = set()
    for op, i1, i2, j1, j2 in SequenceMatcher(None, a, b, autojunk=False).get_opcodes():
        if op == "equal":
            continue
        sa, sb = a[i1:i2], b[j1:j2]
        if op == "replace" and len(sa) == len(sb):
            for ca, cb in zip(sa, sb):
                cat = next((c for c, s in CONFUSION if ca in s and cb in s), None)
                if cat is None:
                    return None
                cats.add(cat)
        elif set(sa + sb) <= {"ء"}:          # hamzah hilang / ditambah
            cats.add("OH")
        else:
            return None
    return "+".join(sorted(cats)) if cats else None


def ortho_profile(text: str) -> dict:
    """Hitung bentuk huruf rawan di akhir/awal kata — sidik jari normalisasi."""
    w = words(text)
    c = Counter()
    for x in w:
        c["final_ta_marbuta_ة"] += x.endswith("ة")
        c["final_ha_ه"] += x.endswith("ه")
        c["final_alif_maqsura_ى"] += x.endswith("ى")
        c["final_ya_ي"] += x.endswith("ي")
        c["initial_hamza_alif_أإآ"] += x[:1] in "أإآ"
        c["initial_bare_alif_ا"] += x[:1] == "ا"
    tm = c["final_ta_marbuta_ة"] + c["final_ha_ه"]
    am = c["final_alif_maqsura_ى"] + c["final_ya_ي"]
    ha = c["initial_hamza_alif_أإآ"] + c["initial_bare_alif_ا"]
    d = dict(c)
    d["share_ة_of_(ة+ه)"] = round(c["final_ta_marbuta_ة"] / tm, 3) if tm else ""
    d["share_ى_of_(ى+ي)"] = round(c["final_alif_maqsura_ى"] / am, 3) if am else ""
    d["share_hamza_of_initial_alif"] = round(c["initial_hamza_alif_أإآ"] / ha, 3) if ha else ""
    return d


# =============================================================================
# 3. INPUT: PDF / GAMBAR -> HALAMAN
# =============================================================================
def list_docs(input_dir: Path) -> list[Path]:
    return sorted(p for p in input_dir.iterdir()
                  if p.is_file() and (p.suffix.lower() == ".pdf" or p.suffix.lower() in IMG_EXT))


def load_pages(doc: Path, out: Path, dpi: int = 300):
    """Kembalikan list PIL.Image per halaman. PDF dirender sekali lalu di-cache."""
    from PIL import Image
    cache = out / "pages" / doc.stem
    cached = sorted(cache.glob("p*.png")) if cache.exists() else []
    if cached:
        return [Image.open(p).convert("RGB") for p in cached]
    cache.mkdir(parents=True, exist_ok=True)
    if doc.suffix.lower() == ".pdf":
        try:
            import pymupdf as fitz  # PyMuPDF baru
        except ImportError:
            import fitz             # PyMuPDF lama
        pages = []
        with fitz.open(doc) as pdf:
            for i, page in enumerate(pdf, 1):
                # batasi sisi terpanjang ~3508 px (A4 @300dpi) agar PDF hasil
                # scan HP berukuran halaman raksasa tidak jadi gambar 10.000 px
                zoom = min(dpi / 72, 3508 / max(page.rect.width, page.rect.height))
                pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom))
                img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
                img.save(cache / f"p{i:03d}.png")
                pages.append(img)
        return pages
    img = Image.open(doc).convert("RGB")
    img.save(cache / "p001.png")
    return [img]


# =============================================================================
# 4. SEGMENTASI BARIS (untuk model per-baris: AMIDDA, ArTrOCR)
# =============================================================================
def segment_lines(img, min_h_frac=0.012, merge_gap_frac=0.006, pad_frac=0.008):
    """Potong halaman jadi baris dengan profil proyeksi horizontal (OpenCV).
    Sederhana & cepat; untuk tulisan miring/rapat, pakai potongan manual
    (folder lines_manual/) atau segmenter kraken di WSL."""
    import numpy as np
    import cv2
    g = np.array(img.convert("L"))
    h, w = g.shape
    _, bw = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    # buang garis buku bergaris (garis horizontal panjang)
    kern = cv2.getStructuringElement(cv2.MORPH_RECT, (max(40, w // 6), 1))
    bw = cv2.subtract(bw, cv2.morphologyEx(bw, cv2.MORPH_OPEN, kern))
    bw = cv2.medianBlur(bw, 3)
    prof = bw.sum(axis=1) / 255.0
    k = max(3, h // 200)
    smooth = np.convolve(prof, np.ones(k) / k, mode="same")
    thr = max(smooth.max() * 0.06, w * 0.003)
    on = smooth > thr
    runs, start = [], None
    for y, v in enumerate(on):
        if v and start is None:
            start = y
        elif not v and start is not None:
            runs.append([start, y]); start = None
    if start is not None:
        runs.append([start, h])
    merged = []
    for r in runs:  # gabung celah kecil (titik/harakat yang terpisah)
        if merged and r[0] - merged[-1][1] < merge_gap_frac * h:
            merged[-1][1] = r[1]
        else:
            merged.append(r)
    pad = int(pad_frac * h)
    boxes = []
    for y0, y1 in merged:
        if y1 - y0 < min_h_frac * h:
            continue
        y0, y1 = max(0, y0 - pad), min(h, y1 + pad)
        cols = np.where(bw[y0:y1].sum(axis=0) > 0)[0]
        if cols.size == 0:
            continue
        x0, x1 = max(0, cols.min() - pad), min(w, cols.max() + pad)
        boxes.append((x0, y0, x1, y1))
    return boxes


def get_lines(doc: Path, pages, out: Path, manual_dir: Path):
    """List (page_idx, PIL line image). Prioritas: lines_manual/<doc>/ jika ada."""
    from PIL import Image, ImageDraw
    man = manual_dir / doc.stem
    if man.exists():
        files = sorted(p for p in man.iterdir() if p.suffix.lower() in IMG_EXT)
        if files:
            return [(0, Image.open(p).convert("RGB")) for p in files]
    ldir = out / "lines" / doc.stem
    ldir.mkdir(parents=True, exist_ok=True)
    result = []
    for pi, page in enumerate(pages, 1):
        boxes = segment_lines(page)
        overlay = page.copy()
        dr = ImageDraw.Draw(overlay)
        for li, b in enumerate(boxes, 1):
            crop = page.crop(b)
            crop.save(ldir / f"p{pi:03d}_l{li:02d}.png")
            dr.rectangle(b, outline=(255, 0, 0), width=4)
            result.append((pi, crop))
        overlay.save(ldir / f"p{pi:03d}_overlay.png")
    return result


# =============================================================================
# 5. PROVIDER OCR
# =============================================================================
RETRY_HINTS = ("429", "RESOURCE_EXHAUSTED", "rate", "503", "UNAVAILABLE",
               "timeout", "Timeout", "overloaded", "500")


def with_retry(fn, tries=6, base=5.0, cap=120.0):
    for k in range(tries):
        try:
            return fn()
        except Exception as e:  # noqa: BLE001
            msg = str(e)
            if k == tries - 1 or not any(h in msg for h in RETRY_HINTS):
                raise
            wait = min(cap, base * 2 ** k)
            print(f"      ! {msg[:90]} -> tunggu {wait:.0f}s lalu coba lagi")
            time.sleep(wait)


def pil_to_png_bytes(img, max_side=None) -> bytes:
    import io
    if max_side and max(img.size) > max_side:
        img = img.copy(); img.thumbnail((max_side, max_side))
    buf = io.BytesIO(); img.save(buf, format="PNG")
    return buf.getvalue()


class Provider:
    name = "base"
    level = "page"   # "page" = satu gambar per halaman, "line" = per baris
    kind = "api"     # "api" (berbayar/online) atau "local" (GPU sendiri)

    def load(self): ...
    def unload(self): ...
    def ocr(self, image) -> str: raise NotImplementedError

    def describe(self) -> str:
        return f"{self.name:<18} {self.kind:<6} per-{self.level:<5}"


class DummyProvider(Provider):
    """Untuk uji pipeline tanpa GPU/API."""
    name, kind = "dummy", "local"

    def ocr(self, image):
        return f"مدرسة جميله في المدينه ({image.size[0]}x{image.size[1]})"


class GeminiProvider(Provider):
    kind = "api"

    def __init__(self, model_id: str, name: str | None = None):
        self.model_id, self.name = model_id, name or model_id
        self.client = None

    def load(self):
        from google import genai
        key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        if not key:
            raise RuntimeError("GEMINI_API_KEY belum di-set (lihat .env)")
        self.client = genai.Client(api_key=key)

    def ocr(self, image):
        from google.genai import types
        # Gemini 2.x: temperature 0 (deterministik). Gemini 3.x: Google menyarankan
        # temperature default 1.0 (nilai rendah bisa memicu loop), jadi tidak diubah.
        kw = {"temperature": 0.0} if self.model_id.startswith("gemini-2") else {}
        if getattr(self, "media_res", "default") != "default":
            kw["media_resolution"] = getattr(types.MediaResolution,
                                             f"MEDIA_RESOLUTION_{self.media_res.upper()}")
        cfg = types.GenerateContentConfig(**kw)
        part = types.Part.from_bytes(data=pil_to_png_bytes(image, 3000), mime_type="image/png")
        resp = with_retry(lambda: self.client.models.generate_content(
            model=self.model_id, contents=[part, FAITHFUL_PROMPT], config=cfg))
        return resp.text or ""


class MistralProvider(Provider):
    kind = "api"

    def __init__(self, model_id="mistral-ocr-latest", name="mistral-ocr"):
        self.model_id, self.name = model_id, name
        self.client = None

    def load(self):
        try:
            from mistralai.client import Mistral   # SDK v2
        except ImportError:
            from mistralai import Mistral          # SDK v1
        key = os.getenv("MISTRAL_API_KEY")
        if not key:
            raise RuntimeError("MISTRAL_API_KEY belum di-set (lihat .env)")
        self.client = Mistral(api_key=key)

    def ocr(self, image):
        import base64
        b64 = base64.b64encode(pil_to_png_bytes(image, 3000)).decode()
        resp = with_retry(lambda: self.client.ocr.process(
            model=self.model_id,
            document={"type": "image_url", "image_url": f"data:image/png;base64,{b64}"}))
        return "\n".join(p.markdown for p in resp.pages)


class HFVisionLM(Provider):
    """Semua VLM Hugging Face (Qwen2/2.5/3.5-VL turunan: QARI, AIN, AMIDDA...)."""
    kind = "local"

    def __init__(self, name, model_id, prompt, level="page", max_new_tokens=2000,
                 quant4=False, processor_kwargs=None, chat_kwargs=None):
        self.name, self.model_id, self.prompt, self.level = name, model_id, prompt, level
        self.max_new_tokens, self.quant4 = max_new_tokens, quant4
        self.processor_kwargs = processor_kwargs or {}
        self.chat_kwargs = chat_kwargs or {}
        self.model = self.processor = None
        self.no_repeat_ngram = 0  # di-set dari CLI

    def load(self):
        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor
        kw = dict(device_map="auto")
        if self.quant4:
            from transformers import BitsAndBytesConfig
            kw["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16)
        else:
            kw["dtype"] = torch.bfloat16
        self.processor = AutoProcessor.from_pretrained(self.model_id, **self.processor_kwargs)
        self.model = AutoModelForImageTextToText.from_pretrained(self.model_id, **kw).eval()

    def unload(self):
        self.model = self.processor = None
        gc.collect()
        try:
            import torch
            torch.cuda.empty_cache()
        except Exception:
            pass

    def ocr(self, image):
        import torch
        messages = [{"role": "user", "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": self.prompt}]}]
        inputs = self.processor.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True,
            return_dict=True, return_tensors="pt", **self.chat_kwargs).to(self.model.device)
        gen = dict(max_new_tokens=self.max_new_tokens, do_sample=False)
        if self.no_repeat_ngram:
            gen["no_repeat_ngram_size"] = self.no_repeat_ngram
        with torch.no_grad():
            out = self.model.generate(**inputs, **gen)
        trimmed = out[:, inputs["input_ids"].shape[1]:]
        return self.processor.batch_decode(trimmed, skip_special_tokens=True)[0].strip()


class TrOCRProvider(Provider):
    """ArTrOCR-HTR (TrOCR, per baris, input 512x102 diletakkan di pojok kiri-atas)."""
    kind, level = "local", "line"

    def __init__(self, name="artrocr", model_id="BushraAlmod03/ArTrOCR-HTR"):
        self.name, self.model_id = name, model_id
        self.model = self.processor = None

    def load(self):
        import torch
        from transformers import TrOCRProcessor, VisionEncoderDecoderModel
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.processor = TrOCRProcessor.from_pretrained(self.model_id)
        self.model = VisionEncoderDecoderModel.from_pretrained(self.model_id).to(self.device).eval()

    def unload(self):
        self.model = self.processor = None
        gc.collect()
        try:
            import torch
            torch.cuda.empty_cache()
        except Exception:
            pass

    def ocr(self, image):
        import torch
        from PIL import Image
        W, H = 512, 102
        im = image.convert("RGB")
        s = min(W / im.width, H / im.height)
        im = im.resize((max(1, int(im.width * s)), max(1, int(im.height * s))), Image.LANCZOS)
        canvas = Image.new("RGB", (W, H), (255, 255, 255))
        canvas.paste(im, (0, 0))
        pv = self.processor(images=canvas, return_tensors="pt").pixel_values.to(self.device)
        with torch.no_grad():
            ids = self.model.generate(pv, max_length=140)
        return self.processor.batch_decode(ids, skip_special_tokens=True)[0].strip()


def build_registry(max_pixels: int) -> dict[str, Provider]:
    pk = {"max_pixels": max_pixels}
    regs = [
        GeminiProvider("gemini-2.5-flash"),
        GeminiProvider("gemini-3-flash-preview", name="gemini-3-flash"),
        MistralProvider(),
        HFVisionLM("amidda", "calfa-ai/amidda1.0Line4B", AMIDDA_PROMPT, level="line",
                   max_new_tokens=128, processor_kwargs={"max_pixels": 401408},
                   chat_kwargs={"enable_thinking": False}),
        TrOCRProvider(),
        HFVisionLM("qari-0.2.2.1", "NAMAA-Space/Qari-OCR-0.2.2.1-VL-2B-Instruct",
                   QARI_PROMPT, processor_kwargs=pk),
        HFVisionLM("ain-7b", "MBZUAI/AIN", FAITHFUL_PROMPT, quant4=True, processor_kwargs=pk),
        HFVisionLM("qwen2.5-vl-7b", "Qwen/Qwen2.5-VL-7B-Instruct", FAITHFUL_PROMPT,
                   quant4=True, processor_kwargs=pk),
        # --- tambahan, tidak ikut "all" ---
        HFVisionLM("qari-0.3", "NAMAA-Space/Qari-OCR-v0.3-VL-2B-Instruct",
                   QARI_PROMPT, processor_kwargs=pk),
        DummyProvider(),
    ]
    return {p.name: p for p in regs}


DEFAULT_ALL = ["gemini-2.5-flash", "gemini-3-flash", "mistral-ocr", "amidda",
               "artrocr", "qari-0.2.2.1", "ain-7b", "qwen2.5-vl-7b"]


def resolve_providers(spec: str, reg: dict) -> list[Provider]:
    names = DEFAULT_ALL if spec.strip() == "all" else [s.strip() for s in spec.split(",") if s.strip()]
    out = []
    for n in names:
        if n.startswith("gemini:"):             # mis. gemini:gemini-3.8-flash
            mid = n.split(":", 1)[1]
            out.append(GeminiProvider(mid))
        elif n in reg:
            out.append(reg[n])
        else:
            sys.exit(f"OCR '{n}' tidak dikenal. Jalankan: python ocr_bench.py list")
    return out


# =============================================================================
# 6. MENJALANKAN OCR
# =============================================================================
def run_provider(p: Provider, docs, args):
    out = Path(args.out)
    raw_dir, clean_dir, meta_dir = (out / "results" / d / p.name for d in ("raw", "clean", "meta"))
    for d in (raw_dir, clean_dir, meta_dir):
        d.mkdir(parents=True, exist_ok=True)
    todo = [d for d in docs if args.force or not (raw_dir / f"{d.stem}.txt").exists()]
    print(f"\n=== {p.name}  ({len(docs) - len(todo)} sudah ada, {len(todo)} akan diproses)")
    if not todo:
        return
    if isinstance(p, HFVisionLM):
        p.no_repeat_ngram = args.no_repeat_ngram
    if isinstance(p, GeminiProvider):
        p.media_res = args.gemini_media_res
    try:
        t0 = time.time(); p.load()
        print(f"    model siap ({time.time() - t0:.0f}s)")
    except Exception as e:  # noqa: BLE001
        print(f"    X GAGAL memuat {p.name}: {e}")
        (meta_dir / "_load_error.txt").write_text(traceback.format_exc(), encoding="utf-8")
        return
    try:
        for doc in todo:
            t0 = time.time(); meta = {"provider": p.name, "doc": doc.name, "level": p.level}
            try:
                pages = load_pages(doc, out, args.dpi)
                if p.level == "line":
                    lines = get_lines(doc, pages, out, Path(args.lines_manual))
                    meta["n_lines"] = len(lines)
                    chunks, last_page = [], None
                    for pi, im in lines:
                        if last_page is not None and pi != last_page:
                            chunks.append("")          # pemisah halaman
                        chunks.append(p.ocr(im)); last_page = pi
                    text = "\n".join(chunks)
                else:
                    text = "\n\n".join(p.ocr(pg) for pg in pages)
                meta["n_pages"] = len(pages)
                meta["ok"] = True
            except Exception as e:  # noqa: BLE001
                text, meta["ok"], meta["error"] = "", False, f"{type(e).__name__}: {e}"
                print(f"    X {doc.name}: {meta['error'][:150]}")
            meta["seconds"] = round(time.time() - t0, 1)
            if meta["ok"]:
                (raw_dir / f"{doc.stem}.txt").write_text(text, encoding="utf-8")
                (clean_dir / f"{doc.stem}.txt").write_text(clean_output(text), encoding="utf-8")
                print(f"    ok {doc.name}  ({meta['seconds']}s, {len(text)} karakter)")
            (meta_dir / f"{doc.stem}.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1),
                                                       encoding="utf-8")
    finally:
        p.unload()


# =============================================================================
# 7. LAPORAN
# =============================================================================
def load_outputs(out: Path) -> dict[str, dict[str, str]]:
    """{provider: {doc_id: clean_text}}"""
    res = defaultdict(dict)
    base = out / "results" / "clean"
    if base.exists():
        for pdir in sorted(base.iterdir()):
            for f in sorted(pdir.glob("*.txt")):
                res[pdir.name][f.stem] = f.read_text(encoding="utf-8")
    return res


def write_csv(path: Path, rows: list[dict]):
    if not rows:
        return
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with open(path, "w", newline="", encoding="utf-8-sig") as f:  # -sig: aman dibuka Excel
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader(); w.writerows(rows)


def align_to_pivot(pivot_words, other_words):
    """Peta indeks kata pivot -> kata provider lain (hanya equal / replace 1:1)."""
    m = {}
    for op, i1, i2, j1, j2 in SequenceMatcher(None, pivot_words, other_words,
                                              autojunk=False).get_opcodes():
        if op == "equal" or (op == "replace" and i2 - i1 == j2 - j1):
            for k in range(i2 - i1):
                m[i1 + k] = other_words[j1 + k]
    return m


def compute_epr(ocr_text: str, err_rows: list[dict]) -> dict:
    """EPR sederhana berbasis kantong-kata: tiap error pelajar (wrong->correct)
    dicek apakah bentuk SALAH masih ada (dipertahankan), berubah ke bentuk BENAR
    (dinormalisasi OCR), atau hilang (salah baca lain)."""
    bag = Counter(words(ocr_text))
    kept = normalized = lost = 0
    per_tag = defaultdict(lambda: Counter())
    for r in err_rows:
        wrong = norm_for_scoring(r.get("wrong", ""))
        right = norm_for_scoring(r.get("correct", ""))
        tag = (r.get("tag") or "-").strip() or "-"
        if wrong and bag[wrong] > 0:
            bag[wrong] -= 1; kept += 1; per_tag[tag]["kept"] += 1
        elif right and bag[right] > 0:
            bag[right] -= 1; normalized += 1; per_tag[tag]["normalized"] += 1
        else:
            lost += 1; per_tag[tag]["lost"] += 1
    n = kept + normalized + lost
    return {"n_errors": n, "kept": kept, "normalized": normalized, "lost": lost,
            "EPR": round(kept / n, 3) if n else "",
            "normalization_rate": round(normalized / n, 3) if n else "",
            "per_tag": {t: dict(c) for t, c in per_tag.items()}}


def make_report(args):
    out = Path(args.out)
    rep = out / "report"; rep.mkdir(parents=True, exist_ok=True)
    outs = load_outputs(out)
    if not outs:
        print("Belum ada hasil OCR. Jalankan dulu: python ocr_bench.py run ...")
        return
    provs = sorted(outs)
    docs = sorted({d for p in provs for d in outs[p]})

    # --- 7a. ringkasan + bendera masalah (tanpa GT)
    summary = []
    for p in provs:
        for d in docs:
            if d not in outs[p]:
                continue
            raw_f = out / "results" / "raw" / p / f"{d}.txt"
            raw = raw_f.read_text(encoding="utf-8") if raw_f.exists() else ""
            meta_f = out / "results" / "meta" / p / f"{d}.json"
            meta = json.loads(meta_f.read_text(encoding="utf-8")) if meta_f.exists() else {}
            t = outs[p][d]
            letters = len(ARABIC_LETTER_RE.findall(t))
            summary.append({
                "provider": p, "doc": d, "seconds": meta.get("seconds", ""),
                "n_chars": len(norm_for_scoring(t)), "n_words": len(words(t)),
                "html_tags_in_raw": len(HTML_TAG_RE.findall(raw)),
                "repetition_4gram": round(repetition_score(t), 3),
                "harakat_per_100_letters": round(100 * len(HARAKAT_RE.findall(t)) / max(1, letters), 2),
                "latin_chars": len(LATIN_RE.findall(t)),
                "illegible_marks": t.count("[؟]"),
            })
    write_csv(rep / "summary.csv", summary)

    # --- 7b. kesepakatan antar-OCR (CER simetris rata-rata; kecil = mirip)
    agree_rows, mean_dist = [], defaultdict(list)
    for a, b in itertools.combinations(provs, 2):
        ds = [d for d in docs if d in outs[a] and d in outs[b]]
        if not ds:
            continue
        vals = []
        for d in ds:
            ta, tb = norm_for_scoring(outs[a][d]), norm_for_scoring(outs[b][d])
            vals.append(levenshtein(ta, tb) / max(1, max(len(ta), len(tb))))
        m = sum(vals) / len(vals)
        agree_rows.append({"provider_A": a, "provider_B": b, "n_docs": len(ds),
                           "mean_char_distance": round(m, 3)})
        mean_dist[a].append(m); mean_dist[b].append(m)
    write_csv(rep / "agreement_pairs.csv", agree_rows)

    # --- 7c. profil ortografi per provider (sidik jari normalisasi)
    prof_rows = []
    for p in provs:
        allt = "\n".join(outs[p].get(d, "") for d in docs)
        prof_rows.append({"provider": p, **ortho_profile(allt)})
    write_csv(rep / "ortho_profile.csv", prof_rows)

    # --- 7d. lembar review ahli: posisi kata yang beda HANYA di huruf rawan
    pivot = args.pivot or (min(mean_dist, key=lambda k: sum(mean_dist[k]) / len(mean_dist[k]))
                           if mean_dist else provs[0])
    review = []
    for d in docs:
        if d not in outs.get(pivot, {}):
            continue
        pw = words(outs[pivot][d])
        maps = {p: align_to_pivot(pw, words(outs[p][d])) for p in provs if p != pivot and d in outs[p]}
        for i, w in enumerate(pw):
            aligned = {p: maps[p].get(i, "") for p in maps}
            cats = {ortho_category(w, v) for v in aligned.values() if v and v != w}
            cats.discard(None)
            if not cats:
                continue
            row = {"doc": d, "word_idx": i, "category": ",".join(sorted(cats)),
                   "context": " ".join(pw[max(0, i - 3): i + 4]), f"{pivot} (pivot)": w}
            row.update(aligned)
            row["EXPERT_what_is_written"] = ""   # diisi ahli sambil melihat gambar
            review.append(row)
    write_csv(rep / "review_orthographic_disagreements.csv", review)

    # --- 7e. skor CER/WER jika ada ground truth verbatim
    gt_dir = Path(args.gt_dir)
    score_rows = []
    if gt_dir.exists():
        for p in provs:
            for d in docs:
                g = gt_dir / f"{d}.txt"
                if g.exists() and d in outs[p]:
                    ref = g.read_text(encoding="utf-8")
                    score_rows.append({"provider": p, "doc": d,
                                       "CER": round(cer(outs[p][d], ref), 4),
                                       "WER": round(wer(outs[p][d], ref), 4)})
    write_csv(rep / "scores_cer_wer.csv", score_rows)

    # --- 7f. EPR jika ada daftar error pelajar
    err_dir = Path(args.errors_dir)
    epr_rows = []
    if err_dir.exists():
        for f in sorted(err_dir.glob("*.csv")):
            if f.stem.startswith("_"):
                continue
            with open(f, encoding="utf-8-sig") as fh:
                errs = [r for r in csv.DictReader(fh) if (r.get("wrong") or "").strip()]
            for p in provs:
                if f.stem in outs[p]:
                    e = compute_epr(outs[p][f.stem], errs)
                    pt = e.pop("per_tag")
                    epr_rows.append({"provider": p, "doc": f.stem, **e,
                                     "per_tag": json.dumps(pt, ensure_ascii=False)})
    write_csv(rep / "scores_epr.csv", epr_rows)

    # --- 7g. HTML berdampingan
    write_html(rep / "side_by_side.html", out, provs, docs, outs, summary, agree_rows,
               score_rows, epr_rows, pivot, len(review))

    # --- ringkasan konsol
    print("\n================ RINGKASAN ================")
    print(f"Provider: {', '.join(provs)} | Dokumen: {len(docs)} | pivot review: {pivot}")
    if score_rows:
        agg = defaultdict(list)
        for r in score_rows:
            agg[r["provider"]].append(r["CER"])
        print("CER rata-rata (vs ground truth, kecil = bagus):")
        for p, v in sorted(agg.items(), key=lambda kv: sum(kv[1]) / len(kv[1])):
            print(f"   {p:<18} {sum(v) / len(v):.3f}")
    else:
        print("Belum ada ground truth -> CER/WER belum dihitung.")
    if epr_rows:
        agg = defaultdict(lambda: [0, 0])
        for r in epr_rows:
            agg[r["provider"]][0] += r["kept"]; agg[r["provider"]][1] += r["n_errors"]
        print("EPR (error pelajar yang dipertahankan, besar = setia):")
        for p, (k, n) in sorted(agg.items(), key=lambda kv: -kv[1][0] / max(1, kv[1][1])):
            print(f"   {p:<18} {k / max(1, n):.3f}  ({k}/{n})")
    print(f"Baris review untuk ahli: {len(review)}")
    print(f"Laporan: {rep.resolve()}")


def write_html(path, out, provs, docs, outs, summary, agree, scores, eprs, pivot, n_review):
    esc = html.escape

    def table(rows):
        if not rows:
            return "<p><i>(kosong)</i></p>"
        keys = list(dict.fromkeys(k for r in rows for k in r))
        h = "".join(f"<th>{esc(str(k))}</th>" for k in keys)
        b = "".join("<tr>" + "".join(f"<td>{esc(str(r.get(k, '')))}</td>" for k in keys) + "</tr>"
                    for r in rows)
        return f"<table><tr>{h}</tr>{b}</table>"

    parts = [f"""<!doctype html><html lang="id"><head><meta charset="utf-8">
<title>OCR bench</title><style>
body{{font-family:system-ui,sans-serif;margin:16px;background:#fff;color:#111}}
table{{border-collapse:collapse;font-size:13px;margin:8px 0}} td,th{{border:1px solid #ccc;padding:3px 6px}}
.grid{{display:grid;grid-template-columns:repeat(auto-fill,minmax(320px,1fr));gap:10px}}
.card{{border:1px solid #ddd;border-radius:8px;padding:8px}}
.ar{{direction:rtl;text-align:right;white-space:pre-wrap;font-size:20px;line-height:1.9;
font-family:'Amiri','Noto Naskh Arabic','Traditional Arabic',serif}}
img.page{{max-width:100%;border:1px solid #ccc}}
</style></head><body><h1>Perbandingan OCR tulisan tangan Arab</h1>
<p>Provider: {esc(', '.join(provs))} — pivot lembar review: <b>{esc(pivot)}</b>
({n_review} baris di <code>review_orthographic_disagreements.csv</code>)</p>
<h2>CER/WER vs ground truth</h2>{table(scores)}
<h2>EPR</h2>{table([{k: v for k, v in r.items() if k != 'per_tag'} for r in eprs])}
<h2>Kesepakatan antar-OCR</h2>{table(agree)}
<h2>Ringkasan & bendera</h2>{table(summary)}"""]
    for d in docs:
        pages = sorted((out / "pages" / d).glob("p*.png"))
        imgs = "".join(f'<a href="../pages/{esc(d)}/{esc(p.name)}" target="_blank">'
                       f'<img class="page" src="../pages/{esc(d)}/{esc(p.name)}"></a>' for p in pages)
        cards = "".join(f'<div class="card"><b>{esc(p)}</b><div class="ar">{esc(outs[p].get(d, "—"))}</div></div>'
                        for p in provs)
        parts.append(f'<h2 id="{esc(d)}">{esc(d)}</h2><div class="grid"><div class="card">{imgs}</div>{cards}</div>')
    parts.append("</body></html>")
    path.write_text("".join(parts), encoding="utf-8")


# =============================================================================
# 8. CLI
# =============================================================================
def cmd_init(args):
    for d in (args.input, args.gt_dir, args.errors_dir, args.lines_manual, args.out):
        Path(d).mkdir(parents=True, exist_ok=True)
    (Path(args.gt_dir) / "_BACA_SAYA.txt").write_text(
        "Taruh transkripsi VERBATIM (apa adanya, termasuk salah ejaan pelajar) di sini.\n"
        "Nama file = nama esai tanpa ekstensi, mis. esai 'aisyah_t1.pdf' -> 'aisyah_t1.txt'.\n"
        "Encoding UTF-8. Tanpa harakat kecuali memang tertulis.\n", encoding="utf-8")
    tpl = Path(args.errors_dir) / "_TEMPLATE.csv"
    with open(tpl, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["wrong", "correct", "tag"])
        w.writerow(["مدرسه", "مدرسة", "OT"])
        w.writerow(["الى", "إلى", "OH"])
    env = Path(".env")
    if not env.exists():
        env.write_text("GEMINI_API_KEY=\nMISTRAL_API_KEY=\n", encoding="utf-8")
    print("Folder siap:")
    for d in (args.input, args.gt_dir, args.errors_dir, args.lines_manual, args.out):
        print("  ", Path(d).resolve())
    print("Isi API key di file .env, lalu taruh esai (PDF/JPG/PNG) di", Path(args.input).resolve())


def cmd_list(args):
    reg = build_registry(args.max_pixels)
    print("Nama               jenis  level     ikut 'all'")
    for n, p in reg.items():
        print(f"{p.describe()}  {'ya' if n in DEFAULT_ALL else '-'}")
    print("\nModel Gemini lain: pakai 'gemini:<model-id>', mis. --providers gemini:gemini-3.8-flash")


def cmd_segment(args):
    docs = list_docs(Path(args.input))
    for doc in docs:
        pages = load_pages(doc, Path(args.out), args.dpi)
        lines = get_lines(doc, pages, Path(args.out), Path(args.lines_manual))
        print(f"{doc.name}: {len(pages)} halaman, {len(lines)} baris -> "
              f"{(Path(args.out) / 'lines' / doc.stem).resolve()}")


def cmd_run(args):
    docs = list_docs(Path(args.input))
    if args.limit:
        docs = docs[:args.limit]
    if not docs:
        sys.exit(f"Tidak ada PDF/gambar di {Path(args.input).resolve()}")
    provs = resolve_providers(args.providers, build_registry(args.max_pixels))
    print(f"{len(docs)} dokumen x {len(provs)} OCR")
    # API dulu (cepat), lalu model lokal satu per satu agar VRAM tidak penuh
    for p in sorted(provs, key=lambda x: x.kind != "api"):
        run_provider(p, docs, args)
    make_report(args)


def main():
    ap = argparse.ArgumentParser(description="Uji banding OCR tulisan tangan Arab")
    ap.add_argument("command", choices=["init", "list", "run", "report", "segment"])
    ap.add_argument("--input", default="data", help="folder esai PDF/gambar (default: data)")
    ap.add_argument("--out", default="bench_out", help="folder hasil (default: bench_out)")
    ap.add_argument("--providers", default="all", help="'all' atau daftar dipisah koma")
    ap.add_argument("--gt-dir", default="ground_truth", help="transkripsi verbatim <doc>.txt")
    ap.add_argument("--errors-dir", default="learner_errors", help="daftar error <doc>.csv")
    ap.add_argument("--lines-manual", default="lines_manual",
                    help="potongan baris manual per dokumen (opsional)")
    ap.add_argument("--dpi", type=int, default=300, help="resolusi render PDF")
    ap.add_argument("--max-pixels", type=int, default=1280 * 28 * 28,
                    help="batas piksel gambar untuk VLM halaman (turunkan bila OOM)")
    ap.add_argument("--no-repeat-ngram", type=int, default=0,
                    help="cegah loop pengulangan VLM lokal (mis. 4). Default 0 = mati")
    ap.add_argument("--gemini-media-res", default="default",
                    choices=["default", "low", "medium", "high"],
                    help="resolusi gambar yang dibaca Gemini (coba 'high' untuk tulisan kecil)")
    ap.add_argument("--pivot", default="", help="provider acuan untuk lembar review")
    ap.add_argument("--limit", type=int, default=0, help="hanya N dokumen pertama (uji coba)")
    ap.add_argument("--force", action="store_true", help="ulang walau hasil sudah ada")
    args = ap.parse_args()
    {"init": cmd_init, "list": cmd_list, "run": cmd_run,
     "report": make_report, "segment": cmd_segment}[args.command](args)


if __name__ == "__main__":
    main()
# -*- coding: utf-8 -*-
"""
ocr_compare.py — Bandingkan 3 OCR open-source (EasyOCR, Surya, QARI-OCR) pada
gambar/PDF tulisan tangan Arab. Fokus: mana yang AKURAT sekaligus SETIA
(tidak diam-diam mengoreksi ejaan) untuk data GEC.

CARA PAKAI (jalankan di venv terpisah, JANGAN di venv311 aplikasi):
    python -m venv venv_ocr
    venv_ocr\\Scripts\\Activate.ps1
    pip install --upgrade pip
    pip install pymupdf pillow numpy
    pip install torch torchvision            # CPU cukup; GPU jauh lebih cepat
    pip install easyocr
    pip install "transformers>=4.49" qwen-vl-utils accelerate   # untuk QARI
    pip install surya-ocr                      # untuk Surya (opsional)

    python ocr_compare.py "sample.pdf"
    # atau: python ocr_compare.py "gambar.jpg" --engines easyocr,qari

Catatan:
- QARI butuh transformers >= 4.49 (Qwen2-VL). Itu sebabnya pakai venv terpisah,
  supaya tidak mengganggu transformers==4.44.2 di venv311 (aplikasi Django).
- Di CPU, QARI ~beberapa menit per halaman. Uji 1 halaman dulu.
- Ambil 1 PDF sampel dari Drive Anda dulu, mis:
  "Perkenalan diri dan Motivasi Belajar -<nama> 1.pdf"
"""

import sys
import io
import argparse

FAITHFUL_PROMPT = (
    "Transcribe the Arabic text in this image EXACTLY as written, character by "
    "character, INCLUDING any spelling/imla mistakes. Do NOT correct, normalize, "
    "or complete anything. Output ONLY the raw transcribed text."
)


def load_images(path, dpi=200):
    """PDF -> list gambar PIL (per halaman). Gambar -> [gambar]."""
    from PIL import Image
    if path.lower().endswith(".pdf"):
        import fitz  # PyMuPDF
        doc = fitz.open(path)
        imgs = []
        for page in doc:
            pix = page.get_pixmap(dpi=dpi)
            imgs.append(Image.open(io.BytesIO(pix.tobytes("png"))).convert("RGB"))
        doc.close()
        return imgs
    return [Image.open(path).convert("RGB")]


# ---------------------------------------------------------------------------
def run_easyocr(imgs):
    import numpy as np
    import easyocr
    import torch
    reader = easyocr.Reader(["ar"], gpu=torch.cuda.is_available())
    parts = []
    for im in imgs:
        res = reader.readtext(np.array(im), detail=0, paragraph=True)
        parts.append("\n".join(res))
    return "\n---\n".join(parts)


def run_surya(imgs):
    from surya.recognition import RecognitionPredictor
    from surya.detection import DetectionPredictor
    rec = RecognitionPredictor()
    det = DetectionPredictor()
    parts = []
    for im in imgs:
        try:
            preds = rec([im], [["ar"]], det)          # API lama
        except TypeError:
            try:
                preds = rec([im], det_predictor=det)  # API baru
            except TypeError:
                preds = rec([im])                     # Surya 2 (VLM tunggal)
        p = preds[0]
        lines = getattr(p, "text_lines", None)
        parts.append("\n".join(getattr(l, "text", "") for l in lines) if lines else str(p))
    return "\n---\n".join(parts)


def run_qari(imgs, model_id="NAMAA-Space/Qari-OCR-v0.3-VL-2B-Instruct"):
    import torch
    from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
    from qwen_vl_utils import process_vision_info
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        model_id, torch_dtype="auto", device_map="auto")
    proc = AutoProcessor.from_pretrained(model_id)
    parts = []
    for im in imgs:
        messages = [{"role": "user", "content": [
            {"type": "image", "image": im},
            {"type": "text", "text": FAITHFUL_PROMPT}]}]
        text = proc.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = proc(text=[text], images=image_inputs, videos=video_inputs,
                      padding=True, return_tensors="pt").to(model.device)
        gen = model.generate(**inputs, max_new_tokens=1024)
        trimmed = [o[len(i):] for i, o in zip(inputs.input_ids, gen)]
        parts.append(proc.batch_decode(trimmed, skip_special_tokens=True)[0])
    return "\n---\n".join(parts)


ENGINES = {"easyocr": run_easyocr, "surya": run_surya, "qari": run_qari}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path", help="PDF atau gambar")
    ap.add_argument("--engines", default="easyocr,surya,qari",
                    help="dipisah koma: easyocr,surya,qari")
    ap.add_argument("--dpi", type=int, default=200)
    args = ap.parse_args()

    print(f"Memuat: {args.path}")
    imgs = load_images(args.path, dpi=args.dpi)
    print(f"Jumlah halaman/gambar: {len(imgs)}\n")

    for name in [e.strip() for e in args.engines.split(",") if e.strip()]:
        fn = ENGINES.get(name)
        if not fn:
            print(f"[{name}] tidak dikenal, dilewati.\n"); continue
        print("=" * 70)
        print(f"OCR: {name.upper()}")
        print("=" * 70)
        try:
            print(fn(imgs))
        except Exception as exc:
            print(f"[GAGAL] {name}: {exc}")
        print()


if __name__ == "__main__":
    main()

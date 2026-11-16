# Uji end-to-end: OCR (sample.pdf) -> AraT5 -> ARETA -> filter scope. Hanya membaca, tidak mengubah apa pun.
import os, sys, time, json
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "gramatical.settings")
import django; django.setup()
import torch
from home import ocr, gec_model, views

print("[0] torch", torch.__version__, "| cuda:", torch.cuda.is_available(),
      torch.cuda.get_device_name(0) if torch.cuda.is_available() else "")
print("[0] tagger env:", os.environ.get("IMLASCAN_TAGGER", "(default areta)"),
      "| ocr provider:", ocr._provider(), "| ocr configured:", ocr.is_configured())

t = time.time()
try:
    with open("sample.pdf", "rb") as f:
        text = ocr.extract_text(f.read(), "sample.pdf")
    print(f"[1] OCR OK ({time.time()-t:.1f}s), {len(text)} char")
    print("    cuplikan:", text[:200].replace("\n", " "))
except Exception as e:
    print("[1] OCR GAGAL:", type(e).__name__, str(e)[:300])
    text = "ذهب الطالب الي المدرسة و احب الاستاذ"
    print("    pakai teks cadangan:", text)

t = time.time()
res = gec_model.analyze_with_model(text[:600])
print(f"[2] analyze_with_model ({time.time()-t:.1f}s):", "None (FALLBACK lexicon!)" if res is None else "OK")
if res is not None:
    print("    LAST_TAGGER:", gec_model.LAST_TAGGER, "| jumlah koreksi mentah:", len(res))
    from collections import Counter
    print("    distribusi tag mentah:", dict(Counter(r["tag"] for r in res)))

out = views.analyze_imla(text[:600])
print("[3] analyze_imla tagger:", out.get("tagger"), "| total_errors (setelah filter scope):", out["total_errors"])
with open("_e2e_result.json", "w", encoding="utf-8") as f:
    json.dump({"ocr_text": text, "raw": res, "final": out}, f, ensure_ascii=False, indent=1)
print("    detail disimpan ke _e2e_result.json")

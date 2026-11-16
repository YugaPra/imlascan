"""OCR batch untuk korpus riset: data_siap/halaman/* -> data_siap/ocr/<halaman>.json

- Cache: halaman yang sudah punya .json dilewati (aman dijalankan ulang).
- Retry: 429/5xx DAN timeout/koneksi putus, backoff 5-10-20-40-60 dtk.
- Log per halaman: versi model asli (modelVersion), durasi, token, versi prompt.
Pakai:  python pipeline/ocr_batch.py --files P02_E02_H1.png P13_E01_H1.png
        python pipeline/ocr_batch.py            (semua halaman)
"""
import argparse, base64, json, os, sys, time
import requests

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IMG_DIR = os.path.join(BASE, "data_siap", "halaman")
OUT_DIR = os.path.join(BASE, "data_siap", "ocr")
URL = "https://generativelanguage.googleapis.com/v1beta/models/{m}:generateContent"

THINKING = "low"
PROMPT_VERSION = "ocr-v2"
PROMPT = (
    "You are a strict OCR engine for a learner-error corpus. Transcribe the handwritten Arabic "
    "text in this image EXACTLY as written, character by character, INCLUDING every spelling, "
    "hamza, ta marbuta/ha, alif and grammar mistake. Do NOT correct, normalize, complete or "
    "improve anything. Do NOT add diacritics (tashkil) that are not written. "
    "HAMZA: write أ إ آ ؤ ئ ء ONLY where the writer actually drew the hamza mark; a bare alif "
    "stays ا (e.g. if the writer wrote الاسلامية, output الاسلامية, not الإسلامية). "
    "SPACING: keep word spacing exactly as written; if the conjunction و is written apart from "
    "the next word (و سنة), keep the space; never join or split words yourself. "
    "If the writer crossed out / struck through a word, put it inside ⟦ ⟧ (e.g. ⟦كلمة⟧) and keep "
    "the replacement word after it. If a word is unreadable, write [?]. Ignore printed page "
    "headers (No., Date) and scanner watermarks. Preserve line breaks. "
    "Output ONLY the raw transcription, no explanation, no markdown."
)

def load_env():
    env = {}
    with open(os.path.join(BASE, ".env"), encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1); env[k.strip()] = v.strip()
    return env

def ocr_one(path, key, model):
    mime = "image/png" if path.lower().endswith(".png") else "image/jpeg"
    payload = {"contents": [{"parts": [
        {"text": PROMPT},
        {"inline_data": {"mime_type": mime, "data": base64.b64encode(open(path, "rb").read()).decode()}},
    ]}], "generationConfig": {"temperature": 0,
                              # gemini-3.8-flash: thinking bawaan >240 dtk/halaman; "low" ~6-7 dtk
                              "thinkingConfig": {"thinkingLevel": THINKING}}}
    last = ""
    for attempt, wait in enumerate([5, 10, 20, 40, 60, None]):
        t0 = time.time()
        try:
            r = requests.post(URL.format(m=model), params={"key": key}, json=payload, timeout=(15, 240))
            if r.status_code == 200:
                d = r.json()
                parts = d["candidates"][0].get("content", {}).get("parts", [])
                text = "".join(p.get("text", "") for p in parts).strip()
                return {"text": text, "model_requested": model, "model_version": d.get("modelVersion"),
                        "finish_reason": d["candidates"][0].get("finishReason"),
                        "usage": d.get("usageMetadata"), "seconds": round(time.time() - t0, 1),
                        "attempts": attempt + 1, "prompt_version": PROMPT_VERSION, "thinking": THINKING,
                        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S")}
            last = f"HTTP {r.status_code}: {r.text[:200]}"
            if r.status_code not in (429, 500, 502, 503, 504):
                raise RuntimeError(last)                     # error permanen: jangan retry
        except (requests.Timeout, requests.ConnectionError) as e:
            last = f"{type(e).__name__} setelah {time.time() - t0:.0f}s"
        if wait is None: break
        print(f"   retry {attempt + 1} ({last[:80]}) tunggu {wait}s", flush=True)
        time.sleep(wait)
    raise RuntimeError(f"gagal setelah retry: {last}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--files", nargs="*")
    ap.add_argument("--sleep", type=float, default=4.0, help="jeda antar halaman (rate limit)")
    a = ap.parse_args()
    env = load_env()
    key, model = env["IMLASCAN_GEMINI_KEY"], env.get("IMLASCAN_GEMINI_MODEL", "gemini-flash-latest")
    os.makedirs(OUT_DIR, exist_ok=True)
    files = a.files or sorted(f for f in os.listdir(IMG_DIR) if not f.startswith("_"))
    ok = fail = skip = 0
    for i, fn in enumerate(files, 1):
        out = os.path.join(OUT_DIR, os.path.splitext(fn)[0] + ".json")
        if os.path.exists(out): skip += 1; continue
        print(f"[{i}/{len(files)}] {fn}", flush=True)
        try:
            res = ocr_one(os.path.join(IMG_DIR, fn), key, model)
            res["file"] = fn
            with open(out, "w", encoding="utf-8") as f: json.dump(res, f, ensure_ascii=False, indent=1)
            u = res.get("usage") or {}
            print(f"   OK {res['seconds']}s, {len(res['text'])} char, model={res['model_version']}, "
                  f"thoughts={u.get('thoughtsTokenCount', 0)}", flush=True); ok += 1
        except Exception as e:
            print(f"   GAGAL: {e}", flush=True); fail += 1
            with open(os.path.join(OUT_DIR, "_gagal.log"), "a", encoding="utf-8") as f:
                f.write(f"{time.strftime('%H:%M:%S')}\t{fn}\t{e}\n")
        time.sleep(a.sleep)
    print(f"SELESAI ok={ok} gagal={fail} dilewati(cache)={skip}", flush=True)

if __name__ == "__main__":
    main()

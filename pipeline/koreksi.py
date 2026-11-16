"""Tahap 3: OCR -> kalimat (dipecah LOKAL) -> koreksi Gemini (silver) -> pasangan via ID.

Kenapa dipecah lokal: sisi "salah" (src) harus persis teks OCR. Gemini hanya mengisi sisi
"benar" (tgt) untuk ID yang sama, jadi tidak mungkin terjadi salah-pasang seperti bug zip().
Aturan coretan siswa (⟦...⟧): DIBUANG dari src (yang dianalisis = versi final pilihan siswa).
Output: data_siap/pilot30_pairs.jsonl  (cache per esai di data_siap/koreksi/)
"""
import csv, json, os, re, sys, time
from collections import defaultdict
import requests
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ocr_batch import load_env, URL, BASE

OCR_DIR = os.path.join(BASE, "data_siap", "ocr")
CACHE = os.path.join(BASE, "data_siap", "koreksi")
OUT = os.path.join(BASE, "data_siap", "pilot30_pairs.jsonl")
THINKING = "low"
PROMPT_VERSION = "kor-v1"
PROMPT = """You are an experienced Arabic teacher correcting sentences written by Indonesian
university students learning Arabic (L2). For EACH sentence, produce the corrected version with
MINIMAL EDITS:
- Fix spelling/orthography (hamza, ta marbuta/ha, alif maqsura/ya, alif, madda, letters),
  morphology (gender/number agreement, verb conjugation, definiteness) and clear grammar errors
  (prepositions, case endings that are written, missing/extra words that make it ungrammatical).
- Fix word spacing only when wrong (e.g. a detached و must be attached: "و احب" -> "وأحب").
- Do NOT rephrase, do NOT change vocabulary, word order or style if it is already acceptable,
  do NOT add new content, do NOT add diacritics. Keep the original punctuation unless it is wrong.
- If a sentence is already correct, return it UNCHANGED.
Return ONLY a JSON array: [{"id": "...", "tgt": "..."}] with exactly one item per input id.
Input sentences (JSON):
"""

def clean(text):
    text = re.sub(r"⟦[^⟧]*⟧", " ", text)              # buang kata yang dicoret siswa
    text = re.sub(r"[ \t]+", " ", text)
    return text

def segment(text):
    paras = [p for p in re.split(r"\n\s*\n", text) if p.strip()]
    sents = []
    for p in paras:
        p = re.sub(r"\s*\n\s*", " ", p).strip()      # baris baru di tengah paragraf = spasi
        for s in re.split(r"(?<=[.!?؟])\s+", p):
            s = s.strip()
            if len(s.split()) >= 2:                   # buang serpihan 1 kata (judul tetap jika >=2 kata)
                sents.append(s)
    return sents

def call(key, model, items):
    body = {"contents": [{"parts": [{"text": PROMPT + json.dumps(items, ensure_ascii=False)}]}],
            "generationConfig": {"temperature": 0, "responseMimeType": "application/json",
                                 "thinkingConfig": {"thinkingLevel": THINKING}}}
    last = ""
    for wait in [5, 10, 20, 40, None][-int(os.environ.get("KOR_TRIES", "5")):]:
        try:
            r = requests.post(URL.format(m=model), params={"key": key}, json=body, timeout=(15, int(os.environ.get("KOR_READ_TIMEOUT", "240"))))
            if r.status_code == 200:
                d = r.json()
                txt = "".join(p.get("text", "") for p in d["candidates"][0]["content"]["parts"])
                return json.loads(txt), d.get("modelVersion")
            last = f"HTTP {r.status_code}: {r.text[:150]}"
            if r.status_code not in (429, 500, 502, 503, 504): break
        except (requests.Timeout, requests.ConnectionError, json.JSONDecodeError, KeyError) as e:
            last = f"{type(e).__name__}: {str(e)[:100]}"
        if wait is None: break
        print("   retry:", last[:90], flush=True); time.sleep(wait)
    raise RuntimeError(last)

def main():
    env = load_env(); key = env["IMLASCAN_GEMINI_KEY"]
    model = env.get("IMLASCAN_GEMINI_MODEL", "gemini-flash-latest")
    os.makedirs(CACHE, exist_ok=True)
    rows = list(csv.DictReader(open(os.path.join(BASE, "data_siap", "pilot30.csv"), encoding="utf-8-sig")))
    pages = defaultdict(list)
    for r in rows: pages[r["esai_id"]].append(r)
    out, problems = [], []
    for eid, rs in sorted(pages.items()):
        rs.sort(key=lambda r: int(r["halaman"]))
        texts = []
        for r in rs:
            p = os.path.join(OCR_DIR, os.path.splitext(r["file"])[0] + ".json")
            if not os.path.exists(p): problems.append(f"{eid}: OCR {r['file']} belum ada"); texts = None; break
            texts.append(json.load(open(p, encoding="utf-8"))["text"])
        if texts is None: continue
        sents = segment(clean("\n\n".join(texts)))
        items = [{"id": f"{eid}_S{i+1:02d}", "src": s} for i, s in enumerate(sents)]
        cpath = os.path.join(CACHE, eid + ".json")
        if os.path.exists(cpath):
            res = json.load(open(cpath, encoding="utf-8"))
        else:
            t0 = time.time()
            try:
                tg, mv = call(key, model, items)
            except Exception as e:
                problems.append(f"{eid}: GAGAL {e}"); print(eid, "GAGAL", e, flush=True); continue
            res = {"items": items, "tgt": tg, "model_version": mv, "prompt_version": PROMPT_VERSION,
                   "thinking": THINKING, "seconds": round(time.time() - t0, 1)}
            json.dump(res, open(cpath, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
            print(f"{eid}: {len(items)} kalimat, {res['seconds']}s", flush=True)
            time.sleep(3)
        tgt = {t.get("id"): t.get("tgt", "") for t in res["tgt"] if isinstance(t, dict)}
        miss = [it["id"] for it in res["items"] if not tgt.get(it["id"], "").strip()]
        extra = set(tgt) - {it["id"] for it in res["items"]}
        if miss or extra: problems.append(f"{eid}: id hilang={miss} id asing={sorted(extra)}")
        for it in res["items"]:
            if it["id"] in miss: continue
            out.append({"id": it["id"], "esai_id": eid, "penulis": rs[0]["penulis"], "split": rs[0]["split"],
                        "src": it["src"], "tgt": tgt[it["id"]].strip(),
                        "model_version": res["model_version"], "prompt_version": res["prompt_version"]})
    with open(OUT, "w", encoding="utf-8") as f:
        for o in out: f.write(json.dumps(o, ensure_ascii=False) + "\n")
    print(f"SELESAI pasangan={len(out)} esai={len({o['esai_id'] for o in out})}")
    for p in problems: print("  !", p)

if __name__ == "__main__":
    main()

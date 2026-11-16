"""Tahap 4: saring pasangan + ekspor dataset training.

Aturan buang (dicatat di dataset/dibuang.jsonl beserta alasannya):
  R1  src mengandung [?] (kata tak terbaca OCR)
  R2  kemiripan karakter src-tgt < 0.70   -> indikasi Gemini menulis ulang, bukan mengoreksi
  R3  rasio jumlah kata tgt/src di luar 0.75-1.35
  R4  > 64 kata (terlalu panjang utk max 256 token dgn aman)
  R5  duplikat (src sama persis)
Pasangan yang TIDAK berubah (src == tgt) SENGAJA dipertahankan: model harus belajar diam
saat kalimat sudah benar (obat over-correction).
Output: dataset/{train,val,test}.jsonl, dataset/datacard.json, dataset/dibuang.jsonl
"""
import difflib, json, os, sys
from collections import Counter

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(BASE, "data_siap", "pilot30_pairs.jsonl")
OUT = os.path.join(BASE, "data_siap", "dataset_pilot30")

import re
HARAKAT = re.compile(r"[ً-ْٰـ]")   # tasykil + tatwil
def norm(s):
    # Normalisasi yang SAMA harus dipakai saat inferensi (ImlaScan):
    #  - buang harakat (162/292 src berharakat, Gemini membuangnya tak konsisten -> bukan sinyal imla)
    #  - rapikan spasi sebelum tanda baca ("كلمة ." -> "كلمة.")
    s = HARAKAT.sub("", s)
    s = re.sub(r"\s+([.,،؛:!?؟])", r"\1", s)
    return re.sub(r"\s+", " ", s).strip()

def why(p, seen):
    s, t = p["src"], p["tgt"]
    if "[?]" in s: return "R1 kata tak terbaca"
    sim = difflib.SequenceMatcher(None, s, t, autojunk=False).ratio()
    if sim < 0.70: return f"R2 rewriting (sim={sim:.2f})"
    r = len(t.split()) / max(1, len(s.split()))
    if not 0.75 <= r <= 1.35: return f"R3 rasio kata {r:.2f}"
    if len(s.split()) > 64: return "R4 terlalu panjang"
    if s in seen: return "R5 duplikat"
    return None

def main():
    pairs = [json.loads(l) for l in open(SRC, encoding="utf-8")]
    for p in pairs:
        p["src_asli"], p["tgt_asli"] = p["src"], p["tgt"]
        p["src"], p["tgt"] = norm(p["src"]), norm(p["tgt"])
    os.makedirs(OUT, exist_ok=True)
    keep, drop, seen = {"train": [], "val": [], "test": []}, [], set()
    for p in pairs:
        w = why(p, seen)
        if w: drop.append({**p, "alasan": w}); continue
        seen.add(p["src"]); keep[p["split"]].append(p)
    for sp, rows in keep.items():
        with open(os.path.join(OUT, f"{sp}.jsonl"), "w", encoding="utf-8") as f:
            for r in rows: f.write(json.dumps({"id": r["id"], "src": r["src"], "tgt": r["tgt"]}, ensure_ascii=False) + "\n")
    with open(os.path.join(OUT, "dibuang.jsonl"), "w", encoding="utf-8") as f:
        for r in drop: f.write(json.dumps(r, ensure_ascii=False) + "\n")

    def stat(rows):
        ch = sum(r["src"] != r["tgt"] for r in rows)
        wsrc = sum(len(r["src"].split()) for r in rows)
        ed = 0
        for r in rows:
            sm = difflib.SequenceMatcher(None, r["src"].split(), r["tgt"].split(), autojunk=False)
            ed += sum(max(i2 - i1, j2 - j1) for op, i1, i2, j1, j2 in sm.get_opcodes() if op != "equal")
        return {"pasangan": len(rows), "berubah": ch, "tidak_berubah": len(rows) - ch,
                "penulis": sorted({r["penulis"] for r in rows}), "esai": len({r["esai_id"] for r in rows}),
                "kata_src": wsrc, "kata_diubah": ed, "rasio_kata_diubah": round(ed / max(1, wsrc), 3)}
    card = {"nama": "pilot30", "sumber": "esai tulisan tangan L2 (UHAMKA/UNJ), OCR gemini-3.8-flash",
            "label": "SILVER (koreksi Gemini, belum divalidasi ahli)",
            "versi_prompt": sorted({p["prompt_version"] for p in pairs}),
            "versi_model_koreksi": sorted({str(p["model_version"]) for p in pairs}),
            "split_per_penulis": True,
            "normalisasi": "harakat+tatwil dibuang, spasi sebelum tanda baca dirapikan (src & tgt)",
            "statistik": {sp: stat(rows) for sp, rows in keep.items()},
            "dibuang": dict(Counter(d["alasan"].split(" ")[0] for d in drop)), "total_awal": len(pairs)}
    json.dump(card, open(os.path.join(OUT, "datacard.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    # cek kebocoran: tidak boleh ada penulis yang muncul di >1 split
    owners = {}
    for sp, rows in keep.items():
        for r in rows: owners.setdefault(r["penulis"], set()).add(sp)
    leak = {p: s for p, s in owners.items() if len(s) > 1}
    print(json.dumps(card["statistik"], ensure_ascii=False, indent=1))
    print("dibuang:", card["dibuang"], "dari", len(pairs), "| kebocoran penulis:", leak or "TIDAK ADA")

if __name__ == "__main__":
    main()

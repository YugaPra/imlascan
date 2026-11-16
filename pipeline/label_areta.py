"""Label ARETA untuk dataset pasangan (src, tgt) + distribusi tag/kelas.
Pakai: python pipeline/label_areta.py <jsonl...> --out nama [--sample N]
Hasil: pipeline/_areta_<nama>.jsonl (per kalimat, cache) dan _areta_<nama>_ringkasan.json
"""
import argparse, json, os, random, sys, time
from collections import Counter
HERE = os.path.dirname(os.path.abspath(__file__)); BASE = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(BASE, "areta_service"))
import areta_client as ac

KELAS = {"O": "Ortografi", "M": "Morfologi", "X": "Sintaksis", "S": "Semantik", "P": "Tanda baca"}
def kelas(tag):
    if tag in ("MG", "SP"): return "Merge/Split"
    if tag.startswith("UNK"): return "UNK"          # label kelas saja (O/M/X/S) tetap masuk kelasnya
    return KELAS.get(tag[0], "Lain")

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("files", nargs="+"); ap.add_argument("--out", required=True)
    ap.add_argument("--sample", type=int, default=0); ap.add_argument("--batch", type=int, default=40)
    a = ap.parse_args()
    rows = [json.loads(l) for f in a.files for l in open(f, encoding="utf-8") if l.strip()]
    if a.sample and a.sample < len(rows): rows = random.Random(42).sample(rows, a.sample)
    cache_p = os.path.join(HERE, f"_areta_{a.out}.jsonl")
    done = {}
    if os.path.exists(cache_p):
        for l in open(cache_p, encoding="utf-8"): d = json.loads(l); done[d["id"]] = d
    todo = [r for r in rows if r["id"] not in done]
    assert ac.is_up(), "Layanan ARETA belum jalan (jalankan_areta.bat)"
    t0 = time.time()
    with open(cache_p, "a", encoding="utf-8") as f:
        for i in range(0, len(todo), a.batch):
            b = todo[i:i + a.batch]
            res = ac.annotate([(r["src"], r["tgt"]) for r in b], timeout=600)
            for r, toks in zip(b, res):
                d = {"id": r["id"], "tags": [t["tag"] for t in toks]}
                f.write(json.dumps(d, ensure_ascii=False) + "\n"); done[r["id"]] = d
            el = time.time() - t0; n = i + len(b)
            print(f"  {n}/{len(todo)}  {el:.0f}s  (~{el / n:.2f}s/kalimat)", flush=True)
    tag_c, kelas_c, gab = Counter(), Counter(), 0
    for r in rows:
        for t in done[r["id"]]["tags"]:
            if t == "UC": continue
            parts = t.split("+"); gab += len(parts) > 1
            for p in parts: tag_c[p] += 1; kelas_c[kelas(p)] += 1
    tot = sum(tag_c.values())
    ring = {"kalimat": len(rows), "label_kesalahan": tot, "token_tag_gabungan": gab,
            "kelas_%": {k: round(100 * v / tot, 1) for k, v in kelas_c.most_common()},
            "tag_%": {k: round(100 * v / tot, 1) for k, v in tag_c.most_common()},
            "tag_jumlah": dict(tag_c.most_common())}
    json.dump(ring, open(os.path.join(HERE, f"_areta_{a.out}_ringkasan.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(json.dumps(ring, ensure_ascii=False))

if __name__ == "__main__":
    main()

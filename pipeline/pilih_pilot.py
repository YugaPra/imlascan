"""Pilot 30 esai: 10 penulis x 3 esai, split PER PENULIS (7 train / 1 val / 2 test).
Deterministik (seed tetap) supaya bisa direproduksi. Output: data_siap/pilot30.csv
"""
import csv, os, random
from collections import defaultdict

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SEED = 2026
ESAI = [2, 5, 8]  # 3 topik tersebar: latar belakang, motivasi masuk PT, kampung halaman

man = list(csv.DictReader(open(os.path.join(BASE, "data_siap", "manifest.csv"), encoding="utf-8-sig")))
esai_per = defaultdict(set)
for m in man:
    esai_per[m["penulis"]].add(int(m["esai_id"].split("_E")[1]))
lengkap = sorted(p for p, s in esai_per.items() if set(range(1, 11)) <= s)  # penulis dgn 10 esai
rng = random.Random(SEED)
dipilih = sorted(rng.sample(lengkap, 10))
urut = dipilih[:]; rng.shuffle(urut)
split = {p: ("train" if i < 7 else "val" if i == 7 else "test") for i, p in enumerate(urut)}

rows = []
for m in man:
    n = int(m["esai_id"].split("_E")[1])
    if m["penulis"] in split and n in ESAI:
        rows.append({**m, "split": split[m["penulis"]]})
with open(os.path.join(BASE, "data_siap", "pilot30.csv"), "w", newline="", encoding="utf-8-sig") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)

print("penulis lengkap:", len(lengkap), lengkap)
for s in ("train", "val", "test"):
    ps = sorted(p for p in split if split[p] == s)
    print(f"{s:5}: penulis={ps}  esai={len({r['esai_id'] for r in rows if r['split']==s})}  "
          f"halaman={sum(1 for r in rows if r['split']==s)}")
print("FILES", " ".join(r["file"] for r in rows))

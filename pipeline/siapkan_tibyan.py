"""Siapkan Tibyan (final, setelah anotasi manusia) untuk training tahap-1.

1. Buang penanda "S" di awal baris, normalisasi PERSIS seperti data kita (saring.norm:
   harakat dibuang, spasi sebelum tanda baca dirapikan).
2. Paragraf -> kalimat yang SEJAJAR: kata sisi-salah dan sisi-benar dijajarkan (difflib),
   lalu dipotong di akhir kalimat sisi-benar (. ! ? ؟) HANYA bila titik potong jatuh di
   bagian yang identik (aman). Titik potong yang ambigu dilewati (kalimat digabung).
3. Saring dgn aturan yang sama: R2 kemiripan<0.70, R3 rasio kata, R4 >64 kata, R5 duplikat.
4. Split train/val 95/5 (seed tetap). TIDAK ada test: test = data L2 kita.
Output: data_eksternal/tibyan/dataset_tibyan/{train,val}.jsonl + datacard.json
"""
import difflib, json, os, random, re, sys
from collections import Counter
HERE = os.path.dirname(os.path.abspath(__file__)); BASE = os.path.dirname(HERE)
sys.path.insert(0, HERE)
from saring import norm, why                      # normalisasi & aturan saring yang sama

D = os.path.join(BASE, "data_eksternal", "tibyan", "Data", "Final-Data-after-human-annotation")
OUT = os.path.join(BASE, "data_eksternal", "tibyan", "dataset_tibyan")
END = re.compile(r"[.!?؟]$")

def strip_s(line):
    return re.sub(r"^\s*S\s+", "", line)

def split_aligned(src, tgt):
    # Penjajaran PER KARAKTER (sisi salah Tibyan sering menempelkan tanda baca: "للغايه؛حيث",
    # sehingga penjajaran per kata gagal). Potong di setiap akhir kalimat sisi-benar.
    ops = difflib.SequenceMatcher(None, src, tgt, autojunk=False).get_opcodes()
    def src_pos(p):                               # posisi di src yang sejajar dgn posisi p di tgt
        for op, i1, i2, j1, j2 in ops:
            if j1 <= p < j2 or (p == j2 and op == "equal"):
                if op == "equal" or (op == "replace" and i2 - i1 == j2 - j1):
                    return i1 + (p - j1)
                if op == "insert": return i1
                # ganti tak sama panjang, mis. "؛" (src) <-> ". " (tgt): potong di ujung blok src
                return i1 if p == j1 else i2
            if p == j2 and op != "equal": return i2
        return None
    cuts_t = [m.end() for m in re.finditer(r"[.!?؟](?=\s)", tgt)]
    pairs, si, ti = [], 0, 0
    for pt in cuts_t + [len(tgt)]:
        ps = src_pos(pt) if pt < len(tgt) else len(src)
        if ps is None or ps <= si: continue       # ambigu -> gabung dgn kalimat berikutnya
        a, b = src[si:ps].strip(), tgt[ti:pt].strip()
        if a and b: pairs.append((a, b))
        si, ti = ps, pt
    return pairs

def main():
    C = open(os.path.join(D, "Tibyan Correct.txt"), encoding="utf-8-sig").read().splitlines()
    I = open(os.path.join(D, "Tibyan Incorrect.txt"), encoding="utf-8-sig").read().splitlines()
    assert len(C) == len(I), (len(C), len(I))
    rows, drop, seen = [], Counter(), set()
    for k, (i, c) in enumerate(zip(I, C)):        # baris sejajar 1:1 (sudah dicek: 6192 = 6192)
        for n, (a, b) in enumerate(split_aligned(norm(strip_s(i)), norm(strip_s(c)))):
            p = {"id": f"TIB{k:05d}_S{n+1:02d}", "src": a, "tgt": b}
            w = why(p, seen)
            if w: drop[w.split(" ")[0]] += 1; continue
            seen.add(a); rows.append(p)
    rnd = random.Random(2026); para = sorted({r["id"][:8] for r in rows}); rnd.shuffle(para)
    val_para = set(para[: int(len(para) * 0.05)])  # split PER PARAGRAF agar tak bocor
    split = {"train": [r for r in rows if r["id"][:8] not in val_para],
             "val": [r for r in rows if r["id"][:8] in val_para]}
    os.makedirs(OUT, exist_ok=True)
    for sp, rs in split.items():
        with open(os.path.join(OUT, sp + ".jsonl"), "w", encoding="utf-8") as f:
            for r in rs: f.write(json.dumps(r, ensure_ascii=False) + "\n")
    ch = sum(r["src"] != r["tgt"] for r in rows)
    wl = sorted(len(r["src"].split()) for r in rows)
    card = {"nama": "tibyan_kalimat", "sumber": "github.com/Ahlam-alrehili/Tibyan-corpus Data.rar "
            "(Final-Data-after-human-annotation), SHA256 Data.rar A7F318D9C64D7D2C214A5F44EE515B70C7D1EE930178B4B6B00CF5C733B0DFDA",
            "lisensi": "TIDAK ADA file lisensi di repo -> pakai internal riset + sitasi; jangan disebarkan ulang",
            "paragraf": len(C), "pasangan_kalimat": len(rows), "berubah": ch, "tidak_berubah": len(rows) - ch,
            "kata_per_kalimat_median": wl[len(wl) // 2], "dibuang": dict(drop),
            "train": len(split["train"]), "val": len(split["val"]),
            "normalisasi": "sama dgn pilot30 (saring.norm)"}
    json.dump(card, open(os.path.join(OUT, "datacard.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(json.dumps(card, ensure_ascii=False, indent=1))
    for r in rows[:3] + rows[len(rows) // 2: len(rows) // 2 + 2]:
        print("\n", r["id"], "\n  S:", r["src"], "\n  T:", r["tgt"])

if __name__ == "__main__":
    main()

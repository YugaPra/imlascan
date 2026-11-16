"""Statistik deskriptif dataset L2 (pilot30) & Tibyan. Output: pipeline/_statistik.json"""
import difflib, json, os, re, statistics as st
from collections import Counter
from transformers import AutoTokenizer
HERE = os.path.dirname(os.path.abspath(__file__)); BASE = os.path.dirname(HERE)
tok = AutoTokenizer.from_pretrained("UBC-NLP/AraT5v2-base-1024")
HAMZA = set("أإآؤئء")

def kelas(a, b):        # jenis edit kasar (heuristik, BUKAN ARETA)
    if re.sub(r"[.,،؛:!?؟]", "", a) == re.sub(r"[.,،؛:!?؟]", "", b): return "tanda_baca"
    if a.replace(" ", "") == b.replace(" ", ""): return "spasi_gabung_pisah"
    norm = lambda s: re.sub("[أإآ]", "ا", s.replace("ة", "ه").replace("ى", "ي").replace("ؤ", "و").replace("ئ", "ي"))
    if norm(a) == norm(b):
        return "hamzah" if (set(a) | set(b)) & HAMZA else "ta_marbutah/alif_maqsurah"
    return "lainnya(kata/morf/sintaks)"

def stat(rows):
    w_src = [len(r["src"].split()) for r in rows]
    t_src = [len(tok(r["src"]).input_ids) for r in rows]
    t_tgt = [len(tok(r["tgt"]).input_ids) for r in rows]
    ed, kinds = 0, Counter()
    for r in rows:
        s, t = r["src"].split(), r["tgt"].split()
        for op, i1, i2, j1, j2 in difflib.SequenceMatcher(None, s, t, autojunk=False).get_opcodes():
            if op == "equal": continue
            if op == "replace" and i2 - i1 == j2 - j1:
                for k in range(i2 - i1):
                    if s[i1 + k] != t[j1 + k]: ed += 1; kinds[kelas(s[i1 + k], t[j1 + k])] += 1
            else:
                ed += max(i2 - i1, j2 - j1); kinds["sisip/hapus/ganti_blok"] += max(i2 - i1, j2 - j1)
    ch = sum(r["src"] != r["tgt"] for r in rows)
    return {"kalimat": len(rows), "kalimat_berubah": ch, "kalimat_tanpa_kesalahan_%": round(100 * (len(rows) - ch) / len(rows), 1),
            "kata_src_total": sum(w_src), "kata_per_kalimat_mean": round(st.mean(w_src), 1), "kata_per_kalimat_median": st.median(w_src),
            "token_src_total": sum(t_src), "token_tgt_total": sum(t_tgt), "token_per_kalimat_mean": round(st.mean(t_src), 1),
            "token_max": max(t_src + t_tgt), "edit_kata": ed, "kepadatan_edit_%kata": round(100 * ed / sum(w_src), 1),
            "edit_per_kalimat_berubah": round(ed / max(1, ch), 2),
            "jenis_edit_%": {k: round(100 * v / ed, 1) for k, v in kinds.most_common()}}

def baca(p): return [json.loads(l) for l in open(p, encoding="utf-8")]
out = {}
L2 = os.path.join(BASE, "data_siap", "dataset_pilot30")
pairs = {r["id"]: r for r in baca(os.path.join(BASE, "data_siap", "pilot30_pairs.jsonl"))}
for sp in ("train", "val", "test"):
    rows = baca(os.path.join(L2, sp + ".jsonl"))
    s = stat(rows)
    s["esai"] = len({r["id"].rsplit("_S", 1)[0] for r in rows}); s["penulis"] = sorted({r["id"][:3] for r in rows})
    out[f"L2_{sp}"] = s
allL2 = sum((baca(os.path.join(L2, sp + ".jsonl")) for sp in ("train", "val", "test")), [])
out["L2_total"] = {**stat(allL2), "esai": len({r["id"].rsplit("_S", 1)[0] for r in allL2}), "penulis": len({r["id"][:3] for r in allL2})}
T = os.path.join(BASE, "data_eksternal", "tibyan", "dataset_tibyan")
for sp in ("train", "val"):
    rows = baca(os.path.join(T, sp + ".jsonl"))
    s = stat(rows); s["paragraf"] = len({r["id"][:8] for r in rows}); out[f"Tibyan_{sp}"] = s
import random
tr = baca(os.path.join(T, "train.jsonl")); sub = random.Random(42).sample(tr, 6000)
s = stat(sub); s["paragraf"] = len({r["id"][:8] for r in sub}); out["Tibyan_subset6000_dipakai_ExpB"] = s
man = list(__import__("csv").DictReader(open(os.path.join(BASE, "data_siap", "manifest.csv"), encoding="utf-8-sig")))
out["korpus_L2_penuh"] = {"penulis": len({m["penulis"] for m in man}), "esai": len({m["esai_id"] for m in man}), "halaman": len(man)}
json.dump(out, open(os.path.join(HERE, "_statistik.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print("ok")

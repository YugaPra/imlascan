"""Bandingkan tagger heuristik ImlaScan (home/gec_model.infer_tag) vs ARETA pada pasangan L2.
Penjajaran kata: ARETA (token src->tgt). Untuk tiap token yang ARETA beri tag != UC dan src/tgt berupa
1 kata, panggil infer_tag(src, tgt) persis seperti ImlaScan (tanda baca dibuang dulu).
Output: pipeline/_banding_tagger.json + contoh ke _banding_tagger_contoh.txt
"""
import json, os, re, sys
from collections import Counter, defaultdict
import importlib.util as u
HERE = os.path.dirname(os.path.abspath(__file__)); BASE = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(BASE, "areta_service")); import areta_client as ac
spec = u.spec_from_file_location("gm", os.path.join(BASE, "home", "gec_model.py")); gm = u.module_from_spec(spec); spec.loader.exec_module(gm)
D = os.path.join(BASE, "data_siap", "dataset_pilot30")
rows = [json.loads(l) for sp in ("train", "val", "test") for l in open(os.path.join(D, sp + ".jsonl"), encoding="utf-8")]
res = []
for i in range(0, len(rows), 40):
    res += ac.annotate([(r["src"], r["tgt"]) for r in rows[i:i + 40]], timeout=600)
cls = lambda t: "Merge/Split" if t in ("MG", "SP") else ("UNK" if t.startswith("UNK") else {"O": "Orto", "M": "Morf", "X": "Sint", "S": "Sem", "P": "TB"}.get(t[:1], "?"))
n = same_exact = same_any = same_cls = heur_none = 0
conf = Counter(); per_areta = defaultdict(lambda: [0, 0, 0]); contoh = []
for toks in res:
    for t in toks:
        if t["tag"] == "UC": continue
        s, g = t["src"], t["tgt"]
        if not s or not g or " " in s.strip() or " " in g.strip(): continue   # hanya 1 kata <-> 1 kata
        w = re.sub(r"[^\w]", "", s, flags=re.UNICODE); c = re.sub(r"[^\w]", "", g, flags=re.UNICODE)
        if not w or not c or w == c: continue                                     # beda tanda baca saja -> ImlaScan abaikan
        h = gm.infer_tag(w, c) or "-"
        a = t["tag"]; a_parts, h_parts = set(a.split("+")), set(h.split("+"))
        n += 1; same_exact += (a == h); same_any += bool(a_parts & h_parts)
        same_cls += bool({cls(x) for x in a_parts} & {cls(x) for x in h_parts if x != "-"}); heur_none += (h == "-")
        conf[(a, h)] += 1
        pa = per_areta[a]; pa[0] += 1; pa[1] += (a == h); pa[2] += bool(a_parts & h_parts)
        if a != h and len(contoh) < 40: contoh.append(f"{s} -> {g} | ARETA={a} | heuristik={h}")
hasil = {"token_dibandingkan": n,
         "sama_persis_%": round(100 * same_exact / n, 1), "ada_tag_sama_%": round(100 * same_any / n, 1),
         "kelas_sama_%": round(100 * same_cls / n, 1), "heuristik_tanpa_tag_%": round(100 * heur_none / n, 1),
         "per_tag_ARETA (n, sama_persis_%, ada_irisan_%)": {k: [v[0], round(100 * v[1] / v[0]), round(100 * v[2] / v[0])]
                                                        for k, v in sorted(per_areta.items(), key=lambda x: -x[1][0])[:15]},
         "kebingungan_teratas (ARETA, heuristik, n)": [[a, h, c] for (a, h), c in conf.most_common(15)]}
json.dump(hasil, open(os.path.join(HERE, "_banding_tagger.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
open(os.path.join(HERE, "_banding_tagger_contoh.txt"), "w", encoding="utf-8").write("\n".join(contoh))
print("ok", n)

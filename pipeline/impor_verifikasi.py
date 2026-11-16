# -*- coding: utf-8 -*-
"""Tahap 5 (korpus v2): impor hasil verifikasi ahli -> dataset latih.

Masukan (hanya DIBACA, tidak pernah disimpan ulang):
  data_siap/v2/pairs_v2.jsonl                              metadata kalimat (penulis, esai, split)
  data_siap/v2/verifikasi_kembali/Verifikasi_P*.xlsx       18 file ahli, sheet "Verifikasi", kolom K-P
  .../_kembali_putaran2/Daftar_Perbaikan_Verifikasi_v2_KEMBALI.xlsx   tinjauan ulang (override via ID)

Aturan pembentukan pasangan (src = tulisan siswa, tgt = koreksi):
  src = kolom L (teks siswa versi ahli) bila K = Tidak, selain itu kolom G (teks OCR)
  tgt = kolom N (koreksi ahli) bila M = Ubah, selain itu kolom H (koreksi AI)
  Putaran 2 menimpa K/L/M/N/O HANYA pada baris yang HASIL TINJAUAN-nya diisi (Diperbaiki/Tetap).

Tingkat (kolom "tier"):
  A   OCR benar + koreksi AI disetujui ahli
  A+  koreksi ditulis ahli sendiri (M = Ubah, N terisi)
  B   OCR salah, src memakai teks siswa versi ahli, tgt tidak memuat kata salah-baca OCR

Tidak masuk train/val/test:
  ditahan.jsonl  T1  K = Tidak + Setuju dan tgt masih memuat kata salah-baca OCR
                     (kunci AI dibuat dari teks OCR yang keliru -> tgt bisa mengubah kata yang benar)
                 T2  K = Ya tetapi L terisi dan berbeda dari OCR (isian bertentangan)
  dibuang.jsonl  X1 Buang · X2 keputusan kosong · X3 Ubah tanpa koreksi ahli
                 R1-R5 = aturan pipeline/saring.py (kata [?], rewriting, rasio kata, >64 kata, duplikat)

Split: kolom Set yang sudah ada dipertahankan (val = P14, test = P15 & P17);
       penulis tanpa Set -> train. Tidak ada penulis di lebih dari satu split.

Jalankan:  python pipeline/impor_verifikasi.py      (butuh openpyxl)
Keluaran:  data_siap/dataset_v2/{train,val,test,ditahan,dibuang}.jsonl, log_impor.csv, datacard.json
"""
import csv, difflib, glob, json, os, re, sys, warnings
from collections import Counter, defaultdict

warnings.filterwarnings("ignore")
import openpyxl

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from saring import norm, why          # normalisasi & aturan R1-R5 yang SAMA dengan pilot30

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
V2 = os.path.join(BASE, "data_siap", "v2")
XL = os.path.join(V2, "verifikasi_kembali")
P2 = os.path.join(XL, "_kembali_putaran2", "Daftar_Perbaikan_Verifikasi_v2_KEMBALI.xlsx")
OUT = os.path.join(BASE, "data_siap", "dataset_v2")
PUNCT_AKHIR = ".!?؟"


def sel(v):
    return str(v).strip() if v is not None else ""


def baca_excel():
    """-> dict id -> isian ahli putaran 1."""
    data = {}
    for f in sorted(glob.glob(os.path.join(XL, "Verifikasi_P*.xlsx"))):
        ws = openpyxl.load_workbook(f, read_only=True, data_only=True)["Verifikasi"]
        for row in ws.iter_rows(min_row=6, values_only=True):
            row = list(row) + [None] * 19
            if not sel(row[1]):
                continue
            data[sel(row[1])] = {
                "file": os.path.basename(f), "set": sel(row[4]).lower(),
                "G": sel(row[6]), "H": sel(row[7]),
                "K": sel(row[10]).lower(), "L": sel(row[11]), "M": sel(row[12]).lower(),
                "N": sel(row[13]), "O": sel(row[14]).lower(), "P": sel(row[15]),
            }
    return data


def baca_putaran2():
    """-> dict id -> {kolom: nilai baru} hanya untuk baris yang ditinjau."""
    if not os.path.isfile(P2):
        return {}
    ws = openpyxl.load_workbook(P2, read_only=True, data_only=True)["Perbaikan"]
    out = {}
    for row in ws.iter_rows(min_row=2, values_only=True):
        row = [sel(c) for c in row] + [""] * 24
        if not row[5] or row[17].lower() not in ("diperbaiki", "tetap"):
            continue
        baru = {"K": row[18].lower(), "L": row[19], "M": row[20].lower(), "N": row[21], "O": row[22].lower()}
        out.setdefault(row[5], {}).update({k: v for k, v in baru.items() if v})
    return out


def kata_salah_baca(ocr, siswa):
    """Kata yang ada di teks OCR tetapi tidak di teks siswa versi ahli (posisi sejajar)."""
    a, b = norm(ocr).split(), norm(siswa).split()
    bersih = lambda w: re.sub(r"[^\w]", "", w, flags=re.UNICODE)
    hasil = set()
    for op, i1, i2, j1, j2 in difflib.SequenceMatcher(None, a, b, autojunk=False).get_opcodes():
        if op in ("replace", "delete"):
            hasil.update(bersih(w) for w in a[i1:i2])
    hasil.discard("")
    return hasil - {bersih(w) for w in b}


def main():
    meta = {}
    for l in open(os.path.join(V2, "pairs_v2.jsonl"), encoding="utf-8"):
        r = json.loads(l); meta[r["id"]] = r
    isian = baca_excel()
    p2 = baca_putaran2()
    assert set(isian) == set(meta), "ID Excel dan pairs_v2.jsonl tidak sama"

    log, keep, tahan, buang, seen = [], {"train": [], "val": [], "test": []}, [], [], set()
    for id_ in sorted(isian):
        d, m = dict(isian[id_]), meta[id_]
        for kol, baru in p2.get(id_, {}).items():           # override putaran 2
            if d[kol] != baru:
                log.append([id_, "putaran2", kol, d[kol], baru])
                d[kol] = baru
        split = d["set"] or "train"
        K, M = d["K"], d["M"]
        rec = {"id": id_, "penulis": m["penulis"], "esai_id": m["esai_id"], "split": split,
               "K": K, "M": M, "keyakinan": d["O"]}

        src_raw = d["L"] if (K == "tidak" and d["L"]) else d["G"]
        tgt_raw = d["N"] if (M == "ubah" and d["N"]) else d["H"]
        rec.update(src=norm(src_raw), tgt=norm(tgt_raw), ocr=norm(d["G"]), ai=norm(d["H"]))

        alasan = None
        if M == "buang": alasan = "X1 dibuang ahli"
        elif M not in ("setuju", "ubah"): alasan = "X2 keputusan kosong"
        elif M == "ubah" and not d["N"]: alasan = "X3 Ubah tanpa koreksi ahli"
        elif K == "tidak" and not d["L"]: alasan = "X4 OCR salah tanpa teks siswa"
        if alasan:
            buang.append({**rec, "alasan": alasan}); continue

        # koreksi ahli sering tanpa tanda baca akhir -> samakan dengan src (dicatat di log)
        if M == "ubah" and rec["tgt"] and rec["src"] and rec["src"][-1] in PUNCT_AKHIR \
                and rec["tgt"][-1] not in PUNCT_AKHIR + "،؛:,":
            log.append([id_, "tanda_baca", "tgt", rec["tgt"], rec["tgt"] + rec["src"][-1]])
            rec["tgt"] += rec["src"][-1]

        if K == "ya" and d["L"] and norm(d["L"]) != norm(d["G"]):
            tahan.append({**rec, "alasan": "T2 K=Ya tetapi L terisi & berbeda", "L": norm(d["L"])}); continue
        if K == "tidak" and M == "setuju":
            bocor = sorted(w for w in kata_salah_baca(d["G"], d["L"])
                           if w in {re.sub(r"[^\w]", "", x, flags=re.UNICODE) for x in rec["tgt"].split()})
            if bocor:
                tahan.append({**rec, "alasan": "T1 tgt memuat kata salah-baca OCR", "kata": bocor}); continue

        w = why(rec, seen)                                    # R1-R5
        if w:
            buang.append({**rec, "alasan": w}); continue
        seen.add(rec["src"])
        rec["tier"] = "A+" if M == "ubah" else ("B" if K == "tidak" else "A")
        keep[split].append(rec)

    os.makedirs(OUT, exist_ok=True)
    def tulis(nama, rows, kolom=None):
        with open(os.path.join(OUT, nama), "w", encoding="utf-8", newline="\n") as f:
            for r in rows:
                f.write(json.dumps({k: r[k] for k in kolom} if kolom else r, ensure_ascii=False) + "\n")
    for sp, rows in keep.items():
        tulis(f"{sp}.jsonl", rows, ["id", "src", "tgt", "tier", "penulis"])
    tulis("ditahan.jsonl", tahan); tulis("dibuang.jsonl", buang)
    with open(os.path.join(OUT, "log_impor.csv"), "w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f); w.writerow(["id", "sumber", "kolom", "sebelum", "sesudah"]); w.writerows(log)

    def stat(rows):
        ch = sum(r["src"] != r["tgt"] for r in rows)
        wsrc = sum(len(r["src"].split()) for r in rows); ed = 0
        for r in rows:
            sm = difflib.SequenceMatcher(None, r["src"].split(), r["tgt"].split(), autojunk=False)
            ed += sum(max(i2 - i1, j2 - j1) for op, i1, i2, j1, j2 in sm.get_opcodes() if op != "equal")
        return {"pasangan": len(rows), "berubah": ch, "tidak_berubah": len(rows) - ch,
                "persen_berubah": round(100 * ch / max(1, len(rows)), 1),
                "tier": dict(sorted(Counter(r["tier"] for r in rows).items())),
                "penulis": sorted({r["penulis"] for r in rows}), "esai": len({r["esai_id"] for r in rows}),
                "kata_src": wsrc, "kata_diubah": ed, "rasio_kata_diubah": round(ed / max(1, wsrc), 3)}
    owners = defaultdict(set)
    for sp, rows in keep.items():
        for r in rows: owners[r["penulis"]].add(sp)
    bocor = {p: sorted(s) for p, s in owners.items() if len(s) > 1}
    card = {
        "nama": "dataset_v2", "total_awal": len(isian),
        "sumber": "156 esai tulisan tangan L2, 18 penulis; OCR + koreksi gemini-3.1-pro-preview (kor-v2)",
        "label": "SILVER tervalidasi satu ahli (bukan gold): koreksi AI disetujui/diubah satu anotator, tanpa kappa",
        "catatan_evaluasi": "kunci test tetap berasal dari AI yang disetujui ahli -> evaluasi masih sirkular",
        "normalisasi": "saring.norm: harakat+tatwil dibuang, spasi sebelum tanda baca dirapikan",
        "split_per_penulis": True, "kebocoran_penulis": bocor,
        "override_putaran2": {"id_ditimpa": len({l[0] for l in log if l[1] == "putaran2"}),
                              "tanda_baca_ditambah": sum(l[1] == "tanda_baca" for l in log)},
        "statistik": {sp: stat(rows) for sp, rows in keep.items()},
        "ditahan": dict(Counter(r["alasan"].split(" ")[0] for r in tahan)),
        "ditahan_per_split": dict(Counter(r["split"] for r in tahan)),
        "dibuang": dict(Counter(r["alasan"].split(" ")[0] for r in buang)),
    }
    json.dump(card, open(os.path.join(OUT, "datacard.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    masuk = sum(len(v) for v in keep.values())
    print(json.dumps(card, ensure_ascii=False, indent=1))
    print(f"\nmasuk {masuk} + ditahan {len(tahan)} + dibuang {len(buang)} = {masuk + len(tahan) + len(buang)} (awal {len(isian)})")
    assert masuk + len(tahan) + len(buang) == len(isian) and not bocor


if __name__ == "__main__":
    main()

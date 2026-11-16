# Membuat pipeline/train_v2.ipynb — latih ulang resep C pada dataset_v2 (hasil verifikasi ahli).
# Fungsi latih/skor disalin apa adanya dari buat_notebook_tibyan.py supaya angka bisa dibandingkan.
import json, os

CELLS = []
def md(s): CELLS.append(("markdown", s.strip("\n")))
def code(s): CELLS.append(("code", s.strip("\n")))

md(r"""
# Latih ulang AraT5 pada dataset_v2 (korpus terverifikasi satu ahli)
| Kode | Model | Data latih L2 | Keterangan |
|---|---|---|---|
| **salin** | tanpa model | – | baseline: keluaran = masukan |
| **C_lama** | AraT5v2 → Tibyan → pilot30 | 181 pasangan | model terbaik lama, **dievaluasi ulang** di test v2 |
| **D** | AraT5v2 → Tibyan → dataset_v2 | 1.020 pasangan | resep C, data 5,6× |
| **E** | AraT5v2 → dataset_v2 | 1.020 pasangan | tanpa Tibyan (opsional, `RUN_E`) |

Semua diuji di **test v2 yang sama** (259 kalimat, penulis P15 & P17 — tidak ada di data latih mana pun).
Angka F0.5 0,505 dari notebook lama **tidak sebanding** (test lama 80 kalimat, kunci berbeda).

**Sebelum mulai**, pastikan di Drive `MyDrive/Kitabah_GEC/` ada:
- `dataset_v2/` — unggah dari `C:\Users\Habib\gramatical\data_siap\dataset_v2` (train/val/test.jsonl)
- `models/ablasi_tibyan/B_tibyan/` dan `C_tibyan_L2/` — hasil notebook ablasi (sudah ada)

Jalankan cell **1 → 9 berurutan**. Bila `B_tibyan` tidak ada, jalankan dulu CELL 1–6 notebook ablasi Tibyan.

Catatan jujur untuk laporan: label = *silver tervalidasi satu ahli*; kunci test berasal dari koreksi AI yang
disetujui ahli, jadi evaluasi masih sirkular; satu seed; val (P14) jauh lebih "kotor" daripada test.
""")

code(r"""
# CELL 1 — GPU + library (transformers 4.x; versi 5 bawaan Colab tidak cocok dgn tokenizer AraT5v2)
!nvidia-smi --query-gpu=name,memory.total --format=csv
!pip install -q "transformers>=4.46,<5" "accelerate>=1.0" sentencepiece
import torch, transformers, numpy, sys
print("python", sys.version.split()[0], "| torch", torch.__version__, "| transformers", transformers.__version__)
assert transformers.__version__.startswith("4."), "transformers masih versi 5 -> Runtime > Restart session, jalankan CELL 1 lagi"
assert torch.cuda.is_available(), "GPU belum aktif: Runtime -> Change runtime type -> GPU"
""")

code(r"""
# CELL 2 — Drive + folder + parameter
from google.colab import drive
drive.mount("/content/drive")
import os, json, random
ROOT    = "/content/drive/MyDrive/Kitabah_GEC"
L2_DIR  = f"{ROOT}/dataset_v2"
B_DIR   = f"{ROOT}/models/ablasi_tibyan/B_tibyan"       # AraT5v2 -> Tibyan (titik awal D)
C_LAMA  = f"{ROOT}/models/ablasi_tibyan/C_tibyan_L2"    # model terbaik lama (pembanding)
OUT     = f"{ROOT}/models/v2"
os.makedirs(OUT, exist_ok=True)
wajib = [f"{L2_DIR}/{f}" for f in ("train.jsonl", "val.jsonl", "test.jsonl")] + \
        [f"{d}/{f}" for d in (B_DIR, C_LAMA) for f in ("config.json", "model.safetensors")]
for p in wajib:
    print(("OK     " if os.path.exists(p) else "HILANG ") + p)
assert all(os.path.exists(p) for p in wajib), "Ada berkas yang HILANG — lihat daftar di atas"

SEED   = 42
EPOCHS = 30      # early stopping (sabar 3 epoch) biasanya berhenti jauh lebih awal
RUN_E  = True    # False = lewati eksperimen E (tanpa Tibyan) untuk menghemat waktu
""")

code(r"""
# CELL 3 — Baca data
def baca(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]
l2_train, l2_val, l2_test = (baca(f"{L2_DIR}/{s}.jsonl") for s in ("train", "val", "test"))
from collections import Counter
for nama, rows in (("train", l2_train), ("val", l2_val), ("test", l2_test)):
    ch = sum(x["src"] != x["tgt"] for x in rows)
    print(f"{nama:5s} {len(rows):5d} pasangan | berubah {100*ch/len(rows):.1f}% | tier {dict(Counter(x['tier'] for x in rows))}"
          f" | penulis {sorted({x['penulis'] for x in rows})}")
assert not ({x["penulis"] for x in l2_train} & {x["penulis"] for x in l2_test}), "penulis test bocor ke train"
assert not ({x["penulis"] for x in l2_val} & {x["penulis"] for x in l2_test}), "penulis test bocor ke val"
""")

code(r"""
# CELL 4 — Tokenizer, dataset, metrik, fungsi training (SAMA PERSIS dgn notebook ablasi Tibyan)
import torch, difflib, time
from transformers import (AutoTokenizer, AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq, Seq2SeqTrainer,
                          Seq2SeqTrainingArguments, EarlyStoppingCallback, set_seed)
BASE_MODEL, MAX_LEN = "UBC-NLP/AraT5v2-base-1024", 256
tok = AutoTokenizer.from_pretrained(BASE_MODEL)
# bf16 HANYA di GPU yang benar-benar mendukung (Ampere+, mis. L4/A100). T4 -> fp32 (bf16 T4 hanya emulasi).
BF16 = torch.cuda.get_device_capability(0)[0] >= 8
VRAM_GB = torch.cuda.get_device_properties(0).total_memory / 1e9
BS, ACC = (8, 1) if VRAM_GB >= 20 else (4, 2)
print(f"GPU {torch.cuda.get_device_name(0)} {VRAM_GB:.0f} GB | bf16={BF16} | batch={BS} x accum {ACC}")

class PairDS(torch.utils.data.Dataset):
    def __init__(self, rows): self.rows = rows
    def __len__(self): return len(self.rows)
    def __getitem__(self, i):
        x = self.rows[i]
        enc = tok(x["src"], max_length=MAX_LEN, truncation=True)
        enc["labels"] = tok(text_target=x["tgt"], max_length=MAX_LEN, truncation=True)["input_ids"]
        return enc

def model_dasar():
    m = AutoModelForSeq2SeqLM.from_pretrained(BASE_MODEL)
    for p in m.parameters(): p.data = p.data.contiguous()   # wajib, kalau tidak checkpoint gagal disimpan
    return m

def latih(model, train_rows, val_rows, nama, epochs, per_steps=None):
    set_seed(SEED)
    strat = "steps" if per_steps else "epoch"
    args = Seq2SeqTrainingArguments(
        output_dir=f"/content/ckpt_{nama}", overwrite_output_dir=True,
        num_train_epochs=epochs, learning_rate=3e-4, warmup_ratio=0.1, weight_decay=0.01,
        per_device_train_batch_size=BS, per_device_eval_batch_size=BS, gradient_accumulation_steps=ACC,
        eval_strategy=strat, save_strategy=strat, eval_steps=per_steps or 500, save_steps=per_steps or 500,
        save_total_limit=2, load_best_model_at_end=True, metric_for_best_model="eval_loss",
        greater_is_better=False, bf16=BF16, fp16=False, logging_strategy=strat, logging_steps=per_steps or 500,
        group_by_length=True, report_to="none", seed=SEED)
    tr = Seq2SeqTrainer(model=model, args=args, train_dataset=PairDS(train_rows), eval_dataset=PairDS(val_rows),
                        tokenizer=tok, data_collator=DataCollatorForSeq2Seq(tok, model=model),
                        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)])
    t0 = time.time(); tr.train()
    print(f"[{nama}] selesai {(time.time()-t0)/60:.1f} menit | val loss terbaik {tr.state.best_metric:.4f}")
    return tr.model, tr

def edits(src, tgt):
    s, t = src.split(), tgt.split()
    E = set()
    for op, i1, i2, j1, j2 in difflib.SequenceMatcher(None, s, t, autojunk=False).get_opcodes():
        if op == "equal": continue
        if op == "replace" and i2 - i1 == j2 - j1:
            E |= {(i1 + k, i1 + k + 1, t[j1 + k]) for k in range(i2 - i1) if s[i1 + k] != t[j1 + k]}
        else:
            E.add((i1, i2, " ".join(t[j1:j2])))
    return E

def skor(rows, hyps, beta=0.5):     # IDENTIK dgn notebook pilot30 (cell 8)
    tp = fp = fn = ok_total = ok_changed = benar_total = benar_diubah = exact = 0
    for x, h in zip(rows, hyps):
        he, re_ = edits(x["src"], h), edits(x["src"], x["tgt"])
        tp += len(he & re_); fp += len(he - re_); fn += len(re_ - he)
        ref_touch = {i for (a, b, _) in re_ for i in range(a, b)}
        hyp_touch = {i for (a, b, _) in he for i in range(a, b)}
        ok = [i for i in range(len(x["src"].split())) if i not in ref_touch]
        ok_total += len(ok); ok_changed += sum(i in hyp_touch for i in ok)
        if x["src"] == x["tgt"]:
            benar_total += 1; benar_diubah += (h.strip() != x["src"].strip())
        exact += h.strip() == x["tgt"].strip()
    P = tp / (tp + fp) if tp + fp else 1.0
    R = tp / (tp + fn) if tp + fn else 1.0
    b2 = beta * beta
    F = (1 + b2) * P * R / (b2 * P + R) if (P + R) else 0.0
    return {"precision": round(P, 3), "recall": round(R, 3), "F0.5": round(F, 3),
            "over_correction_%": round(100 * ok_changed / max(1, ok_total), 2),
            "alarm_palsu_%": round(100 * benar_diubah / max(1, benar_total), 1),
            "persis_sama_kunci_%": round(100 * exact / max(1, len(rows)), 1), "TP": tp, "FP": fp, "FN": fn}

def koreksi(model, rows, bs=16):
    model.eval(); out = []
    for i in range(0, len(rows), bs):
        enc = tok([x["src"] for x in rows[i:i+bs]], return_tensors="pt", padding=True,
                  truncation=True, max_length=MAX_LEN).to(model.device)
        with torch.no_grad():
            g = model.generate(**enc, max_length=MAX_LEN, num_beams=5)
        out += tok.batch_decode(g, skip_special_tokens=True)
    return [o.strip() for o in out]
""")

code(r"""
# CELL 5 — Pembanding: baseline salin-input + model lama C (dievaluasi di test v2)
import numpy as np
HASIL, HYP = {}, {}
HYP["salin"] = [x["src"] for x in l2_test]
HASIL["salin"] = skor(l2_test, HYP["salin"])

def muat(path):
    m = AutoModelForSeq2SeqLM.from_pretrained(path).to("cuda")
    for p in m.parameters(): p.data = p.data.contiguous()
    return m

def simpan_hasil():
    json.dump({"hasil": HASIL, "bf16": BF16, "gpu": torch.cuda.get_device_name(0), "seed": SEED,
               "n_train": len(l2_train), "n_val": len(l2_val), "n_test": len(l2_test)},
              open(f"{OUT}/hasil_v2.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)

def simpan(model, nama):
    d = f"{OUT}/{nama}"; os.makedirs(d, exist_ok=True)
    model.save_pretrained(d); tok.save_pretrained(d); simpan_hasil()
    print("tersimpan:", d)

mC = muat(C_LAMA)
HYP["C_lama"] = koreksi(mC, l2_test); HASIL["C_lama"] = skor(l2_test, HYP["C_lama"])
print("salin :", HASIL["salin"]); print("C_lama:", HASIL["C_lama"])
del mC; torch.cuda.empty_cache(); simpan_hasil()
""")

code(r"""
# CELL 6 — EXP D: (AraT5v2 -> Tibyan) -> dataset_v2        [model utama]
mD, trD = latih(muat(B_DIR), l2_train, l2_val, "D", epochs=EPOCHS)
HYP["D"] = koreksi(mD, l2_test); HASIL["D"] = skor(l2_test, HYP["D"])
HASIL["D_di_val"] = skor(l2_val, koreksi(mD, l2_val))
print("D (test):", HASIL["D"]); print("D (val) :", HASIL["D_di_val"])
simpan(mD, "D_tibyan_v2")
del mD, trD; torch.cuda.empty_cache()
""")

code(r"""
# CELL 7 — EXP E (opsional): AraT5v2 -> dataset_v2, tanpa Tibyan. Menjawab: apakah Tibyan masih perlu?
if RUN_E:
    mE, trE = latih(model_dasar(), l2_train, l2_val, "E", epochs=EPOCHS)
    HYP["E"] = koreksi(mE, l2_test); HASIL["E"] = skor(l2_test, HYP["E"])
    print("E (test):", HASIL["E"])
    simpan(mE, "E_v2_saja")
    del mE, trE; torch.cuda.empty_cache()
else:
    print("E dilewati (RUN_E = False)")
""")

code(r"""
# CELL 8 — Tabel perbandingan + rentang ketidakpastian (bootstrap per kalimat, 2.000 ulangan)
import pandas as pd
KOLOM = ["precision", "recall", "F0.5", "over_correction_%", "alarm_palsu_%", "persis_sama_kunci_%", "TP", "FP", "FN"]
display(pd.DataFrame({k: v for k, v in HASIL.items() if k != "D_di_val"}).T[KOLOM])

def hitung(rows, hyps):          # TP/FP/FN per kalimat, definisi edit sama dengan skor()
    out = []
    for x, h in zip(rows, hyps):
        he, re_ = edits(x["src"], h), edits(x["src"], x["tgt"])
        out.append((len(he & re_), len(he - re_), len(re_ - he)))
    return np.array(out)

def f05(c):
    tp, fp, fn = c.sum(0)
    P = tp / (tp + fp) if tp + fp else 1.0
    R = tp / (tp + fn) if tp + fn else 1.0
    return 1.25 * P * R / (0.25 * P + R) if P + R else 0.0

rng = np.random.default_rng(SEED)
IDX = rng.integers(0, len(l2_test), size=(2000, len(l2_test)))
CNT = {k: hitung(l2_test, HYP[k]) for k in HYP if k != "salin"}
BOOT = {k: np.array([f05(c[i]) for i in IDX]) for k, c in CNT.items()}
print("F0.5 dengan rentang 95% (bootstrap):")
for k, b in BOOT.items():
    print(f"  {k:7s} {f05(CNT[k]):.3f}  [{np.percentile(b, 2.5):.3f} – {np.percentile(b, 97.5):.3f}]")
for a, b in (("D", "C_lama"), ("D", "E")):
    if a in BOOT and b in BOOT:
        d = BOOT[a] - BOOT[b]
        print(f"  selisih {a} − {b}: {f05(CNT[a]) - f05(CNT[b]):+.3f}  [{np.percentile(d, 2.5):+.3f} – {np.percentile(d, 97.5):+.3f}]"
              f"  | {a} lebih baik di {100 * (d > 0).mean():.1f}% ulangan")

# Rincian: per penulis test dan per tingkat label
baris = []
for k in HYP:
    for label, pilih in [("P15", lambda x: x["penulis"] == "P15"), ("P17", lambda x: x["penulis"] == "P17"),
                         ("tier A", lambda x: x["tier"] == "A"), ("tier A+/B", lambda x: x["tier"] != "A")]:
        ii = [i for i, x in enumerate(l2_test) if pilih(x)]
        if ii:
            s = skor([l2_test[i] for i in ii], [HYP[k][i] for i in ii])
            baris.append({"model": k, "bagian": label, "n": len(ii), **{c: s[c] for c in KOLOM[:5]}})
display(pd.DataFrame(baris).set_index(["bagian", "model"]).sort_index())
HASIL["bootstrap_F0.5"] = {k: [round(float(np.percentile(b, 2.5)), 3), round(float(np.percentile(b, 97.5)), 3)] for k, b in BOOT.items()}
simpan_hasil()
""")

code(r"""
# CELL 9 — Contoh berdampingan (15 kalimat) + simpan semua prediksi
pd.set_option("display.max_colwidth", None)
tab = pd.DataFrame([{"id": x["id"], "tier": x["tier"], "src": x["src"], "kunci": x["tgt"],
                     **{k: HYP[k][i] for k in HYP if k != "salin"}} for i, x in enumerate(l2_test)])
tab.to_csv(f"{OUT}/prediksi_test_v2.csv", index=False, encoding="utf-8-sig")
print("tersimpan:", f"{OUT}/prediksi_test_v2.csv", "dan", f"{OUT}/hasil_v2.json")
tab.head(15)
""")

nb = {"nbformat": 4, "nbformat_minor": 5,
      "metadata": {"accelerator": "GPU", "colab": {"provenance": []},
                   "kernelspec": {"name": "python3", "display_name": "Python 3"}}, "cells": []}
for i, (kind, src) in enumerate(CELLS):
    c = {"cell_type": kind, "metadata": {}, "source": src.splitlines(keepends=True), "id": f"c{i:02d}"}
    if kind == "code": c.update({"execution_count": None, "outputs": []})
    nb["cells"].append(c)
out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "train_v2.ipynb")
json.dump(nb, open(out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print("ditulis:", out, "| cell:", len(CELLS))

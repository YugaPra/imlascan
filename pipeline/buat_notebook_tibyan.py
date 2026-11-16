# Membuat pipeline/train_ablasi_tibyan.ipynb — eksperimen A / B / C dgn setting identik.
import json, os

CELLS = []
def md(s): CELLS.append(("markdown", s.strip("\n")))
def code(s): CELLS.append(("code", s.strip("\n")))

md(r"""
# Ablasi Tibyan — A / B / C (AraT5v2)
| Exp | Training | Diuji di |
|---|---|---|
| **A** | AraT5v2 → data L2 kita (pilot30, 181 pasangan) | test L2 (80 kalimat, 2 penulis) |
| **B** | AraT5v2 → Tibyan saja | test L2 yang sama |
| **C** | AraT5v2 → Tibyan → data L2 kita | test L2 yang sama |

Semua setting sama (presisi, seed, lr, metrik), jadi selisih skor = efek Tibyan.
**Sebelum mulai**, upload ke Drive `MyDrive/Kitabah_GEC/`:
- `dataset_pilot30/` (sudah ada) dan `dataset_tibyan/` (dari `data_eksternal\tibyan\dataset_tibyan`).

Jalankan cell **1 → 9 berurutan**. Tahap B paling lama; lihat ETA di progress bar.
Data Tibyan tanpa lisensi di repo-nya → simpan di Drive pribadi, jangan dibagikan.
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
L2_DIR  = f"{ROOT}/dataset_pilot30"
TIB_DIR = f"{ROOT}/dataset_tibyan"
OUT     = f"{ROOT}/models/ablasi_tibyan"
os.makedirs(OUT, exist_ok=True)
for p in [f"{L2_DIR}/{f}" for f in ("train.jsonl", "val.jsonl", "test.jsonl")] + \
         [f"{TIB_DIR}/{f}" for f in ("train.jsonl", "val.jsonl")]:
    print(("OK     " if os.path.exists(p) else "HILANG ") + p)

TIB_N      = 6000    # jumlah pasangan Tibyan utk tahap B (dari ~22.9rb).
# Cek ETA di progress bar CELL 6. Bila > 2 jam di T4: hentikan, turunkan ke 3000. Bila L4/A100: boleh 12000+.
TIB_VAL_N  = 500
TIB_EPOCHS = 2
SEED       = 42
""")

code(r"""
# CELL 3 — Baca data
def baca(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]
l2_train, l2_val, l2_test = (baca(f"{L2_DIR}/{s}.jsonl") for s in ("train", "val", "test"))
tib_all, tib_val_all = baca(f"{TIB_DIR}/train.jsonl"), baca(f"{TIB_DIR}/val.jsonl")
rnd = random.Random(SEED)
tib_train = rnd.sample(tib_all, min(TIB_N, len(tib_all)))
tib_val   = rnd.sample(tib_val_all, min(TIB_VAL_N, len(tib_val_all)))
print(f"L2: train {len(l2_train)} | val {len(l2_val)} | test {len(l2_test)}")
print(f"Tibyan: train {len(tib_train)} (dari {len(tib_all)}) | val {len(tib_val)}")
""")

code(r"""
# CELL 4 — Tokenizer, dataset, metrik, fungsi training (dipakai A, B, C)
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

HASIL, HYP = {"baseline_salin_input": skor(l2_test, [x["src"] for x in l2_test])}, {}
def simpan(model, nama):
    d = f"{OUT}/{nama}"; os.makedirs(d, exist_ok=True)
    model.save_pretrained(d); tok.save_pretrained(d)
    json.dump({"hasil": HASIL, "bf16": BF16, "gpu": torch.cuda.get_device_name(0), "TIB_N": TIB_N},
              open(f"{OUT}/hasil_ablasi.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print("tersimpan:", d)
""")

code(r"""
# CELL 5 — EXP A: AraT5v2 -> data L2 kita (ulang dgn setting yang sama persis dgn B & C)
mA, trA = latih(model_dasar(), l2_train, l2_val, "A", epochs=30)
HYP["A"] = koreksi(mA, l2_test); HASIL["A_L2_saja"] = skor(l2_test, HYP["A"])
print(HASIL["A_L2_saja"])
del mA, trA; torch.cuda.empty_cache()
""")

code(r"""
# CELL 6 — EXP B: AraT5v2 -> Tibyan saja (PALING LAMA; lihat ETA di progress bar)
mB, trB = latih(model_dasar(), tib_train, tib_val, "B", epochs=TIB_EPOCHS, per_steps=300)
HYP["B"] = koreksi(mB, l2_test); HASIL["B_Tibyan_saja"] = skor(l2_test, HYP["B"])
print(HASIL["B_Tibyan_saja"])
simpan(mB, "B_tibyan")          # disimpan: jadi titik awal C (dan cadangan bila sesi putus)
""")

code(r"""
# CELL 7 — EXP C: (hasil B) -> data L2 kita
# Bila sesi sempat putus setelah CELL 6: jalankan CELL 1-4, lalu
#   mB = AutoModelForSeq2SeqLM.from_pretrained(f"{OUT}/B_tibyan")
for p in mB.parameters(): p.data = p.data.contiguous()
mC, trC = latih(mB, l2_train, l2_val, "C", epochs=30)
HYP["C"] = koreksi(mC, l2_test); HASIL["C_Tibyan_lalu_L2"] = skor(l2_test, HYP["C"])
print(HASIL["C_Tibyan_lalu_L2"])
simpan(mC, "C_tibyan_L2")
""")

code(r"""
# CELL 8 — Tabel perbandingan A / B / C
import pandas as pd
pd.DataFrame(HASIL).T[["precision", "recall", "F0.5", "over_correction_%", "alarm_palsu_%",
                       "persis_sama_kunci_%", "TP", "FP", "FN"]]
""")

code(r"""
# CELL 9 — Contoh berdampingan (15 kalimat) + simpan semua prediksi
pd.set_option("display.max_colwidth", None)
tab = pd.DataFrame([{"src": x["src"], "kunci": x["tgt"], **{k: HYP[k][i] for k in HYP}}
                    for i, x in enumerate(l2_test)])
tab.to_csv(f"{OUT}/prediksi_test_ABC.csv", index=False, encoding="utf-8-sig")
tab.head(15)
""")

nb = {"nbformat": 4, "nbformat_minor": 5,
      "metadata": {"accelerator": "GPU", "colab": {"provenance": []},
                   "kernelspec": {"name": "python3", "display_name": "Python 3"}}, "cells": []}
for i, (kind, src) in enumerate(CELLS):
    c = {"cell_type": kind, "metadata": {}, "source": src.splitlines(keepends=True), "id": f"c{i:02d}"}
    if kind == "code": c.update({"execution_count": None, "outputs": []})
    nb["cells"].append(c)
out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "train_ablasi_tibyan.ipynb")
json.dump(nb, open(out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print("ditulis:", out, "| cell:", len(CELLS))

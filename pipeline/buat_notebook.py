# Membuat pipeline/train_arat5_pilot30.ipynb (Colab). Jalankan: python pipeline/buat_notebook.py
import json, os

CELLS = []
def md(s): CELLS.append(("markdown", s.strip("\n")))
def code(s): CELLS.append(("code", s.strip("\n")))

md(r"""
# Training ulang AraT5 — pilot 30 esai (silver)
Urutan: jalankan **cell 1 sampai 10 berurutan**. Sebelum mulai:
1. `Runtime → Change runtime type → GPU` (pilih **L4/A100** kalau ada; kalau hanya **T4** juga bisa).
2. Upload folder `dataset_pilot30` ke Google Drive: `MyDrive/Kitabah_GEC/dataset_pilot30/`
   (isinya `train.jsonl`, `val.jsonl`, `test.jsonl`, `datacard.json`).

Model dilatih **dari AraT5v2 asli** (bukan melanjutkan checkpoint-540). Data = **SILVER** (koreksi Gemini),
jadi skor di akhir berarti "seberapa mirip dengan Gemini", bukan "seberapa benar menurut ahli".
""")

code(r"""
# CELL 1 — Cek GPU + pasang library (transformers 4.x; versi 5 bawaan Colab tidak cocok dgn tokenizer AraT5v2)
!nvidia-smi --query-gpu=name,memory.total --format=csv
!pip install -q "transformers>=4.46,<5" "accelerate>=1.0" sentencepiece
import torch, transformers, numpy, sys
print("python", sys.version.split()[0], "| torch", torch.__version__,
      "| transformers", transformers.__version__, "| numpy", numpy.__version__)
assert transformers.__version__.startswith("4."), "transformers masih versi 5 -> Runtime > Restart session, lalu jalankan CELL 1 lagi"
assert torch.cuda.is_available(), "GPU belum aktif: Runtime -> Change runtime type -> GPU"
""")

code(r"""
# CELL 2 — Hubungkan Google Drive + tentukan folder
from google.colab import drive
drive.mount("/content/drive")
import os
DATA_DIR = "/content/drive/MyDrive/Kitabah_GEC/dataset_pilot30"
OUT_DIR  = "/content/drive/MyDrive/Kitabah_GEC/models/arat5_pilot30"
os.makedirs(OUT_DIR, exist_ok=True)
for f in ["train.jsonl", "val.jsonl", "test.jsonl", "datacard.json"]:
    p = os.path.join(DATA_DIR, f)
    print(("OK   " if os.path.exists(p) else "HILANG ") + p)
""")

code(r"""
# CELL 3 — Baca data + statistik singkat
import json
def baca(nama):
    with open(os.path.join(DATA_DIR, nama), encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]
train, val, test = baca("train.jsonl"), baca("val.jsonl"), baca("test.jsonl")
for nama, d in [("train", train), ("val", val), ("test", test)]:
    berubah = sum(x["src"] != x["tgt"] for x in d)
    print(f"{nama:5}: {len(d):4} pasangan | berubah {berubah} | tidak berubah {len(d)-berubah}")
print("\nContoh:", train[0])
""")

code(r"""
# CELL 4 — Tokenizer + cek panjang token (harus <= 256 agar tidak terpotong)
from transformers import AutoTokenizer
BASE_MODEL = "UBC-NLP/AraT5v2-base-1024"
MAX_LEN = 256
tok = AutoTokenizer.from_pretrained(BASE_MODEL)
panjang = [len(tok(x[k]).input_ids) for d in (train, val, test) for x in d for k in ("src", "tgt")]
print("token terpanjang:", max(panjang), "| rata-rata:", round(sum(panjang)/len(panjang), 1))
print("kalimat > MAX_LEN:", sum(p > MAX_LEN for p in panjang))
""")

code(r"""
# CELL 5 — Siapkan dataset untuk model
import torch
class PairDS(torch.utils.data.Dataset):
    def __init__(self, rows): self.rows = rows
    def __len__(self): return len(self.rows)
    def __getitem__(self, i):
        x = self.rows[i]
        enc = tok(x["src"], max_length=MAX_LEN, truncation=True)
        enc["labels"] = tok(text_target=x["tgt"], max_length=MAX_LEN, truncation=True)["input_ids"]
        return enc
ds_train, ds_val = PairDS(train), PairDS(val)
print(len(ds_train), len(ds_val))
""")

code(r"""
# CELL 6 — Model + pengaturan training
#  - TIDAK pakai fp16 (T5 rawan overflow). bf16 bila GPU mendukung (L4/A100), selain itu fp32 (T4).
#  - Evaluasi tiap epoch, simpan model TERBAIK (val loss terendah), early stopping 3 epoch.
from transformers import (AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq, Seq2SeqTrainer,
                          Seq2SeqTrainingArguments, EarlyStoppingCallback, set_seed)
set_seed(42)
model = AutoModelForSeq2SeqLM.from_pretrained(BASE_MODEL)
for p in model.parameters():      # wajib: bobot AraT5v2 ada yang non-contiguous -> checkpoint gagal disimpan
    p.data = p.data.contiguous()
BF16 = torch.cuda.get_device_capability(0)[0] >= 8   # T4 (Turing) -> fp32; bf16 di T4 hanya emulasi
VRAM_GB = torch.cuda.get_device_properties(0).total_memory / 1e9
BS, ACC = (8, 1) if VRAM_GB >= 20 else (4, 2)
print(f"GPU {torch.cuda.get_device_name(0)} {VRAM_GB:.0f} GB | bf16={BF16} | batch={BS} x accum {ACC}")

args = Seq2SeqTrainingArguments(
    output_dir="/content/ckpt", overwrite_output_dir=True,
    num_train_epochs=30, learning_rate=3e-4, warmup_ratio=0.1, weight_decay=0.01,
    per_device_train_batch_size=BS, per_device_eval_batch_size=BS, gradient_accumulation_steps=ACC,
    eval_strategy="epoch", save_strategy="epoch", save_total_limit=2,
    load_best_model_at_end=True, metric_for_best_model="eval_loss", greater_is_better=False,
    bf16=BF16, fp16=False, logging_strategy="epoch", report_to="none", seed=42,
)
trainer = Seq2SeqTrainer(
    model=model, args=args, train_dataset=ds_train, eval_dataset=ds_val, tokenizer=tok,
    data_collator=DataCollatorForSeq2Seq(tok, model=model),
    callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
)
""")

code(r"""
# CELL 7 — TRAINING (bisa 10–40 menit tergantung GPU)
hasil = trainer.train()
print(hasil)
print("checkpoint terbaik:", trainer.state.best_model_checkpoint, "| val loss terbaik:", trainer.state.best_metric)
import pandas as pd
log = pd.DataFrame([l for l in trainer.state.log_history if "eval_loss" in l])[["epoch", "eval_loss"]]
log
""")

code(r"""
# CELL 8 — Evaluasi di data TEST (penulis yang tidak pernah dilihat saat training)
# Metrik = pendekatan M2 tingkat kata (BUKAN scorer resmi QALB; untuk paper pakai m2scorer resmi).
#  precision   : dari semua perubahan model, berapa yang sama dengan kunci (Gemini)
#  recall      : dari semua perubahan di kunci, berapa yang ditemukan model
#  F0.5        : gabungan, precision dibobot 2x lebih penting
#  over-correction : dari kata yang menurut kunci SUDAH BENAR, berapa % yang tetap diubah model
#  alarm palsu : dari kalimat yang menurut kunci SUDAH BENAR, berapa % yang diubah model
import difflib
def edits(src, tgt):
    s, t = src.split(), tgt.split()
    sm = difflib.SequenceMatcher(None, s, t, autojunk=False)
    E = set()
    for op, i1, i2, j1, j2 in sm.get_opcodes():
        if op == "equal": continue
        if op == "replace" and i2 - i1 == j2 - j1:          # ganti 1:1 -> dihitung per kata
            E |= {(i1 + k, i1 + k + 1, t[j1 + k]) for k in range(i2 - i1) if s[i1 + k] != t[j1 + k]}
        else:
            E.add((i1, i2, " ".join(t[j1:j2])))
    return E

def skor(rows, hyps, beta=0.5):
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
            "persis_sama_kunci_%": round(100 * exact / max(1, len(rows)), 1),
            "TP": tp, "FP": fp, "FN": fn}

def koreksi(rows, bs=16):
    model.eval(); out = []
    for i in range(0, len(rows), bs):
        enc = tok([x["src"] for x in rows[i:i+bs]], return_tensors="pt", padding=True,
                  truncation=True, max_length=MAX_LEN).to(model.device)
        with torch.no_grad():
            g = model.generate(**enc, max_length=MAX_LEN, num_beams=5)   # sama dengan ImlaScan
        out += tok.batch_decode(g, skip_special_tokens=True)
    return [o.strip() for o in out]

hyp_test = koreksi(test)
hasil_eval = {
    "baseline_salin_input": skor(test, [x["src"] for x in test]),
    "model_pilot30": skor(test, hyp_test),
}
import pandas as pd
pd.DataFrame(hasil_eval).T
""")

code(r"""
# CELL 9 — Lihat 15 contoh (src = tulisan siswa, kunci = Gemini, model = hasil model baru)
pd.set_option("display.max_colwidth", None)
contoh = pd.DataFrame([{"src": x["src"], "kunci": x["tgt"], "model": h,
                        "status": "sama" if h == x["tgt"] else ("tidak diubah" if h == x["src"] else "beda")}
                       for x, h in zip(test, hyp_test)])
print(contoh["status"].value_counts())
contoh.head(15)
""")

code(r"""
# CELL 10 — Simpan model terbaik + tokenizer (spiece.model) + hasil evaluasi ke Drive
import shutil
trainer.save_model(OUT_DIR)          # model terbaik (load_best_model_at_end=True)
tok.save_pretrained(OUT_DIR)         # supaya ImlaScan tidak perlu internet untuk tokenizer
with open(os.path.join(OUT_DIR, "eval_test.json"), "w", encoding="utf-8") as f:
    json.dump({"hasil": hasil_eval, "best_val_loss": trainer.state.best_metric,
               "epoch_terbaik": trainer.state.best_model_checkpoint,
               "bf16": BF16, "batch": BS, "accum": ACC, "gpu": torch.cuda.get_device_name(0)},
              f, ensure_ascii=False, indent=1)
contoh.to_csv(os.path.join(OUT_DIR, "contoh_test.csv"), index=False, encoding="utf-8-sig")
shutil.copy(os.path.join(DATA_DIR, "datacard.json"), OUT_DIR)
print(sorted(os.listdir(OUT_DIR)))
print("Selesai. Download folder ini dari Drive:", OUT_DIR)
""")

nb = {"nbformat": 4, "nbformat_minor": 5,
      "metadata": {"accelerator": "GPU", "colab": {"provenance": []},
                   "kernelspec": {"name": "python3", "display_name": "Python 3"}},
      "cells": []}
for i, (kind, src) in enumerate(CELLS):
    c = {"cell_type": kind, "metadata": {}, "source": src.splitlines(keepends=True), "id": f"c{i:02d}"}
    if kind == "code": c.update({"execution_count": None, "outputs": []})
    nb["cells"].append(c)
out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "train_arat5_pilot30.ipynb")
json.dump(nb, open(out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print("ditulis:", out, "| cell:", len(CELLS))

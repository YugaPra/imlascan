# -*- coding: utf-8 -*-
"""Cek pemasangan model di ImlaScan: muat lewat home/gec_model.py, koreksi test v2, hitung skor
dengan fungsi yang sama seperti notebook (pipeline/train_v2.ipynb). Harus mendekati angka Colab."""
import os, sys, json, time, difflib
ROOT = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, ROOT)
from dotenv import load_dotenv; load_dotenv(os.path.join(ROOT, ".env"))
import torch
from home import gec_model
nb = json.load(open(os.path.join(ROOT, "pipeline", "train_v2.ipynb"), encoding="utf-8"))
c4 = "".join(nb["cells"][4]["source"]); ns = {"difflib": difflib}
exec(c4[c4.index("def edits"):c4.index("def koreksi")], ns)
test = [json.loads(l) for l in open(os.path.join(ROOT, "data_siap", "dataset_v2", "test.jsonl"), encoding="utf-8")]
t0 = time.time(); assert gec_model._lazy_load(), "model gagal dimuat"
print("model:", os.environ["IMLASCAN_MODEL_PATH"], "| device:", gec_model._DEVICE, "| muat %.1f dtk" % (time.time() - t0))
tok, model = gec_model._TOKENIZER, gec_model._MODEL
print("tokenizer:", type(tok).__name__, "| vocab", len(tok), "| embedding", model.get_input_embeddings().weight.shape[0])
from transformers import AutoTokenizer
base = AutoTokenizer.from_pretrained("UBC-NLP/AraT5v2-base-1024")
beda = sum(tok(x["src"])["input_ids"] != base(x["src"])["input_ids"] for x in test)
print("kalimat yang token-nya berbeda dari tokenizer dasar:", beda, "dari", len(test))
t0 = time.time(); hyp = []
for i in range(0, len(test), 16):
    enc = tok([x["src"] for x in test[i:i+16]], return_tensors="pt", padding=True, truncation=True, max_length=256).to(gec_model._DEVICE)
    with torch.no_grad():
        g = model.generate(**enc, max_length=256, num_beams=5)
    hyp += [o.strip() for o in tok.batch_decode(g, skip_special_tokens=True)]
print("koreksi 259 kalimat (cara notebook): %.1f dtk" % (time.time() - t0))
print("SKOR:", ns["skor"](test, hyp))
# Jalur yang benar-benar dipakai web (urut panjang + batas panjang keluaran) harus memberi hasil yang sama
t0 = time.time(); hyp2 = gec_model.correct_batch([x["src"] for x in test])
print("koreksi 259 kalimat (jalur web): %.1f dtk | kalimat yang keluarannya berbeda: %d" % (time.time() - t0, sum(a != b for a, b in zip(hyp, hyp2))))
print("SKOR WEB:", ns["skor"](test, hyp2))

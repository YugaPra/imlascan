# Diagnostik pemuatan model (cara manual) + inferensi.
import os, django
os.environ["DJANGO_SETTINGS_MODULE"] = "gramatical.settings"
django.setup()

import torch
from transformers import (AutoConfig, AutoTokenizer, AutoModelForSeq2SeqLM,
                          GenerationConfig)
import home.gec_model as g

root = os.environ.get("IMLASCAN_MODEL_PATH", "")
wd = g._find_weights_dir(root)
cd = g._find_config_dir(root, wd)
print("weights_dir:", wd)
print("config_dir :", cd)

print("\n== [1] config ==")
config = AutoConfig.from_pretrained(cd)
print("OK vocab:", config.vocab_size)

print("\n== [2] model (manual load) ==")
model = AutoModelForSeq2SeqLM.from_config(config)
wfile = os.path.join(wd, "pytorch_model.bin")
try:
    state = torch.load(wfile, map_location="cpu", weights_only=True)
except TypeError:
    state = torch.load(wfile, map_location="cpu")
missing, unexpected = model.load_state_dict(state, strict=False)
print("OK. missing:", len(missing), "unexpected:", len(unexpected))
try:
    model.generation_config = GenerationConfig.from_pretrained(cd)
except Exception as e:
    print("gen_config skip:", e)
model.eval()

print("\n== [3] tokenizer base ==")
import transformers as _t
print("transformers version:", _t.__version__)
tok = AutoTokenizer.from_pretrained("UBC-NLP/AraT5v2-base-1024")
print("OK len:", len(tok), "| fast:", tok.is_fast)

print("\n== [4] inferensi (tanpa prefix) ==")
for kalimat in ["الطالب يكتبو الدرس", "احمد ذهب الي المدرسه فى مدينه جميله"]:
    enc = tok(kalimat, return_tensors="pt")
    out = model.generate(**enc, max_length=128, num_beams=5)
    print("IN :", kalimat)
    print("OUT:", repr(tok.decode(out[0], skip_special_tokens=True)))
    print()

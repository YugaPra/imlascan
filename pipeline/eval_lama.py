# Uji checkpoint-540 (model lama) di test pilot30 dengan metrik IDENTIK notebook (cell 8).
import json, os, sys, time
HERE = os.path.dirname(os.path.abspath(__file__)); BASE = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(BASE, "home"))
nb = json.load(open(os.path.join(HERE, "train_arat5_pilot30.ipynb"), encoding="utf-8"))
cell8 = ["".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code" and "def skor" in "".join(c["source"])][0]
ns = {}; exec(cell8.split("def koreksi")[0], ns)               # edits() + skor() persis notebook
os.environ["IMLASCAN_MODEL_PATH"] = r"D:\All projects\clone yuga\arat5-gec-20260720T113623Z-1-001\arat5-gec"
os.environ["IMLASCAN_DEVICE"] = "cuda"
import importlib.util as u
spec = u.spec_from_file_location("gm", os.path.join(BASE, "home", "gec_model.py")); gm = u.module_from_spec(spec); spec.loader.exec_module(gm)
assert gm._lazy_load()
import torch
test = [json.loads(l) for l in open(os.path.join(BASE, "data_siap", "dataset_pilot30", "test.jsonl"), encoding="utf-8")]
t0 = time.time(); hyp = []
for i in range(0, len(test), 16):
    enc = gm._TOKENIZER([x["src"] for x in test[i:i+16]], return_tensors="pt", padding=True, truncation=True, max_length=256).to(gm._DEVICE)
    with torch.no_grad():
        g = gm._MODEL.generate(**enc, max_length=256, num_beams=5)
    hyp += [o.strip() for o in gm._TOKENIZER.batch_decode(g, skip_special_tokens=True)]
res = {"checkpoint540_lama": ns["skor"](test, hyp), "baseline_salin_input": ns["skor"](test, [x["src"] for x in test])}
json.dump({"hasil": res, "hyp": hyp}, open(os.path.join(HERE, "_eval_lama.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print(f"{time.time()-t0:.0f}s"); print(json.dumps(res, indent=1))

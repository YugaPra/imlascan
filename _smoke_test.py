import os, sys, time
from dotenv import load_dotenv

ROOT = r"C:\Users\Habib\gramatical"
load_dotenv(os.path.join(ROOT, ".env"))
sys.path.insert(0, ROOT)

# PAKSA cpu -> hindari print "CUDA ... -> memakai CPU" yang crash di console cp1252
os.environ["IMLASCAN_DEVICE"] = "cpu"
print("DEVICE dipaksa:", os.environ["IMLASCAN_DEVICE"])

from home import gec_model

t0 = time.time()
ok = gec_model._lazy_load()
print("LOAD OK      :", ok, "| detik:", round(time.time() - t0, 1))

if ok:
    for s in ["احمد ذهب الي المدرسه فى مدينه جميله",
              "الطالبه قرأت الكتاب فى المكتبه الجديده"]:
        t1 = time.time()
        out = gec_model._correct_text(s)
        dt = round(time.time() - t1, 1)
        print("---")
        print("INFER detik  :", dt)
        print("INPUT        :", s)
        print("OUTPUT       :", out)
        print("IDENTIK?     :", out.strip() == s.strip())
        print("HASIL TAG    :", gec_model.analyze_with_model(s))

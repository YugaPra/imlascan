"""Uji opsi B: jalankan pipeline ImlaScan (tanpa server web) pada beberapa kalimat.
Pakai:  venv311\\Scripts\\python.exe areta_service\\_smoke_b.py
"""
import os, sys, json, time
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "gramatical.settings")
import django
django.setup()
from home import views, gec_model

TEKS = [
    "ذهب احمد الي المدرسه فى مدينه جميله",
    "إن أمتحان الاستاذة صعبة",
    "انا طالب في سنة الثالثة في الجامعه و احب اللغة العربيه",
]
for t in TEKS:
    t0 = time.time()
    raw = views.run_gec_model(t)
    hasil = views.analyze_imla(t)
    print("=" * 70)
    print("INPUT :", t)
    print("KOREKSI MODEL:", getattr(gec_model, "_LAST_CORRECTED", "(tidak tercatat)"))
    print("TAGGER:", hasil.get("tagger", "(lama)"), "| detik", round(time.time() - t0, 1))
    print("MENTAH :", json.dumps(raw, ensure_ascii=False))
    print("TAMPIL :", [(e["word"], e["suggestion"], e["error_type"]) for e in hasil["errors"]])

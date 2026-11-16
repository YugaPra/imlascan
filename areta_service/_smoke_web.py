"""Uji opsi B lewat halaman web asli (Django test client): POST teks -> HTML hasil."""
import os, sys, re
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT); os.chdir(ROOT)
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "gramatical.settings")
import django; django.setup()
from django.conf import settings
settings.ALLOWED_HOSTS = list(settings.ALLOWED_HOSTS) + ["testserver"]
from django.test import Client
from home import views
c = Client(enforce_csrf_checks=False)
r = c.post("/", {"teks_arab": "ذهب احمد الي المدرسه فى مدينه جميله"})
html = r.content.decode("utf-8")
print("HTTP", r.status_code, "| tagger:", views._LAST_SOURCE)
for code in ["OH", "OT", "OA", "XG", "OM"]:
    print(f"  badge {code}:", len(re.findall(rf">\s*{code}\b", html)) or html.count(f"{code} ·"))

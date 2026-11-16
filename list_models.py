# Daftar model Gemini yang tersedia untuk API key Anda (mendukung generateContent).
import os, django, requests
os.environ["DJANGO_SETTINGS_MODULE"] = "gramatical.settings"
django.setup()

key = os.environ.get("IMLASCAN_GEMINI_KEY", "").strip()
if not key:
    raise SystemExit("IMLASCAN_GEMINI_KEY kosong di .env")

r = requests.get(f"https://generativelanguage.googleapis.com/v1beta/models?key={key}",
                 timeout=30)
if r.status_code != 200:
    print("Gagal:", r.status_code, r.text[:300])
    raise SystemExit

print("Model yang mendukung generateContent (pakai salah satu untuk IMLASCAN_GEMINI_MODEL):\n")
for m in r.json().get("models", []):
    if "generateContent" in m.get("supportedGenerationMethods", []):
        name = m["name"].replace("models/", "")
        # tandai yang 'flash' (murah & cepat, cocok untuk OCR)
        mark = "  <== flash (disarankan)" if "flash" in name and "lite" not in name else ""
        print(f"  {name}{mark}")

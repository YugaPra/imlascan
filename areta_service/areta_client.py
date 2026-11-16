"""
Klien kecil untuk memanggil areta_service.py dari aplikasi lain (mis. ImlaScan / opsi B).
Hanya memakai pustaka standar Python, jadi aman dipakai di venv aplikasi utama (Python 3.11+).

Contoh:
    from areta_client import annotate
    hasil = annotate([("إن أمتحان الاستاذة صعبة", "إن إمتحان الأستاذ صعب")])
    for tok in hasil[0]:
        print(tok["src"], tok["tgt"], tok["tag"])
"""
import json
import os
import urllib.error
import urllib.request

ARETA_URL = os.environ.get("ARETA_URL", "http://127.0.0.1:8765")


def annotate(pairs, timeout=120):
    """pairs: list of (kalimat_salah, kalimat_benar) -> list per kalimat of {src, tgt, tag, info}."""
    body = json.dumps({"pairs": [{"src": s, "tgt": t} for s, t in pairs]}, ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(f"{ARETA_URL}/api/annotate", data=body,
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read().decode("utf-8"))["sentences"]
    except urllib.error.HTTPError as e:  # tampilkan pesan error dari layanan ARETA
        raise RuntimeError(f"ARETA error {e.code}: {e.read().decode('utf-8', 'replace')}") from None


def is_up():
    try:
        with urllib.request.urlopen(f"{ARETA_URL}/api/health", timeout=3) as r:
            return r.status == 200
    except OSError:
        return False


if __name__ == "__main__":
    if not is_up():
        raise SystemExit(f"Layanan ARETA belum jalan di {ARETA_URL}. Jalankan areta_service.py dulu.")
    for tok in annotate([("إن أمتحان الاستاذة صعبة", "إن إمتحان الأستاذ صعب")])[0]:
        print(f"{tok['src']:>12}  {tok['tgt']:>12}  {tok['tag']}")

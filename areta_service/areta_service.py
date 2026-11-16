"""
Layanan ARETA
=============
Satu program kecil yang:
  1. memuat ARETA (versi perbaikan dari repo CAMeL-Lab/arabic-gec) SEKALI saja,
  2. menyediakan API JSON di POST /api/annotate (dipakai ImlaScan) dan GET /api/health.

Layanan internal: tidak punya halaman untuk pengguna. Pengguna cukup membuka ImlaScan.

Jalankan dari environment ARETA (Python 3.7/3.8), BUKAN venv aplikasi utama:
    python areta_service.py --repo ~/arabic-gec
"""
import argparse
import json
import os
import sys
import tempfile
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

HERE = os.path.dirname(os.path.abspath(__file__))

# ---------------------------------------------------------------------------
# Kamus label (taksonomi ALC/ARETA, Belkebir & Habash 2021) dalam Bahasa Indonesia
# ---------------------------------------------------------------------------
TAG_INFO = {
    "UC": ("—", "Tidak ada kesalahan"),
    "OH": ("Ortografi", "Salah hamzah"),
    "OT": ("Ortografi", "Salah ta marbuta / ha"),
    "OA": ("Ortografi", "Salah alif maqshurah / ya"),
    "OW": ("Ortografi", "Salah alif fariqah (و ا)"),
    "ON": ("Ortografi", "Salah nun / tanwin"),
    "OS": ("Ortografi", "Vokal panjang dipendekkan"),
    "OG": ("Ortografi", "Vokal pendek dipanjangkan"),
    "OC": ("Ortografi", "Urutan huruf tertukar"),
    "OR": ("Ortografi", "Huruf diganti"),
    "OD": ("Ortografi", "Huruf berlebih"),
    "OM": ("Ortografi", "Huruf hilang"),
    "OO": ("Ortografi", "Kesalahan ortografi lain"),
    "MI": ("Morfologi", "Salah bentuk kata (infleksi)"),
    "MT": ("Morfologi", "Salah kala (tense) kata kerja"),
    "MO": ("Morfologi", "Kesalahan morfologi lain"),
    "XC": ("Sintaksis", "Salah i'rab / kasus"),
    "XF": ("Sintaksis", "Salah ma'rifah / nakirah (al-)"),
    "XG": ("Sintaksis", "Salah gender (mudzakkar/muannats)"),
    "XN": ("Sintaksis", "Salah jumlah (mufrad/mutsanna/jamak)"),
    "XT": ("Sintaksis", "Kata berlebih"),
    "XM": ("Sintaksis", "Kata hilang"),
    "XO": ("Sintaksis", "Kesalahan sintaksis lain"),
    "SW": ("Semantik", "Salah pilih kata"),
    "SF": ("Semantik", "Salah penggunaan kata sambung (fasl/wasl)"),
    "SO": ("Semantik", "Kesalahan semantik lain"),
    "PC": ("Tanda baca", "Tanda baca tertukar"),
    "PT": ("Tanda baca", "Tanda baca berlebih"),
    "PM": ("Tanda baca", "Tanda baca hilang"),
    "PO": ("Tanda baca", "Kesalahan tanda baca lain"),
    "MG": ("Gabung", "Dua kata menempel (harus dipisah)"),
    "SP": ("Pisah", "Satu kata terpisah (harus digabung)"),
    "UNK": ("?", "ARETA tidak yakin — perlu dicek manual"),
    # Label tingkat-kelas: ARETA tahu kelasnya tetapi tidak bisa menentukan sub-kelas
    "O": ("Ortografi", "Kesalahan ortografi (sub-jenis tak terdeteksi)"),
    "M": ("Morfologi", "Kesalahan morfologi (sub-jenis tak terdeteksi)"),
    "X": ("Sintaksis", "Kesalahan sintaksis (sub-jenis tak terdeteksi)"),
    "S": ("Semantik", "Kesalahan semantik (sub-jenis tak terdeteksi)"),
    "P": ("Tanda baca", "Kesalahan tanda baca (sub-jenis tak terdeteksi)"),
}


class Areta:
    """Pembungkus ARETA + aligner. Dimuat sekali, dipakai berkali-kali."""

    def __init__(self, repo):
        repo = os.path.abspath(os.path.expanduser(repo))
        self.areta_dir = os.path.join(repo, "areta")
        align_dir = os.path.join(repo, "alignment")
        for d in (self.areta_dir, align_dir):
            if not os.path.isdir(d):
                sys.exit(f"[ERROR] Folder tidak ditemukan: {d}\n"
                         f"Pastikan --repo menunjuk ke hasil 'git clone' arabic-gec.")
        # ARETA membaca config.json dengan path relatif -> harus berjalan di folder areta/
        os.chdir(self.areta_dir)
        sys.path.insert(0, self.areta_dir)
        # Levenshtein 0.12.0 versi pure-Python (identik dgn C 0.12.0, tanpa compiler) -> hasil sama di Windows & Linux
        sys.path.insert(0, os.path.join(HERE, "lev_shim"))

        t0 = time.time()
        print("[ARETA] Memuat kamus morfologi CAMeL (sekali saja, ~10 detik)...", flush=True)
        # Muat ARETA dulu: ia punya paket bernama 'aligner' (areta/aligner/)
        from scripts.annotation.an_annote_sys_ref import annote_ref_sys  # noqa: E402

        # ARETA (aligner/align_text_api.py) menulis fout2.basic dengan open(..., 'w') lalu membacanya
        # lagi. Di Windows itu menghasilkan '\r\n' dan label multi-kata berubah (UNK/S -> X).
        # Tambal dari luar: paksa akhir baris '\n' untuk penulisan teks di modul itu saja.
        import builtins
        import aligner.align_text_api as _ata

        def _open_lf(file, mode="r", *args, **kwargs):
            if "b" not in mode and ("w" in mode or "a" in mode):
                kwargs.setdefault("newline", "\n")
                kwargs.setdefault("encoding", "utf-8")
            return builtins.open(file, mode, *args, **kwargs)

        _ata.open = _open_lf

        # alignment/aligner.py bernama sama -> dimuat dengan nama lain agar tidak bentrok
        import importlib.util
        sys.path.append(align_dir)  # untuk ced_alignment.py & utils.py miliknya
        spec = importlib.util.spec_from_file_location("gec_aligner", os.path.join(align_dir, "aligner.py"))
        aligner = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(aligner)
        from ced_alignment import align_words  # noqa: E402
        self._aligner = aligner
        self._align_words = align_words
        self._annote = annote_ref_sys
        self._lock = threading.Lock()  # ARETA menulis file sementara -> satu permintaan per waktu
        self.work_dir = os.path.join(tempfile.gettempdir(), "areta_service_work")
        os.makedirs(self.work_dir, exist_ok=True)
        print(f"[ARETA] Siap dalam {time.time() - t0:.1f} detik.", flush=True)

    @staticmethod
    def _write_alignment(alignment, path):
        """Sama persis dengan aligner.write_data, tetapi selalu UTF-8 + akhir baris '\\n'.
        (Di Windows, write_data asli menghasilkan '\\r\\n' -> ARETA tidak mengenali baris pemisah.)"""
        with open(path, mode="w", encoding="utf-8", newline="\n") as f:
            f.write("SOURCE\tTARGET\n")
            for example in alignment:
                for s, t in zip(example["src"], example["tgt"]):
                    s = s.replace("PNX", "").replace("NIL", "")
                    t = t.replace("PNX", "").replace("NIL", "")
                    f.write(f"{s}\t{t}\n")
                f.write("\n")

    @staticmethod
    def _clean(text):
        return " ".join(text.replace("\t", " ").split())

    def annotate(self, pairs):
        """pairs: list of (kalimat_salah, kalimat_benar).
        return: list (per kalimat) of list of {src, tgt, tag}."""
        pairs = [(self._clean(s), self._clean(t)) for s, t in pairs]
        for i, (s, t) in enumerate(pairs, 1):
            if not s or not t:
                raise ValueError(f"Pasangan ke-{i} kosong.")

        with self._lock:
            # 1) Penjodohan kata (aligner Alhafni dkk. 2023)
            basic = [self._align_words(s, t) for s, t in pairs]
            clean = self._aligner.post_process_alignment(basic)
            self._aligner.verify([s for s, _ in pairs], [t for _, t in pairs], clean)

            # Folder kerja tetap (ditimpa tiap permintaan; aman karena ada lock). TemporaryDirectory
            # tidak dipakai: di Windows + Python 3.8 pembersihannya gagal bila ARETA masih memegang file.
            align_path = os.path.join(self.work_dir, "align.txt")
            self._write_alignment(clean, align_path)
            # 2) Pelabelan ARETA
            lines = self._annote(align_path, False)

        return self._parse(lines, len(pairs))

    @staticmethod
    def _parse(lines, n):
        sents, cur = [], []
        for line in lines.split("\n"):
            parts = line.split("\t")
            if len(parts) >= 3:
                cur.append({"src": parts[0], "tgt": parts[1], "tag": parts[2]})
            elif cur:
                sents.append(cur)
                cur = []
        if cur:
            sents.append(cur)
        if len(sents) != n:
            raise RuntimeError(f"Jumlah kalimat keluaran ({len(sents)}) != masukan ({n}).")
        return sents


def describe(tag):
    parts = tag.split("+")
    return [{"code": p, "kelas": TAG_INFO.get(p, ("?", "Label tidak dikenal"))[0],
             "arti": TAG_INFO.get(p, ("?", "Label tidak dikenal"))[1]} for p in parts]


def make_handler(areta):
    class Handler(BaseHTTPRequestHandler):
        def _send(self, code, body, ctype="application/json; charset=utf-8"):
            if isinstance(body, (dict, list)):
                body = json.dumps(body, ensure_ascii=False).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path == "/api/health":
                self._send(200, {"ok": True})
            else:
                self._send(404, {"error": "not found"})

        def do_POST(self):
            if self.path != "/api/annotate":
                return self._send(404, {"error": "not found"})
            try:
                n = int(self.headers.get("Content-Length", 0))
                data = json.loads(self.rfile.read(n).decode("utf-8"))
                pairs = [(p["src"], p["tgt"]) for p in data["pairs"]]
                if not pairs:
                    raise ValueError("Tidak ada kalimat.")
                if len(pairs) > 200:
                    raise ValueError("Maksimal 200 pasangan per permintaan.")
                t0 = time.time()
                result = areta.annotate(pairs)
                out = [[dict(tok, info=describe(tok["tag"])) for tok in sent] for sent in result]
                self._send(200, {"sentences": out, "seconds": round(time.time() - t0, 2)})
            except (KeyError, ValueError, json.JSONDecodeError) as e:
                self._send(400, {"error": str(e)})
            except Exception as e:  # noqa: BLE001
                import traceback
                traceback.print_exc()
                self._send(500, {"error": f"{type(e).__name__}: {e}"})

        def log_message(self, fmt, *args):
            sys.stderr.write("[HTTP] " + (fmt % args) + "\n")

    return Handler


def main():
    # ARETA menulis file sementara tanpa encoding eksplisit -> di Windows jatuh ke cp1252 dan
    # gagal pada huruf Arab. Paksa mode UTF-8 Python (PEP 540) dengan menjalankan ulang diri sendiri.
    if not sys.flags.utf8_mode:
        import subprocess
        try:
            sys.exit(subprocess.call([sys.executable, "-X", "utf8", os.path.abspath(__file__)] + sys.argv[1:]))
        except KeyboardInterrupt:
            sys.exit(0)

    ap = argparse.ArgumentParser(description="Layanan ARETA untuk ImlaScan")
    ap.add_argument("--repo", default="~/arabic-gec", help="Lokasi hasil git clone CAMeL-Lab/arabic-gec")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8765)
    args = ap.parse_args()

    areta = Areta(args.repo)
    srv = ThreadingHTTPServer((args.host, args.port), make_handler(areta))
    print(f"[ARETA] Layanan aktif di http://{args.host}:{args.port} (internal, dipakai ImlaScan). "
          f"Biarkan jendela ini terbuka.", flush=True)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\n[ARETA] Berhenti.")


if __name__ == "__main__":
    main()

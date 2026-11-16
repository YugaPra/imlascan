# -*- coding: utf-8 -*-
"""
home/kuota.py — kuota GLOBAL pemakaian Gemini OCR (semua pengunjung digabung).

Pembatas laju per-IP di views.py hanya menahan satu pengunjung; kuota ini menahan TOTAL
halaman yang dikirim ke Gemini per hari / per bulan, sehingga biaya API punya batas atas.
Saat kuota habis, views.py otomatis memakai Tesseract dan memberi tahu pengguna.

Disimpan di berkas JSON kecil (bukan memori) supaya tidak kembali nol saat server dinyalakan
ulang. Berkas ini juga mencatat jumlah token riil dari jawaban Gemini -> dasar hitung biaya.

Konfigurasi (.env):
    IMLASCAN_GEMINI_DAILY_MAX    halaman per hari  (default 30; 0 = tanpa batas)
    IMLASCAN_GEMINI_MONTHLY_MAX  halaman per bulan (default 0 = tanpa batas)
    IMLASCAN_KUOTA_UTC_OFFSET    zona waktu pergantian hari, jam dari UTC (default 7 = WIB)
    IMLASCAN_KUOTA_FILE          lokasi berkas (default: <proyek>/.imlascan_kuota.json)

Catatan: aman untuk satu proses server (kunci thread). Bila kelak memakai beberapa worker
gunicorn, pindahkan penyimpanan ke database/Redis.
"""
import json
import os
import threading
from datetime import datetime, timedelta, timezone

_BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_LOCK = threading.Lock()
# Hitungan dipegang di MEMORI (sumber kebenaran selama proses hidup); berkas hanya penyimpanan
# agar bertahan saat restart. Bila berkas gagal ditulis (disk penuh, terkunci), batas tetap
# ditegakkan dari memori — versi awal membaca ulang berkas tiap kali, sehingga gagal tulis
# berarti kuota tidak pernah bertambah (= tanpa batas).
_STATE = None
_STATE_PATH = None
_TOKEN_KEYS = ("token_masukan", "token_keluaran", "token_berpikir")


class KuotaHabis(RuntimeError):
    """Kuota Gemini OCR tidak cukup untuk permintaan ini (pesannya aman ditampilkan)."""


def _int_env(nama, default):
    try:
        return max(0, int(os.environ.get(nama, str(default)).strip() or default))
    except ValueError:
        return default


def batas_harian():
    return _int_env("IMLASCAN_GEMINI_DAILY_MAX", 30)


def batas_bulanan():
    return _int_env("IMLASCAN_GEMINI_MONTHLY_MAX", 0)


def _path():
    return os.environ.get("IMLASCAN_KUOTA_FILE", "").strip() or os.path.join(_BASE_DIR, ".imlascan_kuota.json")


def _sekarang():
    try:
        jam = float(os.environ.get("IMLASCAN_KUOTA_UTC_OFFSET", "7"))
    except ValueError:
        jam = 7.0
    return datetime.now(timezone.utc) + timedelta(hours=jam)


def _kosong():
    return {"halaman": 0, "berhasil": 0, **{k: 0 for k in _TOKEN_KEYS}}


def _baca():
    """Muat berkas dan gulirkan ke hari/bulan berjalan (hitungan lama otomatis di-nol-kan)."""
    global _STATE, _STATE_PATH
    if _STATE is None or _STATE_PATH != _path():
        try:
            with open(_path(), encoding="utf-8") as f:
                d = json.load(f)
            if not isinstance(d, dict):
                d = {}
        except (OSError, ValueError):
            d = {}
        _STATE, _STATE_PATH = d, _path()
    d = _STATE
    now = _sekarang()
    hari, bulan = now.strftime("%Y-%m-%d"), now.strftime("%Y-%m")
    if d.get("hari") != hari:
        d["hari"], d["hari_ini"] = hari, _kosong()
    if d.get("bulan") != bulan:
        d["bulan"], d["bulan_ini"] = bulan, _kosong()
    for bagian in ("hari_ini", "bulan_ini"):
        d[bagian] = {**_kosong(), **(d.get(bagian) if isinstance(d.get(bagian), dict) else {})}
    return d


def lupa():
    """Buang hitungan di memori supaya dibaca ulang dari berkas (dipakai uji / setelah edit manual)."""
    global _STATE
    with _LOCK:
        _STATE = None


def _tulis(d):
    """Tulis atomik (berkas sementara lalu ganti) agar tidak pernah setengah tertulis."""
    p = _path()
    tmp = f"{p}.{os.getpid()}.tmp"
    try:
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(d, f, ensure_ascii=False, indent=1)
        os.replace(tmp, p)
    except OSError as exc:      # disk penuh / tak bisa ditulis: jangan matikan OCR, cukup beri tahu log
        print(f"[ImlaScan] Kuota tidak bisa disimpan ({exc}).")


def _sisa(d):
    """Sisa halaman (None = tanpa batas) dengan mempertimbangkan batas harian DAN bulanan."""
    kandidat = []
    if batas_harian():
        kandidat.append(batas_harian() - d["hari_ini"]["halaman"])
    if batas_bulanan():
        kandidat.append(batas_bulanan() - d["bulan_ini"]["halaman"])
    return max(0, min(kandidat)) if kandidat else None


def status():
    """Untuk ditampilkan di form: {"terbatas", "sisa", "maks", "terpakai"}."""
    with _LOCK:
        d = _baca()
    sisa = _sisa(d)
    return {"terbatas": sisa is not None, "sisa": sisa, "maks": batas_harian(),
            "terpakai": d["hari_ini"]["halaman"], "maks_bulan": batas_bulanan(),
            "terpakai_bulan": d["bulan_ini"]["halaman"]}


def ambil(n):
    """Pesan n halaman sebelum memanggil Gemini (semua atau tidak sama sekali).
    Dipesan DI MUKA supaya permintaan bersamaan tidak bisa melewati batas, dan tetap dihitung
    walau panggilannya gagal (panggilan gagal/timeout pun bisa ditagih oleh Google)."""
    n = max(1, int(n))
    with _LOCK:
        d = _baca()
        sisa = _sisa(d)
        if sisa is not None and n > sisa:
            raise KuotaHabis(
                f"kuota Gemini OCR tidak cukup (perlu {n} halaman, sisa {sisa} hari ini)")
        d["hari_ini"]["halaman"] += n
        d["bulan_ini"]["halaman"] += n
        _tulis(d)


def catat_token(masukan=0, keluaran=0, berpikir=0):
    """Tambahkan pemakaian token riil dari usageMetadata satu jawaban Gemini."""
    with _LOCK:
        d = _baca()
        for bagian in ("hari_ini", "bulan_ini"):
            d[bagian]["berhasil"] += 1
            d[bagian]["token_masukan"] += int(masukan or 0)
            d[bagian]["token_keluaran"] += int(keluaran or 0)
            d[bagian]["token_berpikir"] += int(berpikir or 0)
        _tulis(d)

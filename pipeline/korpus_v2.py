"""Korpus v2 (DARI AWAL, semua 156 esai): OCR Gemini -> kalimat -> koreksi Gemini (kor-v2)
-> Excel verifikasi ahli per penulis.

Semua output di data_siap/v2/ ; data lama (pilot30, ocr/, koreksi/) TIDAK disentuh.
Pakai (dari folder gramatical):
  venv311\\Scripts\\python.exe pipeline\\korpus_v2.py semua                 (semua esai)
  venv311\\Scripts\\python.exe pipeline\\korpus_v2.py semua --penulis P15   (satu penulis)
  tahap terpisah: ocr | koreksi | excel   (+ --esai P15_E02 untuk uji)
Cache per halaman (OCR) & per esai (koreksi): aman dijalankan ulang, yang sudah ada dilewati.

Beda dengan pilot30:
- prompt koreksi kor-v2: larang ubah tense/aspek & ganti sinonim (masalah أدرس->درست).
- normalisasi (buang harakat/tatwil, rapikan spasi sebelum tanda baca) SEBELUM koreksi.
- kalimat boleh menyeberang halaman bila halaman sebelumnya tidak diakhiri . ! ? ؟
  (pilot30 memotong paksa di batas halaman -> serpihan kalimat).
- tidak ada penyaringan R1-R5 di sini: semua kalimat ke ahli; tanda bahaya ditulis di kolom
  'Peringatan sistem' agar ahli memutuskan.
"""
import argparse, csv, difflib, json, math, os, re, sys, time
from collections import OrderedDict, defaultdict
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ocr_batch import load_env, ocr_one, BASE, IMG_DIR, PROMPT_VERSION as OCR_PROMPT_VERSION
import koreksi as K

V2 = os.path.join(BASE, "data_siap", "v2")
OCR_DIR = os.path.join(V2, "ocr")
KOR_DIR = os.path.join(V2, "koreksi")
PAIRS = os.path.join(V2, "pairs_v2.jsonl")
XL_DIR = os.path.join(V2, "verifikasi")
PETA = os.path.join(BASE, "data_mentah", "peta_file.csv")
PILOT = os.path.join(BASE, "data_siap", "pilot30.csv")

KOR_PROMPT_VERSION = "kor-v2"
PROMPT_V2 = """You are an experienced Arabic teacher correcting sentences written by Indonesian
university students learning Arabic (L2). For EACH sentence, produce the corrected version with
MINIMAL EDITS:
- Fix spelling/orthography (hamza, ta marbuta/ha, alif maqsura/ya, alif, madda, letters),
  morphology (gender/number agreement, verb conjugation, definiteness) and clear grammar errors
  (prepositions, missing/extra words that make the sentence ungrammatical).
- Fix word spacing only when wrong (e.g. a detached و must be attached: "و احب" -> "وأحب").
- Do NOT change the verb tense/aspect chosen by the student (e.g. do not turn أدرس into درست)
  unless the context makes the original clearly ungrammatical.
- Do NOT replace a word with a synonym or a more elegant word; do NOT rephrase, reorder or add
  content; do NOT add diacritics. Keep punctuation unless it is clearly wrong.
- Keep any [?] marker exactly as it is.
- If a sentence is already correct, return it UNCHANGED.
- Never merge or split items: exactly one output per input id.
Return ONLY a JSON array: [{"id": "...", "tgt": "..."}] with exactly one item per input id.
Input sentences (JSON):
"""

HARAKAT = re.compile(r"[ً-ْٰـ]")   # tasykil + tatwil (sama dgn saring.py)
END = re.compile(r"[.!?؟]$")

def norm_text(s):
    """Normalisasi tapi baris baru dipertahankan (untuk segmentasi)."""
    s = HARAKAT.sub("", s)
    s = re.sub(r"[ \t]+([.,،؛:!?؟])", r"\1", s)
    return re.sub(r"[ \t]+", " ", s)

def norm(s):
    return re.sub(r"\s+", " ", norm_text(s or "")).strip()

def load_pages(penulis=None, esai=None):
    rows = list(csv.DictReader(open(PETA, encoding="utf-8-sig")))
    if penulis: rows = [r for r in rows if r["penulis"] in penulis]
    if esai: rows = [r for r in rows if r["esai_id"] in esai]
    es = OrderedDict()
    for r in sorted(rows, key=lambda r: (r["esai_id"], int(r["halaman"]))):
        es.setdefault(r["esai_id"], []).append(r)
    return es

def pilot_split():
    if not os.path.exists(PILOT): return {}
    return {r["penulis"]: r["split"] for r in csv.DictReader(open(PILOT, encoding="utf-8-sig"))}

# ---------------- Tahap 1: OCR ----------------
def tahap_ocr(es, key, model, sleep):
    os.makedirs(OCR_DIR, exist_ok=True)
    files = [r["file"] for rs in es.values() for r in rs]
    ok = fail = skip = 0
    for i, fn in enumerate(files, 1):
        out = os.path.join(OCR_DIR, os.path.splitext(fn)[0] + ".json")
        if os.path.exists(out): skip += 1; continue
        print(f"[OCR {i}/{len(files)}] {fn}", flush=True)
        try:
            res = ocr_one(os.path.join(IMG_DIR, fn), key, model); res["file"] = fn
            with open(out, "w", encoding="utf-8") as f: json.dump(res, f, ensure_ascii=False, indent=1)
            print(f"   OK {res['seconds']}s {len(res['text'])} char {res['model_version']} "
                  f"finish={res['finish_reason']}", flush=True); ok += 1
        except Exception as e:
            print("   GAGAL", e, flush=True); fail += 1
            with open(os.path.join(OCR_DIR, "_gagal.log"), "a", encoding="utf-8") as f:
                f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')}\t{fn}\t{e}\n")
        time.sleep(sleep)
    print(f"OCR selesai ok={ok} gagal={fail} cache={skip}", flush=True)

# ---------------- Tahap 2: kalimat + koreksi ----------------
def segment(pages):
    """pages: [(no_halaman, teks_ocr)] -> [(kalimat, [halaman, ...])].
    Paragraf (baris kosong) = batas kalimat. Batas halaman = batas kalimat HANYA jika halaman
    sebelumnya diakhiri . ! ? ؟ ; selain itu kalimat disambung ke halaman berikutnya."""
    toks, prev_end = [], True
    for k, (pg, raw) in enumerate(pages):
        t = norm_text(K.clean(raw))                      # K.clean: buang kata coretan ⟦...⟧
        paras = [p for p in re.split(r"\n\s*\n", t) if p.strip()]
        for i, p in enumerate(paras):
            for j, w in enumerate(p.split()):
                toks.append((w, pg, j == 0 and (k == 0 or i > 0 or prev_end)))
        if toks: prev_end = bool(END.search(toks[-1][0]))
    sents, cur = [], []
    def flush():
        if len(cur) >= 2:                               # serpihan 1 kata dibuang (sama dgn pilot30)
            sents.append((" ".join(w for w, _ in cur), sorted({p for _, p in cur})))
        cur.clear()
    for w, pg, brk in toks:
        if brk and cur: flush()
        cur.append((w, pg))
        if END.search(w): flush()
    flush()
    return sents

ARABIC = re.compile(r"[ء-ي]")
NAMA = re.compile(r"^\s*(ال)?[اإ]سم\s*[:：]")
def bukan_header(s):
    """Buang baris non-Arab (mis. 'Essay 2: Latar Belakang Pendidikan') dan baris formulir
    'الاسم: ...' (berisi nama asli siswa -> anonimitas). Nama di DALAM kalimat esai tidak tersaring."""
    return bool(ARABIC.search(s)) and not NAMA.match(s)

def tahap_koreksi(es, key, model):
    os.makedirs(KOR_DIR, exist_ok=True)
    K.PROMPT = PROMPT_V2                                  # K.call() memakai K.PROMPT
    for eid, rs in es.items():
        cpath = os.path.join(KOR_DIR, eid + ".json")
        if os.path.exists(cpath): continue
        pages = []
        for r in rs:
            p = os.path.join(OCR_DIR, os.path.splitext(r["file"])[0] + ".json")
            if not os.path.exists(p): print(f"[KOR] {eid}: OCR {r['file']} belum ada, dilewati", flush=True); pages = None; break
            pages.append((int(r["halaman"]), json.load(open(p, encoding="utf-8"))["text"]))
        if not pages: continue
        sents = [(s, pg) for s, pg in segment(pages) if bukan_header(s)]
        if not sents: print(f"[KOR] {eid}: 0 kalimat?", flush=True); continue
        items = [{"id": f"{eid}_S{i+1:02d}", "src": s} for i, (s, _) in enumerate(sents)]
        hal = {it["id"]: pg for it, (_, pg) in zip(items, sents)}
        t0 = time.time()
        try:
            tg, mv = K.call(key, model, items)
        except Exception as e:
            print(f"[KOR] {eid}: GAGAL {e}", flush=True); continue
        if isinstance(tg, dict):                          # jaga-jaga {"items":[...]}
            tg = next((v for v in tg.values() if isinstance(v, list)), [])
        res = {"items": items, "halaman": hal, "tgt": tg, "model_version": mv,
               "prompt_version": KOR_PROMPT_VERSION, "ocr_prompt_version": OCR_PROMPT_VERSION,
               "thinking": K.THINKING, "seconds": round(time.time() - t0, 1),
               "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S")}
        with open(cpath, "w", encoding="utf-8") as f: json.dump(res, f, ensure_ascii=False, indent=1)
        print(f"[KOR] {eid}: {len(items)} kalimat, {res['seconds']}s, {mv}", flush=True)
        time.sleep(3)

KEPALA = re.compile(r"^\s*([A-Za-z][^؀-ۿ]*?)(?=[؀-ۿ])")
def buang_kepala_latin(s):
    """Buang kepala halaman Latin yang menempel di awal kalimat pertama, mis.
    '<Nama Siswa> - Esai 3 ...', '<Nama> (1) ...', '<Nama Siswa> Essai 2 ...' (berisi NAMA siswa).
    Hanya dibuang jika awalan Latin memuat 'esai/essay', '(angka)' atau angka; kata Latin di
    tengah kalimat (nama tempat dalam kurung, dsb.) tidak disentuh."""
    m = KEPALA.match(s or "")
    if m and re.search(r"(?i)e\s*s+\s*a\s*[iy1]|essay|\(\s*\d+\s*\)|\d", m.group(1)):
        return s[m.end():].strip()
    return s

def kumpulkan():
    """Gabungkan SEMUA cache koreksi -> pairs_v2.jsonl (tanpa memanggil API)."""
    split, out, problems = pilot_split(), [], []
    for eid, rs in load_pages().items():
        cpath = os.path.join(KOR_DIR, eid + ".json")
        if not os.path.exists(cpath): problems.append(f"{eid}: belum dikoreksi"); continue
        res = json.load(open(cpath, encoding="utf-8"))
        tgt = {t.get("id"): str(t.get("tgt", "")) for t in res["tgt"] if isinstance(t, dict)}
        ids = [it["id"] for it in res["items"]]
        miss = [i for i in ids if not tgt.get(i, "").strip()]
        extra = sorted(set(tgt) - set(ids))
        if miss or extra: problems.append(f"{eid}: id hilang={miss} id asing={extra}")
        files = {r["halaman"]: r["file"] for r in rs}
        for it in res["items"]:
            out.append({"id": it["id"], "esai_id": eid, "esai": int(eid.split("_E")[1]),
                        "penulis": rs[0]["penulis"], "split": split.get(rs[0]["penulis"], ""),
                        "halaman": res["halaman"][it["id"]], "files": files,
                        "src": buang_kepala_latin(it["src"]),
                        "tgt": buang_kepala_latin(norm(tgt.get(it["id"], ""))), "tgt_kosong": it["id"] in miss,
                        "model_version": res["model_version"], "prompt_version": res["prompt_version"]})
    os.makedirs(V2, exist_ok=True)
    with open(PAIRS, "w", encoding="utf-8") as f:
        for o in out: f.write(json.dumps(o, ensure_ascii=False) + "\n")
    return out, problems

# ---------------- Tahap 3: Excel verifikasi ahli ----------------
from openpyxl import Workbook
from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
from openpyxl.worksheet.datavalidation import DataValidation
from openpyxl.formatting.rule import FormulaRule
from openpyxl.cell.rich_text import CellRichText, TextBlock
from openpyxl.cell.text import InlineFont
from openpyxl.utils import get_column_letter as L

F, AR = "Arial", "Traditional Arabic"
_t = Side(style="thin", color="BFBFBF"); BRD = Border(left=_t, right=_t, top=_t, bottom=_t)
def fill(c): return PatternFill("solid", fgColor=c)
HDR, INP, AUTO, DATA = fill("1F4E78"), fill("FFF2CC"), fill("E7E6E6"), fill("DDEBF7")
RTL = Alignment(horizontal="right", vertical="top", wrap_text=True, readingOrder=2)
LTR = Alignment(horizontal="left", vertical="top", wrap_text=True)
CTR = Alignment(horizontal="center", vertical="top", wrap_text=True)
# (judul, lebar, kelompok d=data/i=ahli/a=otomatis, arab?)
COLS = [("No", 6, "d", 0), ("ID Kalimat", 17, "d", 0), ("Nama_esai", 13, "d", 0), ("Esai", 6, "d", 0),
        ("Set", 8, "d", 0), ("Gambar (klik)", 10, "d", 0), ("Teks OCR (tulisan siswa)", 48, "d", 1),
        ("Koreksi AI", 48, "d", 1), ("Perubahan AI (asal ← koreksi)", 32, "d", 1),
        ("Peringatan sistem", 22, "d", 0), ("1. OCR sesuai gambar?", 12, "i", 0),
        ("Teks siswa yang benar (isi jika 'Tidak')", 42, "i", 1), ("2. Keputusan koreksi", 13, "i", 0),
        ("Koreksi versi ahli (isi jika 'Ubah')", 42, "i", 1), ("Keyakinan", 10, "i", 0),
        ("Catatan ahli", 30, "i", 0), ("Teks siswa FINAL (src)", 42, "a", 1),
        ("Koreksi FINAL (tgt)", 42, "a", 1), ("Status", 12, "a", 0)]
GF = {"d": DATA, "i": INP, "a": AUTO}
PETUNJUK = [
    ("PETUNJUK VERIFIKASI DATA KOREKSI ESAI BAHASA ARAB", "t"),
    ("Tujuan: memastikan setiap pasangan (teks siswa → teks koreksi) benar, agar bisa dipakai melatih dan menguji model koreksi otomatis. Satu baris = satu kalimat.", "n"),
    ("", "n"),
    ("YANG SUDAH DISIAPKAN (kolom BIRU, mohon tidak diubah)", "h"),
    ("• Kolom F: klik 'H1' / 'H2' untuk membuka foto halaman tulisan siswa. Folder 'gambar' harus tetap berada di samping file Excel ini.", "n"),
    ("• Kolom G: teks hasil pembacaan mesin (OCR). Kolom H: koreksi dari AI (Gemini). Kata yang diubah AI ditandai MERAH di teks siswa dan HIJAU di koreksi.", "n"),
    ("• Kolom I: daftar perubahan, dibaca kanan ke kiri (kata asal ← koreksi; ∅ = tidak ada kata). Kolom J: peringatan otomatis (kata tak terbaca, AI menulis ulang terlalu banyak, kalimat menyeberang halaman).", "n"),
    ("• Koreksi AI BELUM tentu benar. Mohon diperiksa seperti memeriksa jawaban murid, jangan langsung disetujui.", "n"),
    ("", "n"),
    ("YANG BAPAK/IBU ISI (kolom KUNING), per baris", "h"),
    ("Langkah 1. Buka gambar (kolom F). Apakah teks OCR (kolom G) SAMA PERSIS dengan tulisan siswa, termasuk kesalahannya? Pilih di kolom K.", "n"),
    ("      → Ya: lanjut ke langkah 2.   → Tidak: tulis ulang teks sesuai tulisan siswa di kolom L (salin apa adanya, JANGAN dikoreksi).", "n"),
    ("Langkah 2. Periksa koreksi AI (kolom H). Pilih di kolom M:", "n"),
    ("      → Setuju: koreksi AI benar dan cukup.", "n"),
    ("      → Ubah: ada koreksi yang salah, berlebihan, atau terlewat. Tulis kalimat koreksi LENGKAP versi Bapak/Ibu di kolom N.", "n"),
    ("      → Buang: kalimat tidak bisa dipakai (tak terbaca, terpotong, bukan bahasa Arab). Jelaskan singkat di kolom P.", "n"),
    ("      Catatan: bila di langkah 1 memilih 'Tidak', koreksi AI dibuat dari teks OCR yang salah. Jika koreksinya tidak cocok lagi dengan tulisan siswa, pilih 'Ubah'.", "n"),
    ("Langkah 3. Pilih keyakinan (kolom O) dan isi catatan bila perlu (kolom P).", "n"),
    ("", "n"),
    ("PRINSIP KOREKSI", "h"),
    ("1. Koreksi MINIMAL: ubah hanya yang salah secara nahwu, sharaf, atau imla'. Jangan memperindah gaya bahasa.", "n"),
    ("2. Jangan mengubah kala (tense), pilihan kata, atau makna bila tidak salah. Contoh: أدرس jangan diubah menjadi درست kecuali konteks mewajibkan.", "n"),
    ("3. Bila ada beberapa koreksi yang sama-sama benar, pilih yang paling dekat dengan tulisan siswa.", "n"),
    ("4. Harakat tidak perlu ditulis; data ini memakai teks tanpa harakat.", "n"),
    ("5. Kalimat yang sudah benar: bila AI tidak mengubah apa pun, pilih 'Setuju'. Bila AI mengubah kalimat yang sebenarnya sudah benar, pilih 'Ubah' dan salin kalimat aslinya ke kolom N.", "n"),
    ("6. Bila ragu (misalnya ada dua pendapat), pilih 'Ragu' di kolom O dan jelaskan di kolom P.", "n"),
    ("7. Mohon bekerja sendiri tanpa berdiskusi dengan ahli lain, agar tingkat kesepakatan antar-ahli dapat diukur.", "n"),
    ("", "n"),
    ("KETERANGAN WARNA & STATUS", "h"),
    ("BIRU = disiapkan peneliti • KUNING = diisi ahli • ABU-ABU = otomatis (jangan diubah).", "n"),
    ("Kolom S (Status): 'Selesai' (hijau) bila baris lengkap; 'Kurang' (merah) bila kolom L atau N wajib diisi tetapi masih kosong; 'Belum' bila belum dikerjakan. Sheet 'Rekap' menghitung progres otomatis.", "n"),
    ("Mohon file dikembalikan dalam format .xlsx tanpa mengubah urutan atau menghapus baris.", "n"),
]

def diff(src, tgt):
    a, b = src.split(), tgt.split()
    ra, rb, ch = [], [], []
    for op, i1, i2, j1, j2 in difflib.SequenceMatcher(None, a, b, autojunk=False).get_opcodes():
        x, y = " ".join(a[i1:i2]), " ".join(b[j1:j2])
        if op == "equal":
            ra.append((x, False)); rb.append((y, False))
        else:
            if x: ra.append((x, True))
            if y: rb.append((y, True))
            ch.append(f"{x or '∅'} ← {y or '∅'}")
    return ra, rb, ch

def rich(parts, color):
    base = InlineFont(rFont=AR, sz=14)
    hi = InlineFont(rFont=AR, sz=14, b=True, color=color, u="single")
    # PENTING (uji Excel 2026-10-08): blok yang isinya HANYA spasi (<t> </t>) membuat Excel
    # menampilkan "We found a problem with some content". Spasi digabung ke akhir blok kata,
    # lalu fix_spasi_xml() menambahkan xml:space="preserve" agar spasi tidak dibuang Excel.
    parts = [(t, h) for t, h in parts if t]
    out = CellRichText()
    for i, (txt, h) in enumerate(parts):
        out.append(TextBlock(hi if h else base, txt if i == len(parts) - 1 else txt + " "))
    return out

def fix_spasi_xml(path):
    """Tambahkan xml:space="preserve" pada <t> yang diawali/diakhiri spasi (aturan OOXML)."""
    import zipfile, shutil
    tmp = path + ".tmp"
    with zipfile.ZipFile(path) as zin, zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED) as zout:
        for it in zin.infolist():
            data = zin.read(it.filename)
            if it.filename.startswith("xl/worksheets/sheet"):
                x = data.decode("utf-8")
                x = re.sub(r"<t>(\s[^<]*|[^<]*\s)</t>",
                           lambda m: '<t xml:space="preserve">' + m.group(1) + "</t>", x)
                data = x.encode("utf-8")
            zout.writestr(it, data)
    shutil.move(tmp, path)

def peringatan(r):
    src, tgt, fl = r["src"], r["tgt"], []
    if "[?]" in src: fl.append("⚠ ada kata tak terbaca [?]")
    if r.get("tgt_kosong") or not tgt:
        fl.append("⚠ AI tidak memberi koreksi")
    else:
        sim = difflib.SequenceMatcher(None, src, tgt, autojunk=False).ratio()
        if sim < 0.70: fl.append(f"⚠ AI banyak menulis ulang (kemiripan {sim:.2f})")
        rt = len(tgt.split()) / max(1, len(src.split()))
        if not 0.75 <= rt <= 1.35: fl.append(f"⚠ jumlah kata berubah jauh ({rt:.2f}×)")
    if len(src.split()) > 64: fl.append("ℹ kalimat sangat panjang")
    if len(r["halaman"]) > 1: fl.append("ℹ kalimat menyeberang halaman")
    return fl

PETA_PENULIS = os.path.join(BASE, "data_mentah", "peta_penulis.csv")
def nama_panggilan():
    """kode -> nama panggilan (huruf kecil) HANYA untuk Excel ahli. Dataset/paper tetap kode P.
    Contoh: '<Nama Depan> <Nama Belakang> (5)' -> nama depan huruf kecil; 'Muh <Nama> (2)' -> '<nama>'."""
    out = {}
    for r in csv.DictReader(open(PETA_PENULIS, encoding="utf-8-sig")):
        toks = [t for t in re.sub(r"\(.*?\)", " ", r["folder_asli"]).lower().split()
                if re.sub(r"[^a-z]", "", t) not in ("", "muh", "muhammad", "moh", "mohammad")]
        out[r["kode"]] = re.sub(r"[^a-z]", "", toks[0]) if toks else r["kode"].lower()
    return out

def build_xlsx(rows, path, pen, nick):
    wb = Workbook()
    p = wb.active; p.title = "Petunjuk"; p.sheet_view.showGridLines = False
    p.column_dimensions["A"].width = 3; p.column_dimensions["B"].width = 115
    for i, (t, k) in enumerate(PETUNJUK, 2):
        c = p.cell(i, 2, t); c.alignment = Alignment(wrap_text=True, vertical="top")
        c.font = Font(name=F, size={"t": 14, "h": 11}.get(k, 10), bold=k in ("t", "h"),
                      color="1F4E78" if k in ("t", "h") else None)

    v = wb.create_sheet("Verifikasi")
    pv = ", ".join(sorted({r["prompt_version"] for r in rows}))
    mv = ", ".join(sorted({str(r["model_version"]) for r in rows}))
    info = [("Nama ahli:", None), ("Tanggal selesai:", None),
            ("Data:", f"Penulis {pen} · {len(rows)} kalimat · {len({r['esai_id'] for r in rows})} esai · "
                      f"OCR {OCR_PROMPT_VERSION} + koreksi {pv} ({mv}) · dibuat {time.strftime('%Y-%m-%d')}")]
    for i, (k, val) in enumerate(info, 1):
        v.merge_cells(start_row=i, start_column=1, end_row=i, end_column=2)
        v.cell(i, 1, k).font = Font(name=F, bold=True)
        v.merge_cells(start_row=i, start_column=3, end_row=i, end_column=8)
        c = v.cell(i, 3, val); c.font = Font(name=F, size=10)
        if val is None:
            c.fill = INP
            for cc in range(3, 9): v.cell(i, cc).border = BRD
    HR = 5
    for s, e, t, col in [(1, 10, "DISIAPKAN PENELITI (jangan diubah)", "2E75B6"),
                         (11, 16, "DIISI AHLI", "BF8F00"), (17, 19, "OTOMATIS", "7F7F7F")]:
        v.merge_cells(start_row=HR - 1, start_column=s, end_row=HR - 1, end_column=e)
        c = v.cell(HR - 1, s, t); c.font = Font(name=F, bold=True, color="FFFFFF"); c.fill = fill(col); c.alignment = CTR
    for i, (h, w, g, ar) in enumerate(COLS, 1):
        c = v.cell(HR, i, h); c.font = Font(name=F, bold=True, color="FFFFFF"); c.fill = HDR
        c.alignment = CTR; c.border = BRD
        v.column_dimensions[L(i)].width = w
    v.row_dimensions[HR].height = 45

    first = HR + 1; last = HR + len(rows)
    for n, r in enumerate(rows):
        x = first + n
        ra, rb, ch = diff(r["src"], r["tgt"]) if r["tgt"] else ([], [], [])
        hal = r["halaman"] or [1]
        img = "gambar/" + os.path.splitext(r["files"][str(hal[0])])[0] + ".jpg"
        vals = [n + 1, r["id"], f"{nick}_{r['esai']:02d}", r["esai"], r["split"], "H" + "+H".join(map(str, hal)),
                rich(ra, "C00000") if ch else r["src"], rich(rb, "007A33") if ch else r["tgt"],
                " ؛ ".join(ch) if ch else ("(tidak ada perubahan)" if r["tgt"] else ""),
                "\n".join(peringatan(r))]
        for j, val in enumerate(vals, 1): v.cell(x, j, val)
        v.cell(x, 17, f'=IF(OR(B{x}="",M{x}="Buang"),"",IF(K{x}="Tidak",L{x},G{x}))')
        v.cell(x, 18, f'=IF(OR(B{x}="",M{x}="Buang"),"",IF(M{x}="Ubah",N{x},IF(M{x}="Setuju",H{x},"")))')
        v.cell(x, 19, f'=IF(B{x}="","",IF(OR(K{x}="",M{x}="",O{x}=""),"Belum",'
                      f'IF(AND(K{x}="Tidak",L{x}=""),"Kurang (L)",IF(AND(M{x}="Ubah",N{x}=""),"Kurang (N)",'
                      f'IF(M{x}="Buang","Dibuang","Selesai")))))')
        for j, (h, w, g, ar) in enumerate(COLS, 1):
            c = v.cell(x, j); c.border = BRD; c.fill = GF[g]
            if ar:
                c.alignment = RTL; c.font = Font(name=AR, size=14)
            else:
                c.alignment = CTR if j in (1, 3, 4, 5, 6, 11, 13, 15, 19) else LTR
                c.font = Font(name=F, size=10, color="C00000" if j == 10 else None)
        c = v.cell(x, 6); c.hyperlink = img
        c.font = Font(name=F, size=10, color="0563C1", underline="single")
        longest = max(len(r["src"]), len(r["tgt"]))
        v.row_dimensions[x].height = min(220, 24 * max(2, math.ceil(longest / 42)))

    d = wb.create_sheet("Daftar")
    for col, vals in zip("ABCD", (["Ya", "Tidak"], ["Setuju", "Ubah", "Buang"], ["Yakin", "Ragu"], ["train", "val", "test"])):
        for j, val in enumerate(vals, 1): d[f"{col}{j}"] = val
    d.sheet_state = "hidden"
    if rows:
        for frm, col, msg in [("=Daftar!$A$1:$A$2", "K", "Ya / Tidak"), ("=Daftar!$B$1:$B$3", "M", "Setuju / Ubah / Buang"),
                              ("=Daftar!$C$1:$C$2", "O", "Yakin / Ragu"), ("=Daftar!$D$1:$D$3", "E", "train / val / test")]:
            dv = DataValidation(type="list", formula1=frm, allow_blank=True, showErrorMessage=True,
                                errorTitle="Pilihan tidak valid", error="Silakan pilih dari daftar.",
                                promptTitle="Pilih", prompt=msg)
            dv.showInputMessage = True; v.add_data_validation(dv); dv.add(f"{col}{first}:{col}{last}")
        red, green, grey = fill("F8CBAD"), fill("C6EFCE"), fill("D9D9D9")
        cf = v.conditional_formatting
        cf.add(f"L{first}:L{last}", FormulaRule(formula=[f'AND($K{first}="Tidak",$L{first}="")'], fill=red))
        cf.add(f"N{first}:N{last}", FormulaRule(formula=[f'AND($M{first}="Ubah",$N{first}="")'], fill=red))
        cf.add(f"S{first}:S{last}", FormulaRule(formula=[f'$S{first}="Selesai"'], fill=green))
        cf.add(f"S{first}:S{last}", FormulaRule(formula=[f'$S{first}="Dibuang"'], fill=grey))
        cf.add(f"S{first}:S{last}", FormulaRule(formula=[f'LEFT($S{first},6)="Kurang"'], fill=red))
    v.freeze_panes = f"C{first}"
    v.auto_filter.ref = f"A{HR}:S{max(last, first)}"

    k = wb.create_sheet("Rekap", 2); k.sheet_view.showGridLines = False
    for col, w in zip("ABC", (42, 14, 14)): k.column_dimensions[col].width = w
    k["A1"] = f"REKAP VERIFIKASI {pen} (otomatis)"; k["A1"].font = Font(name=F, size=13, bold=True, color="1F4E78")
    rng = lambda c: f"Verifikasi!${c}${first}:${c}${max(last, first)}"
    rek = [("Jumlah kalimat", f'=COUNTA({rng("B")})', None),
           ("Sudah selesai (termasuk dibuang)", f'=COUNTIF({rng("S")},"Selesai")+COUNTIF({rng("S")},"Dibuang")', "B3"),
           ("Belum / kurang lengkap", "=B3-B4", "B3"), ("", None, None),
           ("OCR tidak sesuai gambar", f'=COUNTIF({rng("K")},"Tidak")', "B4"),
           ("Koreksi AI disetujui", f'=COUNTIF({rng("M")},"Setuju")', "B4"),
           ("Koreksi AI diubah ahli", f'=COUNTIF({rng("M")},"Ubah")', "B4"),
           ("Kalimat dibuang", f'=COUNTIF({rng("M")},"Buang")', "B4"),
           ("Ahli ragu", f'=COUNTIF({rng("O")},"Ragu")', "B4")]
    for c, t in zip(("A2", "B2", "C2"), ("Indikator", "Jumlah", "Persentase")):
        k[c] = t; k[c].font = Font(name=F, bold=True, color="FFFFFF"); k[c].fill = HDR; k[c].alignment = CTR
    for i, (lab, f_, den) in enumerate(rek, 3):
        k.cell(i, 1, lab).font = Font(name=F)
        if not f_: continue
        k.cell(i, 2, f_).font = Font(name=F); k.cell(i, 2).alignment = CTR
        if den:
            c = k.cell(i, 3, f'=IF({den}=0,"-",B{i}/{den})'); c.number_format = "0.0%"
            c.font = Font(name=F); c.alignment = CTR
        for col in (1, 2, 3): k.cell(i, col).border = BRD
    k["A13"] = ("Persentase 'Belum' terhadap jumlah kalimat; baris OCR s.d. Ragu terhadap kalimat yang sudah selesai. "
                "'% diubah ahli' = seberapa sering koreksi AI (silver) salah.")
    k["A13"].font = Font(name=F, size=9, italic=True); k["A13"].alignment = Alignment(wrap_text=True, vertical="top")
    k.merge_cells("A13:C15")
    wb.active = 0
    wb.save(path)
    fix_spasi_xml(path)

def tahap_excel(es):
    from PIL import Image
    pairs, problems = kumpulkan()
    sel = {rs[0]["penulis"] for rs in es.values()}
    by = defaultdict(list)
    for r in pairs:
        if r["penulis"] in sel: by[r["penulis"]].append(r)
    for pen, rows in sorted(by.items()):
        d = os.path.join(XL_DIR, pen); g = os.path.join(d, "gambar"); os.makedirs(g, exist_ok=True)
        for fn in sorted({f for r in rows for f in r["files"].values()}):
            dst = os.path.join(g, os.path.splitext(fn)[0] + ".jpg")
            if os.path.exists(dst): continue
            im = Image.open(os.path.join(IMG_DIR, fn)).convert("RGB")
            im.thumbnail((2400, 2400)); im.save(dst, "JPEG", quality=85, optimize=True)
        nick = nama_panggilan().get(pen, pen.lower())
        path = os.path.join(d, f"Verifikasi_{pen}_{nick}.xlsx")
        build_xlsx(rows, path, pen, nick)
        print(f"[XLSX] {path}: {len(rows)} kalimat, {len({r['esai_id'] for r in rows})} esai", flush=True)
    for p in problems:
        if p.split("_")[0] in sel: print("  !", p, flush=True)
    print(f"pairs_v2.jsonl total={len(pairs)} kalimat", flush=True)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("tahap", choices=["ocr", "koreksi", "excel", "semua"])
    ap.add_argument("--penulis", nargs="*")
    ap.add_argument("--esai", nargs="*")
    ap.add_argument("--sleep", type=float, default=4.0, help="jeda antar halaman OCR (rate limit)")
    ap.add_argument("--model-kor", default="gemini-3.1-pro-preview",
                    help="model koreksi (default: Pro terbaik yang tersedia di key ini per 2026-10-07)")
    ap.add_argument("--thinking-kor", default="high")
    a = ap.parse_args()
    env = load_env(); key = env["IMLASCAN_GEMINI_KEY"]
    model = env.get("IMLASCAN_GEMINI_MODEL", "gemini-flash-latest")
    es = load_pages(a.penulis, a.esai)
    print(f"=== korpus_v2 {a.tahap}: esai={len(es)} halaman={sum(map(len, es.values()))} model={model} "
          f"{time.strftime('%Y-%m-%d %H:%M:%S')}", flush=True)
    if a.tahap in ("ocr", "semua"): tahap_ocr(es, key, model, a.sleep)
    if a.tahap in ("koreksi", "semua"):
        K.THINKING = a.thinking_kor
        print(f"koreksi: model={a.model_kor} thinking={K.THINKING}", flush=True)
        tahap_koreksi(es, key, a.model_kor)
    if a.tahap in ("excel", "semua"): tahap_excel(es)
    print("=== SELESAI", time.strftime('%Y-%m-%d %H:%M:%S'), flush=True)

if __name__ == "__main__":
    main()

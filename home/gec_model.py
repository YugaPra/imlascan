# -*- coding: utf-8 -*-
"""
home/gec_model.py — Integrasi model AraT5 (lokal, transformers) untuk ImlaScan.

Alur:
    teks mentah --generate()--> kalimat koreksi --align kata--> pasangan
    (kata_salah, koreksi) --tagger heuristik--> tag ARETA (OH/OT/OA/...).

Loader ini TAHAN terhadap folder Trainer "mentah" (seperti hasil download Anda):
- Bobot boleh berada langsung di folder, ATAU di subfolder checkpoint-XXXX/.
- config.json boleh di folder atas; bobot di checkpoint → tetap bisa dimuat.
- Bila file tokenizer (spiece.model) tidak ada, tokenizer diambil dari model
  dasar (IMLASCAN_TOKENIZER, default "UBC-NLP/AraT5v2-base-1024") — butuh
  internet saat pertama kali (unduh ~beberapa MB).

Desain aman: jika torch/transformers belum ada ATAU pemuatan gagal, fungsi
mengembalikan None → views.py otomatis fallback ke mesin lexicon (tidak crash).

Environment variable (set di .env):
    IMLASCAN_MODEL_PATH = folder model (mis. ...\\arat5-gec)
    IMLASCAN_TOKENIZER  = sumber tokenizer bila spiece.model tak ada
                          (default: UBC-NLP/AraT5v2-base-1024)
    IMLASCAN_PREFIX     = prefix tugas saat training, mis. "صحح: " (default kosong)
    IMLASCAN_DEVICE     = "cuda" | "cpu" (default auto)
    IMLASCAN_MAX_TOKENS = panjang maksimal (default 256)
"""

import os
import re
import difflib
import threading

# Satu kunci untuk pemuatan model (cegah dua permintaan pertama memuat model dua kali -> RAM 2x)
# dan satu untuk inferensi (satu generate() per waktu; server kecil / satu GPU).
_LOAD_LOCK = threading.Lock()
_INFER_LOCK = threading.Lock()

_MODEL = None
_TOKENIZER = None
_DEVICE = None
_LOAD_FAILED = False

_WEIGHT_FILES = ("model.safetensors", "pytorch_model.bin")


def _find_weights_dir(root):
    """Cari folder yang berisi bobot: root itu sendiri, atau checkpoint-* terbaru."""
    for f in _WEIGHT_FILES:
        if os.path.isfile(os.path.join(root, f)):
            return root
    candidates = []
    try:
        for name in os.listdir(root):
            p = os.path.join(root, name)
            if os.path.isdir(p) and name.lower().startswith("checkpoint"):
                if any(os.path.isfile(os.path.join(p, f)) for f in _WEIGHT_FILES):
                    candidates.append(p)
    except OSError:
        return None
    if not candidates:
        return None

    def _step(p):
        m = re.search(r"(\d+)$", os.path.basename(p))
        return int(m.group(1)) if m else -1
    return sorted(candidates, key=_step)[-1]  # checkpoint dengan step tertinggi


def _find_config_dir(root, weights_dir):
    """Folder mana yang punya config.json."""
    if weights_dir and os.path.isfile(os.path.join(weights_dir, "config.json")):
        return weights_dir
    if os.path.isfile(os.path.join(root, "config.json")):
        return root
    return None


def _lazy_load():
    """Muat model+tokenizer sekali. Return True bila siap."""
    if _MODEL is not None and _TOKENIZER is not None:
        return True
    with _LOAD_LOCK:
        return _load_locked()


def _load_locked():
    global _MODEL, _TOKENIZER, _DEVICE, _LOAD_FAILED
    if _MODEL is not None and _TOKENIZER is not None:
        return True
    if _LOAD_FAILED:
        return False

    root = os.environ.get("IMLASCAN_MODEL_PATH", "").strip()
    if not root or not os.path.isdir(root):
        _LOAD_FAILED = True
        return False

    try:
        import torch
        from transformers import (AutoConfig, AutoTokenizer,
                                  AutoModelForSeq2SeqLM, GenerationConfig)

        weights_dir = _find_weights_dir(root)
        if weights_dir is None:
            print(f"[ImlaScan] Bobot model tidak ditemukan di {root} "
                  f"(cari .safetensors/.bin di folder atau checkpoint-*/).")
            _LOAD_FAILED = True
            return False

        config_dir = _find_config_dir(root, weights_dir)
        if config_dir is None:
            print(f"[ImlaScan] config.json tidak ditemukan di {root}.")
            _LOAD_FAILED = True
            return False

        requested = os.environ.get("IMLASCAN_DEVICE", "").strip().lower()
        if requested == "cuda" and not torch.cuda.is_available():
            print("[ImlaScan] CUDA diminta tapi tidak tersedia → memakai CPU.")
            requested = "cpu"
        _DEVICE = requested or ("cuda" if torch.cuda.is_available() else "cpu")

        config = AutoConfig.from_pretrained(config_dir)

        # Bangun arsitektur dari config, lalu muat bobot MANUAL. Ini menghindari
        # from_pretrained() yang auto-mencari generation_config.json di folder
        # checkpoint (tidak ada di sana) → memicu error path NoneType.
        model = AutoModelForSeq2SeqLM.from_config(config)

        wfile = next((os.path.join(weights_dir, f) for f in _WEIGHT_FILES
                      if os.path.isfile(os.path.join(weights_dir, f))), None)
        if wfile is None:
            raise FileNotFoundError(f"Bobot tidak ada di {weights_dir}")

        if wfile.endswith(".safetensors"):
            from safetensors.torch import load_file
            state = load_file(wfile)
        else:
            try:
                state = torch.load(wfile, map_location="cpu", weights_only=True)
            except TypeError:  # torch lama tanpa arg weights_only
                state = torch.load(wfile, map_location="cpu")

        missing, unexpected = model.load_state_dict(state, strict=False)
        # encoder/decoder.embed_tokens berbagi bobot dengan "shared" di T5 → wajar tak tersimpan.
        missing = [k for k in missing if not k.endswith("embed_tokens.weight")]
        if missing:
            print(f"[ImlaScan] Peringatan: {len(missing)} bobot tak terisi "
                  f"(mis. {missing[:3]}).")

        # Ambil generation_config dari folder yang memilikinya (folder atas).
        try:
            model.generation_config = GenerationConfig.from_pretrained(config_dir)
        except Exception:
            pass

        model.to(_DEVICE).eval()

        # Tokenizer: pakai lokal HANYA bila tokenizer.json ada. Bila tidak
        # (kasus folder Trainer tanpa spiece.model), ambil dari model DASAR.
        # use_fast=False → tokenizer SentencePiece (T5) langsung, tanpa konversi
        # fast yang rewel.
        # WAJIB tokenizer.json: bila hanya spiece.model yang ada, transformers membangun ulang
        # tokenizer dari SentencePiece dan hasilnya BERBEDA dari tokenizer saat training
        # (terukur 16-11-2026: F0.5 model D turun 0,579 -> 0,498 tanpa pesan error apa pun).
        has_local_spm = os.path.isfile(os.path.join(root, "tokenizer.json"))
        tok_src = root if has_local_spm else os.environ.get(
            "IMLASCAN_TOKENIZER", "UBC-NLP/AraT5v2-base-1024")
        if not has_local_spm:
            print(f"[ImlaScan] tokenizer.json lokal tak ada → tokenizer dari "
                  f"'{tok_src}' (butuh internet sekali untuk unduh).")
        # Tokenizer fast (transformers 4.44.2 di Python 3.11 memuat T5TokenizerFast
        # dari tokenizer.json dengan benar). T5TokenizerFast sebagai cadangan.
        try:
            _TOKENIZER = AutoTokenizer.from_pretrained(tok_src)
        except Exception as e_fast:
            print(f"[ImlaScan] AutoTokenizer gagal ({e_fast}); coba T5TokenizerFast...")
            from transformers import T5TokenizerFast
            _TOKENIZER = T5TokenizerFast.from_pretrained(tok_src)

        # Jaga-jaga: perbesar embedding bila tokenizer LEBIH BESAR dari model. Tidak pernah
        # dikecilkan (AraT5v2 memang punya 110.208 baris embedding untuk tokenizer 110.100).
        try:
            n_emb, n_tok = model.get_input_embeddings().weight.shape[0], len(_TOKENIZER)
            if n_tok > n_emb:
                model.resize_token_embeddings(n_tok)
        except Exception:
            pass
        _MODEL = model   # baru sekarang dianggap siap (tokenizer sudah ada)

        print(f"[ImlaScan] Model dimuat. weights={weights_dir} device={_DEVICE}")
        return True
    except Exception as exc:
        print(f"[ImlaScan] Gagal memuat model AraT5: {exc}")
        _MODEL = _TOKENIZER = None
        _LOAD_FAILED = True
        return False


# Normalisasi & pemecahan kalimat — HARUS sama dengan pipeline/saring.py & pipeline/koreksi.py
# (model dilatih per kalimat, tanpa harakat). Versi lama mengirim seluruh esai sekaligus dan
# memotongnya di 256 token, sehingga sisa esai hilang diam-diam.
_HARAKAT = re.compile(r"[ً-ْٰـ]")


def normalize_text(s):
    s = _HARAKAT.sub("", s)
    s = re.sub(r"[ \t]+([.,،؛:!?؟])", r"\1", s)
    return re.sub(r"[ \t]+", " ", s).strip()


def split_sentences(text):
    sents = []
    for p in re.split(r"\n\s*\n", text):
        p = re.sub(r"\s*\n\s*", " ", p).strip()
        sents += [s.strip() for s in re.split(r"(?<=[.!?؟])\s+", p) if s.strip()]
    return sents


_MAX_WORDS = 64     # sama dengan batas data latih (saring.py R4: > 64 kata dibuang)


def _split_long(sent):
    """Pecah kalimat yang melebihi _MAX_WORDS (umumnya hasil OCR tanpa titik) di tanda koma
    terdekat, atau paksa per _MAX_WORDS kata. Kalimat sepanjang itu tak pernah dilihat model
    saat training dan membuat generate() sangat lambat (terukur: 12 kalimat OCR = 59 detik)."""
    words = sent.split()
    if len(words) <= _MAX_WORDS:
        return [sent]
    parts, cur = [], []
    for w in words:
        cur.append(w)
        if (len(cur) >= 24 and w[-1:] in "،,;؛:") or len(cur) >= _MAX_WORDS:
            parts.append(" ".join(cur)); cur = []
    if cur:
        if parts and len(cur) < 4:
            parts[-1] += " " + " ".join(cur)
        else:
            parts.append(" ".join(cur))
    return parts


def correct_batch(sents):
    """Koreksi daftar kalimat (sudah dinormalisasi) → daftar kalimat koreksi, urutan sama.
    Kalimat diurutkan menurut panjang agar batch seragam, dan panjang keluaran dibatasi
    relatif terhadap masukan (koreksi tidak pernah jauh lebih panjang dari kalimat asal)."""
    import torch
    max_tokens = int(os.environ.get("IMLASCAN_MAX_TOKENS", "256"))
    prefix = os.environ.get("IMLASCAN_PREFIX", "")  # mis. "صحح: " bila dipakai saat training
    bs = max(1, int(os.environ.get("IMLASCAN_BATCH", "16")))
    out_sents = list(sents)
    idx = sorted((i for i, s in enumerate(sents) if len(s.split()) >= 2),   # serpihan 1 kata tak dikirim
                 key=lambda i: len(sents[i].split()))
    for b in range(0, len(idx), bs):
        chunk = idx[b:b + bs]
        enc = _TOKENIZER([prefix + sents[i] for i in chunk], return_tensors="pt", padding=True,
                         truncation=True, max_length=max_tokens).to(_DEVICE)
        batas = min(max_tokens, int(1.5 * enc["input_ids"].shape[1]) + 16)
        with _INFER_LOCK, torch.no_grad():
            out = _MODEL.generate(**enc, max_length=batas, num_beams=5)
        for i, o in zip(chunk, _TOKENIZER.batch_decode(out, skip_special_tokens=True)):
            out_sents[i] = o.strip() or sents[i]     # keluaran kosong -> anggap tidak berubah
    return out_sents


def _correct_sentences(text):
    """Inferensi seq2seq PER KALIMAT → (kalimat_sumber, kalimat_koreksi), dua list sejajar."""
    sents = [p for s in split_sentences(text) for p in _split_long(s)]
    return sents, correct_batch(sents)


def _correct_text(text):
    """Kompatibilitas: teks koreksi sebagai satu string (kalimat digabung spasi)."""
    return " ".join(_correct_sentences(text)[1])


# ---------------------------------------------------------------------------
# Tagger heuristik: kode ARETA dari pasangan (salah, benar). Ringan, tanpa
# dependency. Bukan pengganti ARETA/CAMeL Tools presisi, tapi memadai untuk demo.
# ---------------------------------------------------------------------------
_HAMZA_FORMS = set("أإآءؤئ")


def _strip_last(s):
    return s[:-1] if s else s


def infer_tag(wrong, correct):
    """Kembalikan kode tag ARETA (str) untuk satu pasangan kata."""
    if wrong == correct:
        return None

    # OT: beda hanya Ta' Marbuthah/Ha di akhir (ة <-> ه)
    if _strip_last(wrong) == _strip_last(correct) and \
       {wrong[-1:], correct[-1:]} <= {"ة", "ه"}:
        return "OT"

    # OA: beda hanya Alif Maqsurah/Ya di akhir (ى <-> ي)
    if _strip_last(wrong) == _strip_last(correct) and \
       {wrong[-1:], correct[-1:]} <= {"ى", "ي"}:
        return "OA"

    # OW: Waw jamak +/- Alif Fariqah (...و <-> ...وا)
    if correct == wrong + "ا" and wrong.endswith("و"):
        return "OW"
    if wrong == correct + "ا" and correct.endswith("و"):
        return "OW"

    # OH: perbedaan menyangkut bentuk Hamzah
    diff_chars = set(wrong) ^ set(correct)
    if diff_chars & _HAMZA_FORMS or \
       (len(wrong) == len(correct) and any(
           (a in _HAMZA_FORMS or b in _HAMZA_FORMS) and a != b
           for a, b in zip(wrong, correct))):
        if wrong[-1:] in ("ي", "ى") and correct[-1:] in ("ي", "ى") \
           and wrong[-1:] != correct[-1:]:
            return "OH+OA"
        return "OH"

    # OD / OM: selisih panjang
    if len(wrong) > len(correct):
        return "OD"
    if len(wrong) < len(correct):
        return "OM"

    # OR: penggantian huruf lain (panjang sama)
    return "OR"


# ---------------------------------------------------------------------------
# Tagger ARETA (opsi B). ARETA berjalan sebagai layanan terpisah (venv_areta,
# Python 3.8) di http://127.0.0.1:8765 — lihat areta_service/README.md.
# IMLASCAN_TAGGER = "areta" (default) | "heuristic"
# Bila layanan mati / error → otomatis kembali ke infer_tag() (tidak crash).
# ---------------------------------------------------------------------------
LAST_TAGGER = None      # "areta" | "heuristic" — dibaca views.py untuk ditampilkan
_LAST_CORRECTED = None  # kalimat hasil AraT5 terakhir (untuk debug)


def _areta_client():
    import sys
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    svc = os.path.join(here, "areta_service")
    if svc not in sys.path:
        sys.path.insert(0, svc)
    import areta_client
    return areta_client


def _tag_with_areta(sents, out_sents):
    """Return list[dict] {word, suggestion, tag} dari ARETA, atau None bila tak tersedia.
    Dikirim PER KALIMAT (bukan satu esai utuh): penjodohan kata ARETA berbiaya kuadratik
    terhadap panjang teks, jadi per kalimat jauh lebih cepat dan lebih tepat."""
    if os.environ.get("IMLASCAN_TAGGER", "areta").strip().lower() != "areta":
        return None
    try:
        client = _areta_client()
        if not client.is_up():
            print("[ImlaScan] Layanan ARETA tidak aktif → pakai tagger heuristik.")
            return None
        pairs = [(s, t) for s, t in zip(sents, out_sents) if s.strip() and t.strip()]
        sentences = []
        for i in range(0, len(pairs), 100):          # layanan membatasi 200 pasangan per permintaan
            sentences += client.annotate(pairs[i:i + 100], timeout=60)
    except Exception as exc:
        print(f"[ImlaScan] ARETA gagal ({exc}) → pakai tagger heuristik.")
        return None

    results = []
    for sent in sentences:
        for tok in sent:
            if tok["tag"] == "UC":
                continue  # kata tidak berubah
            results.append({"word": tok["src"], "suggestion": tok["tgt"], "tag": tok["tag"]})
    return results


def analyze_with_model(text):
    """
    Titik masuk utama. Return list[dict] {word, suggestion, tag} bila model
    tersedia, atau None bila tidak (agar views.py fallback ke lexicon).
    Tag diberikan ARETA bila layanannya aktif; jika tidak, oleh infer_tag().
    """
    global LAST_TAGGER, _LAST_CORRECTED
    if not _lazy_load():
        return None

    # Sumber dinormalisasi dulu (harakat dibuang) agar perbandingan kata & ARETA
    # tidak menganggap "harakat hilang" sebagai kesalahan siswa.
    norm = normalize_text(text)                    # baris baru dipertahankan (batas paragraf/judul)
    try:
        sents, out_sents = _correct_sentences(norm)
    except Exception as exc:
        print(f"[ImlaScan] Inferensi gagal: {exc}")
        return None
    text = " ".join(sents)                         # sumber untuk perbandingan, sejajar dgn output
    corrected = " ".join(out_sents)
    _LAST_CORRECTED = corrected

    areta_result = _tag_with_areta(sents, out_sents)
    if areta_result is not None:
        LAST_TAGGER = "areta"
        return areta_result
    LAST_TAGGER = "heuristic"

    src_words = text.split()
    tgt_words = corrected.split()

    results = []
    sm = difflib.SequenceMatcher(a=src_words, b=tgt_words, autojunk=False)
    for op, i1, i2, j1, j2 in sm.get_opcodes():
        if op != "replace":
            continue  # 'equal' = tak berubah; insert/delete = ranah sintaksis (luar scope)
        for k in range(min(i2 - i1, j2 - j1)):
            wrong = re.sub(r"[^\w]", "", src_words[i1 + k], flags=re.UNICODE)
            correct = re.sub(r"[^\w]", "", tgt_words[j1 + k], flags=re.UNICODE)
            if not wrong or wrong == correct:
                continue
            tag = infer_tag(wrong, correct)
            if tag:
                results.append({"word": wrong, "suggestion": correct, "tag": tag})
    return results

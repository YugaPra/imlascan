"""
Pengganti pure-Python untuk python-Levenshtein 0.12.0 — HANYA fungsi yang dipakai ARETA.

Kenapa ada: ARETA memakai `Levenshtein.editops`. Versi 0.12.0 (yang dipakai tim CAMeL)
butuh compiler C di Windows; versi baru (0.21+) punya wheel Windows tetapi memilih
jalur edit berbeda saat seri (tie) -> ~0,2% label ARETA berubah. Modul ini menyalin
algoritma C 0.12.0 (lev_u_editops_find + ueditops_from_cost_matrix, _levenshtein.c)
baris demi baris, sehingga label identik dengan referensi dan tetap jalan tanpa compiler.
"""

__all__ = ["editops", "apply_edit", "distance"]


def _editops_unicode(s1, s2):
    len1, len2 = len(s1), len(s2)
    # buang prefiks yang sama
    o = 0
    while len1 > 0 and len2 > 0 and s1[o] == s2[o]:
        len1 -= 1
        len2 -= 1
        o += 1
    a = s1[o:o + len1]
    b = s2[o:o + len2]
    # buang sufiks yang sama
    while len1 > 0 and len2 > 0 and a[len1 - 1] == b[len2 - 1]:
        len1 -= 1
        len2 -= 1
    a, b = a[:len1], b[:len2]
    len1 += 1
    len2 += 1

    m = [0] * (len1 * len2)
    for i in range(len2):
        m[i] = i
    for i in range(1, len1):
        m[len2 * i] = i
    for i in range(1, len1):
        prev = (i - 1) * len2
        p = i * len2 + 1
        c1 = a[i - 1]
        x = i
        for j in range(len2 - 1):
            c3 = m[prev] + (c1 != b[j])
            prev += 1
            x += 1
            if x > c3:
                x = c3
            c3 = m[prev] + 1
            if x > c3:
                x = c3
            m[p] = x
            p += 1

    n = m[len1 * len2 - 1]
    if not n:
        return []
    ops = [None] * n
    pos = n
    i, j = len1 - 1, len2 - 1
    p = len1 * len2 - 1
    d = 0
    while i or j:
        if d < 0 and j and m[p] == m[p - 1] + 1:
            pos -= 1; j -= 1
            ops[pos] = ("insert", i + o, j + o)
            p -= 1
            continue
        if d > 0 and i and m[p] == m[p - len2] + 1:
            pos -= 1; i -= 1
            ops[pos] = ("delete", i + o, j + o)
            p -= len2
            continue
        if i and j and m[p] == m[p - len2 - 1] and a[i - 1] == b[j - 1]:
            i -= 1; j -= 1
            p -= len2 + 1
            d = 0
            continue
        if i and j and m[p] == m[p - len2 - 1] + 1:
            pos -= 1; i -= 1; j -= 1
            ops[pos] = ("replace", i + o, j + o)
            p -= len2 + 1
            d = 0
            continue
        if d == 0 and j and m[p] == m[p - 1] + 1:
            pos -= 1; j -= 1
            ops[pos] = ("insert", i + o, j + o)
            p -= 1
            d = -1
            continue
        if d == 0 and i and m[p] == m[p - len2] + 1:
            pos -= 1; i -= 1
            ops[pos] = ("delete", i + o, j + o)
            p -= len2
            d = 1
            continue
        raise AssertionError("lost in the cost matrix")
    return ops


def editops(s1, s2, *rest):
    if rest:
        raise NotImplementedError("lev_shim: editops 3-argumen tidak didukung (tidak dipakai ARETA).")
    return _editops_unicode(s1, s2)


def distance(s1, s2):
    return len(_editops_unicode(s1, s2))


def apply_edit(ops, s1, s2):
    """Terapkan daftar editops (tuple 3) ke s1 -> hasil (seperti Levenshtein.apply_edit)."""
    out, k = [], 0
    for op, sp, dp in ops:
        out.append(s1[k:sp])
        k = sp
        if op == "insert":
            out.append(s2[dp])
        elif op == "replace":
            out.append(s2[dp])
            k += 1
        elif op == "delete":
            k += 1
    out.append(s1[k:])
    return "".join(out)

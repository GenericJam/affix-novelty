"""Stream the big multilingual Numberbatch file and save per-language word
vectors (single alphabetic words only) to compact .npz files."""

import gzip

import numpy as np

SRC = "numberbatch/numberbatch-19.08.txt.gz"
# every language used across the experiments (English validation, Romance,
# Arabic, agglutinative, diminutive-broadening)
LANGS = {"en", "es", "fr", "it", "pt", "ar", "tr", "fi", "de", "ru"}

buf = {l: ([], []) for l in LANGS}
with gzip.open(SRC, "rt", encoding="utf-8") as f:
    f.readline()  # header: "<count> <dim>"
    for i, line in enumerate(f):
        sp = line.rstrip("\n").split(" ")
        parts = sp[0].split("/")  # /c/<lang>/<word>  -> ['', 'c', lang, word]
        if len(parts) != 4:
            continue
        lang, word = parts[2], parts[3]
        if lang not in LANGS or not word.isalpha():
            continue
        buf[lang][0].append(word)
        buf[lang][1].append(np.asarray(sp[1:], dtype=np.float32))
        if i % 1_000_000 == 0:
            print(f"  {i:,} lines; " + " ".join(f"{l}={len(buf[l][0])}" for l in sorted(LANGS)))

for l in LANGS:
    vocab, rows = buf[l]
    np.savez(
        f"numberbatch/nb_{l}.npz",
        vocab=np.array(vocab, dtype=object),
        mat=np.array(rows, dtype=np.float32),
    )
    print(f"saved nb_{l}.npz : {len(vocab)} words")

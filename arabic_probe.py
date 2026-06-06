"""
Exploratory Arabic probe (preliminary, see caveats in notes.md).

Arabic morphology is templatic (root-and-pattern), so the prefix/suffix contrast
does not apply. Instead we test the prediction the serial-position hypothesis
makes for a language whose anchor is a consonantal root preserved across every
derivation: derivational families should stay semantically COHERENT.

Measure: within-family mean pairwise cosine ("coherence"), compared to a random
baseline. We contrast:
  - English families grouped by MorphoLEX root, split into prefix-only vs
    suffix-only families. Hypothesis: prefix families are LESS coherent (they
    drift away from the root anchor); suffix families MORE coherent.
  - Arabic families grouped by an approximate triliteral root skeleton (heuristic,
    unvocalized). Hypothesis: high coherence vs baseline (root holds them together).

Heuristic Arabic root extraction is noisy; noise lowers measured coherence, so a
positive result is conservative.
"""
import random
import numpy as np
import gensim.downloader as api
from wordfreq import top_n_list, zipf_frequency
import core, morpholex_entries
from nbkv import NBKV

random.seed(0)


def mean_pairwise(words, kv):
    vs = [kv[w] for w in words if w in kv]
    if len(vs) < 2:
        return None
    sims = []
    for i in range(len(vs)):
        for j in range(i + 1, len(vs)):
            sims.append(core.cos(vs[i], vs[j]))
    return float(np.mean(sims))


def baseline(words, kv, n=4000):
    words = [w for w in words if w in kv]
    sims = []
    for _ in range(n):
        a, b = random.sample(words, 2)
        sims.append(core.cos(kv[a], kv[b]))
    return float(np.mean(sims))


def english():
    print("=== English: within-root family coherence (GloVe) ===")
    kv = api.load("glove-wiki-gigaword-300")
    entries = [e for e in morpholex_entries.load() if e["word"] in kv]
    fam = {}
    for e in entries:
        fam.setdefault(e["root"], []).append(e)
    coh = {"prefix": [], "suffix": []}
    for root, members in fam.items():
        if len(members) < 3:
            continue
        sides = {m["side"] for m in members}
        if len(sides) != 1:
            continue                         # pure prefix-family or pure suffix-family
        side = sides.pop()
        c = mean_pairwise([m["word"] for m in members], kv)
        if c is not None:
            coh[side].append(c)
    base = baseline([e["word"] for e in entries], kv)
    print(f"  random-pair baseline coherence: {base:+.3f}")
    for side in ("prefix", "suffix"):
        arr = np.array(coh[side])
        print(f"  {side}-only families (n={len(arr):4d}): mean within-family "
              f"coherence={arr.mean():+.3f}  (lift over baseline {arr.mean()-base:+.3f})")


def arabic():
    print("\n=== Arabic: approximate-root family coherence (Numberbatch) ===")
    kv = NBKV("ar")
    AL = "ال"
    proclitics = ["وال", "بال", "فال", "كال", "لل", "و", "ف", "ب", "ك", "ل"]
    longv = str.maketrans("", "", "اوىيءأإآ")  # drop long vowels / hamza forms
    drop_end = ["ات", "ون", "ين", "ها", "هم", "نا", "ة", "ه", "ي"]

    def skeleton(w):
        for p in [AL] + proclitics:
            if w.startswith(p) and len(w) - len(p) >= 3:
                w = w[len(p):]; break
        for s in drop_end:
            if w.endswith(s) and len(w) - len(s) >= 3:
                w = w[: -len(s)]; break
        return w.translate(longv)

    words = [w for w in top_n_list("ar", 60000)
             if w.isalpha() and w in kv and zipf_frequency(w, "ar") >= 2.5]
    fam = {}
    for w in words:
        sk = skeleton(w)
        if len(sk) == 3:                     # triliteral skeleton
            fam.setdefault(sk, []).append(w)
    families = {k: v for k, v in fam.items() if len(v) >= 3}
    coh = [mean_pairwise(v, kv) for v in families.values()]
    coh = [c for c in coh if c is not None]
    base = baseline(words, kv)
    print(f"  words={len(words)}  triliteral families(>=3)={len(families)}")
    print(f"  random-pair baseline coherence: {base:+.3f}")
    print(f"  mean within-root coherence: {np.mean(coh):+.3f}  "
          f"(lift over baseline {np.mean(coh)-base:+.3f})")
    # show a couple of example families
    for sk, v in list(families.items())[:5]:
        print(f"    root~{sk}: {' '.join(v[:6])}")


if __name__ == "__main__":
    english()
    arabic()

"""
Counterfactual probe: position vs grammatical-function.

The serial-position account and the "class-changing function" account agree on
English (suffixes happen to be the class-changing ones) but DISAGREE on
class-maintaining suffixes, above all diminutives/augmentatives. Holding position
constant (both are suffixes), we compare:
  - DIMINUTIVE / evaluative suffixes (class-maintaining: ito, illo, ino, etto...)
  - CLASS-CHANGING suffixes (cion, miento, dad, zione, mento...)
Prediction under serial-position: both lowish (root is first -> anchored).
Prediction under class-changing: diminutives drift like prefixes (high novelty).

Spanish and Italian, Numberbatch vectors. Vowel-restoration repair handles the
root alternation (bolsillo -> bols -> bolso; manzanilla -> manzan -> manzana).
"""
import numpy as np
from wordfreq import top_n_list, zipf_frequency, word_frequency
import core
from nbkv import NBKV

GROUPS = {
    "es": {
        "diminutive": ["ito", "ita", "illo", "illa", "ico", "ica", "ín",
                       "ino", "ina", "uelo", "ote", "azo", "ejo"],
        "class_changing": ["ción", "sión", "miento", "dad", "tad", "eza",
                           "ura", "ismo", "ista", "able", "ible", "oso",
                           "dor", "mente", "anza", "encia"],
    },
    "it": {
        "diminutive": ["ino", "ina", "etto", "etta", "ello", "ella",
                       "uccio", "otto", "one", "ona", "accio"],
        "class_changing": ["zione", "sione", "mento", "tà", "ezza", "ura",
                           "ismo", "ista", "abile", "ibile", "oso", "mente",
                           "tore", "aggio", "anza", "enza"],
    },
}


def build(lang, suffixes, label, kv, valid):
    suffixes = sorted(set(suffixes), key=len, reverse=True)
    cands = [w for w in top_n_list(lang, 60000)
             if w.isalpha() and len(w) >= 5 and w in kv
             and zipf_frequency(w, lang) >= 2.0]

    def root_of(stub):
        for c in (stub, stub + "o", stub + "a", stub + "e", stub + "ón"):
            if c in valid:
                return c
        return None

    seen = set()
    entries = []
    for w in cands:
        for af in suffixes:
            if w.endswith(af) and len(w) - len(af) >= 3:
                r = root_of(w[: -len(af)])
                if r and r != w and (af, r) not in seen:
                    seen.add((af, r))
                    entries.append(dict(word=w, side="suffix", affix=af, root=r,
                                        group=label, freq=word_frequency(w, lang),
                                        zipf=zipf_frequency(w, lang)))
                    break
    return entries


def run(lang):
    kv = NBKV(lang)
    valid = {w for w in top_n_list(lang, 60000)
             if w.isalpha() and zipf_frequency(w, lang) >= 2.8 and w in kv}
    dim = build(lang, GROUPS[lang]["diminutive"], "diminutive", kv, valid)
    cls = build(lang, GROUPS[lang]["class_changing"], "class_changing", kv, valid)
    # score together so offsets are learned per-affix consistently
    scored = core.score(dim + cls, kv)
    print(f"\n==== {lang} : suffix sub-types (position held constant) ====")
    for label in ("diminutive", "class_changing"):
        g = [e for e in scored if e["group"] == label]
        c = np.array([e["comp"] for e in g])
        nov = float((c < 0.15).mean()) if len(c) else float("nan")
        print(f"  {label:14s} n={len(g):4d}  mean_comp={c.mean():+.3f}  novel<0.15={nov:.1%}")
        ex = sorted(g, key=lambda e: e["comp"])[:8]
        print("     most-novel: " +
              ", ".join(f"{e['word']}({e['affix']}->{e['root']},{e['comp']:+.2f})" for e in ex))
    from scipy.stats import mannwhitneyu
    dc = [e["comp"] for e in scored if e["group"] == "diminutive"]
    cc = [e["comp"] for e in scored if e["group"] == "class_changing"]
    if dc and cc:
        _, p = mannwhitneyu(dc, cc, alternative="less")  # diminutive < class_changing?
        print(f"  diminutive MORE novel than class-changing?  Mann-Whitney p={p:.2e}")


if __name__ == "__main__":
    for lang in ("es", "it"):
        run(lang)

"""
Agglutinative suffixing languages (Turkish, Finnish). These are heavily suffixing
with almost no prefixes, so there is no within-language prefix contrast. The
serial-position account predicts their suffixal derivation should stay
transparent (the root is first = the anchor). We measure overall suffix-derivation
novelty and, where data allows, the diminutive (class-maintaining) vs
class-changing split, to see if the Romance dissociation replicates in a non-IE
family.

Caveats: no morphological analyzer; vowel harmony means each suffix has several
allomorphs; agglutination and consonant gradation/alternation mean the free-root
match is rough. Directional only.
"""
import numpy as np
from wordfreq import top_n_list, zipf_frequency, word_frequency
import core
from nbkv import NBKV

LANG = {
    "tr": {
        "class_changing": [
            "lik", "lık", "luk", "lük",        # -ness/-ship
            "ci", "cı", "cu", "cü", "çi", "çı", "çu", "çü",  # agentive
            "siz", "sız", "suz", "süz",        # -less
            "li", "lı", "lu", "lü",            # -with/having (adj)
            "lık",
        ],
        "diminutive": ["cik", "cık", "cuk", "cük", "cağız", "ceğiz"],
    },
    "fi": {
        "class_changing": [
            "uus", "yys", "us", "ys",          # -ness
            "ja", "jä",                        # agentive -er
            "ton", "tön",                      # -less
            "inen", "llinen",                  # adjective
            "minen",                           # action nominal
            "sto", "stö",                      # collective
            "la", "lä",                        # place
        ],
        "diminutive": ["nen"],                 # ambiguous (also adjective marker)
    },
}


def build(lang, suffixes, label, kv, valid):
    suffixes = sorted(set(suffixes), key=len, reverse=True)
    cands = [w for w in top_n_list(lang, 60000)
             if w.isalpha() and len(w) >= 5 and w in kv
             and zipf_frequency(w, lang) >= 2.0]

    def root_of(stub):
        # try as-is and a couple of light alternation repairs
        for c in (stub, stub + "k", stub + "t", stub + "p", stub[:-1] if stub else stub):
            if c in valid:
                return c
        return None

    seen, entries = set(), []
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
    cc = build(lang, LANG[lang]["class_changing"], "class_changing", kv, valid)
    dm = build(lang, LANG[lang]["diminutive"], "diminutive", kv, valid)
    scored = core.score(cc + dm, kv)
    print(f"\n==== {lang} (agglutinative, suffix-only) : vocab={len(kv)} ====")
    for label in ("class_changing", "diminutive"):
        g = [e for e in scored if e["group"] == label]
        if not g:
            print(f"  {label:14s} (no decompositions)"); continue
        c = np.array([e["comp"] for e in g])
        print(f"  {label:14s} n={len(g):4d}  mean_comp={c.mean():+.3f}  "
              f"novel<0.15={(c < 0.15).mean():.1%}")
        ex = sorted(g, key=lambda e: e["comp"])[:6]
        print("     most-novel: " +
              ", ".join(f"{e['word']}({e['affix']}->{e['root']},{e['comp']:+.2f})" for e in ex))


if __name__ == "__main__":
    for lang in ("tr", "fi"):
        run(lang)

"""
HUMAN-track confirmatory test of the referentiality hypothesis.

The per-suffix gradient suggested that derived words naming CONCRETE things drift,
while words for ABSTRACT qualities stay compositional. Tested here without any affix
hand-coding: join each derived word's compositionality to its concreteness rating
(Brysbaert et al. 2014, 1-5) and ask whether concreteness predicts drift, with
frequency and affix side controlled.

comp low = drifted. concreteness high = concrete. Referentiality predicts a
NEGATIVE concreteness-comp relationship (concrete -> drifted).
"""

import gensim.downloader as api
import numpy as np
import openpyxl
from scipy.stats import spearmanr

import core
import morpholex_entries


def load_concreteness(path="brysbaert_concreteness.xlsx"):
    wb = openpyxl.load_workbook(path, read_only=True)
    it = wb["Sheet1"].iter_rows(values_only=True)
    header = list(next(it))
    wi, ci = header.index("Word"), header.index("Conc.M")
    return {str(r[wi]).lower(): float(r[ci]) for r in it if r[wi] and r[ci] is not None}


def main():
    kv = api.load("glove-wiki-gigaword-300")
    conc = load_concreteness()
    scored = core.score(morpholex_entries.load(), kv)
    rows = [(e, conc[e["word"]]) for e in scored if e["word"] in conc]
    print(f"{len(rows)} of {len(scored)} scored words have a concreteness rating\n")

    comp = np.array([e["comp"] for e, _ in rows])
    cnc = np.array([c for _, c in rows])
    zipf = np.array([e["zipf"] for e, _ in rows])
    pref = np.array([1.0 if e["side"] == "prefix" else 0.0 for e, _ in rows])

    def sp(mask, label):
        r, p = spearmanr(cnc[mask], comp[mask])
        print(f"  {label:18s} n={mask.sum():5d}  Spearman(conc, comp) r={r:+.3f}  p={p:.1e}")

    print(
        "Spearman correlation of concreteness with compositionality "
        "(negative = concrete words drift):"
    )
    sp(np.ones(len(rows), bool), "all")
    sp(pref == 1, "prefix only")
    sp(pref == 0, "suffix only")

    print("\nMean compositionality by concreteness tertile (suffix only):")
    su = pref == 0
    cs, cm = cnc[su], comp[su]
    lo, hi = np.quantile(cs, [1 / 3, 2 / 3])
    for name, m in [
        ("abstract (low conc)", cs <= lo),
        ("mid", (cs > lo) & (cs <= hi)),
        ("concrete (high conc)", cs > hi),
    ]:
        print(
            f"  {name:22s} n={m.sum():5d}  mean_comp={cm[m].mean():+.3f}  "
            f"novel<0.15={(cm[m] < 0.15).mean():.1%}  mean_conc={cs[m].mean():.2f}"
        )

    # standardized OLS: does concreteness predict comp controlling for freq and side?
    def z(x):
        return (x - x.mean()) / x.std()

    X = np.column_stack([np.ones(len(rows)), z(cnc), z(zipf), pref])
    beta, *_ = np.linalg.lstsq(X, z(comp), rcond=None)
    print("\nStandardized OLS  comp ~ concreteness + frequency + is_prefix:")
    for name, b in zip(
        ["intercept", "concreteness", "log-frequency", "is_prefix"], beta, strict=False
    ):
        print(f"  {name:14s} beta={b:+.3f}")
    print(
        "  (negative concreteness beta => concrete words are less compositional, "
        "controlling for frequency and side)"
    )


if __name__ == "__main__":
    main()

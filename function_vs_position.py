"""
HUMAN-ONLY analysis (no LLM): is the real driver affix POSITION or grammatical
FUNCTION? Tested within English on validated MorphoLEX morphology, which is cleaner
than the orthographic Romance diminutives.

English prefixes are all class-maintaining, so position and function are confounded
in the bare prefix/suffix split. We break it by adding class-maintaining SUFFIXES
(-hood, -ship, -let, -ling, -ster: childhood, friendship, booklet, duckling). Three
groups:
  prefix        (class-maintaining, affix-first)
  suffix_CM     (class-maintaining, affix-last)   <- the critical cell
  suffix_CC     (class-changing,    affix-last)

POSITION predicts: prefix drifts; BOTH suffix groups stay transparent.
FUNCTION predicts: class-maintaining affixes drift (prefix AND suffix_CM); only
class-changing suffixes stay transparent. So suffix_CM should pattern with prefix.
"""
import numpy as np
from scipy.stats import mannwhitneyu
import gensim.downloader as api
import core, morpholex_entries

# Reliably class-MAINTAINING suffixes (preserve part of speech: N->N or A->A).
SUFFIX_CM = {"hood", "ship", "let", "ling", "ster", "eer", "ese", "dom",
             "ery", "ry", "age", "ist", "ism", "ite", "ling"}
# Reliably class-CHANGING suffixes (assign/alter part of speech).
SUFFIX_CC = {"tion", "ation", "sion", "ment", "ity", "ness", "er", "or",
             "ize", "ise", "ate", "ify", "able", "ible", "ous", "ive",
             "ful", "less", "ly", "al", "ial", "ic", "ical", "ant", "ent",
             "ary", "ory", "ish", "ward", "wise", "y", "ine", "esque"}


def group_of(e):
    if e["side"] == "prefix":
        return "prefix"            # all English prefixes are class-maintaining
    if e["affix"] in SUFFIX_CM:
        return "suffix_CM"
    if e["affix"] in SUFFIX_CC:
        return "suffix_CC"
    return None                    # unclassified suffix, drop from the clean contrast


def main():
    kv = api.load("glove-wiki-gigaword-300")
    entries = morpholex_entries.load()
    scored = core.score(entries, kv)
    for e in scored:
        e["grp"] = group_of(e)

    print("=== per-suffix compositionality (low = drifted), sorted ===")
    bys = {}
    for e in scored:
        if e["side"] == "suffix":
            bys.setdefault(e["affix"], []).append(e["comp"])
    rows = []
    for af, cs in bys.items():
        if len(cs) >= 12:
            lab = "CM" if af in SUFFIX_CM else ("CC" if af in SUFFIX_CC else "??")
            rows.append((af, lab, len(cs), float(np.mean(cs)), float((np.array(cs) < 0.15).mean())))
    for af, lab, n, m, nov in sorted(rows, key=lambda r: r[3]):
        print(f"  {af:7s} [{lab}] n={n:4d}  mean_comp={m:+.3f}  novel<0.15={nov:.1%}")

    print("\n=== group comparison (function vs position) ===")
    G = {}
    for g in ("prefix", "suffix_CM", "suffix_CC"):
        cs = np.array([e["comp"] for e in scored if e["grp"] == g])
        G[g] = cs
        print(f"  {g:10s} n={len(cs):4d}  mean_comp={cs.mean():+.3f}  "
              f"novel<0.15={(cs < 0.15).mean():.1%}")

    _, p_cm_cc = mannwhitneyu(G["suffix_CM"], G["suffix_CC"], alternative="less")
    _, p_cm_pre = mannwhitneyu(G["suffix_CM"], G["prefix"], alternative="two-sided")
    print(f"\n  suffix_CM more drifted than suffix_CC?  p={p_cm_cc:.2e}  (FUNCTION effect)")
    print(f"  suffix_CM vs prefix (two-sided)?        p={p_cm_pre:.2e}  "
          f"(large p => they pattern together, i.e. function not position)")

    print("\n  Reading: if suffix_CM clusters with prefix and far from suffix_CC,")
    print("  grammatical function drives the effect, not affix position.")


if __name__ == "__main__":
    main()

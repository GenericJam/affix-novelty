"""
Are prefixes more likely than suffixes to generate "new words"?

A "new word" here = an affixed word whose meaning has drifted away from the
literal sum (affix + root), like remember (re + member) no longer meaning
"to re-member". We detect drift with compositional distributional semantics:

  compositionality(word) = cosine( vec(word),  vec(root) + offset(affix) )

where offset(affix) is the typical semantic shift that affix imparts, learned
(two-pass, self-cleaning) from all of that affix's own decomposable words.

  high compositionality  -> meaning is predictable from parts  (transparent)
  low  compositionality  -> meaning is NOT predictable         (novel / drifted)

Decomposition is orthographic: affix stripped, remaining root must be a real
standalone word (light spelling repair so the suffix side is treated as fairly
as the prefix side). Inflectional suffixes (-s, -ed, -ing) are excluded.
"""

import csv
import numpy as np
import gensim.downloader as api
from wordfreq import top_n_list, zipf_frequency
from scipy.stats import mannwhitneyu

# ---------------------------------------------------------------- config
WORD_FREQ_FLOOR = 2.5     # candidate word must be at least this common (zipf)
ROOT_FREQ_FLOOR = 3.0     # root must be at least this common (kills junk splits)
MIN_ROOT_LEN    = 4       # 4+ char roots: kills co+ach, de+als, nev+er, etc.
MIN_PAIRS       = 8       # affix needs this many pairs to learn an offset
N_WORDS         = 60000   # how many of the most common English words to scan
THRESHOLDS      = [0.10, 0.15, 0.20]   # compositionality below this = "novel"

# Hand-vetted, unambiguously real affixes only. The short Latinate ones
# (co-, de-, con-, ex-, en-, -er, -or, -al, -ent, -ory ...) are dropped: they
# generate too many orthographic-coincidence splits (color, decade, surgery).
PREFIXES = [
    "re","un","dis","pre","mis","over","under","non","inter","super",
    "anti","counter","out","fore","trans","semi","multi",
]
# derivational only; NO -s / -ed / -ing (those are inflection, not word-making)
SUFFIXES = [
    "ness","less","ful","ment","tion","sion","ation","ship","hood","dom",
    "able","ible","ize","ist","ism","ous","ive","ward","wise","like","ity",
]
# longest-first so we strip the maximal affix and don't double-count nested ones
PREFIXES = sorted(set(PREFIXES), key=len, reverse=True)
SUFFIXES = sorted(set(SUFFIXES), key=len, reverse=True)

# ---------------------------------------------------------------- load data
print("loading GloVe vectors (cached after first run)...")
KV = api.load("glove-wiki-gigaword-300")

def vec(w):
    return KV[w] if w in KV else None

def cos(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

print(f"building word lists from top {N_WORDS} English words...")
common = [w for w in top_n_list("en", N_WORDS) if w.isalpha() and w.islower()]
# valid roots: reasonably common AND have a vector
VALID = {w for w in common if zipf_frequency(w, "en") >= ROOT_FREQ_FLOOR and w in KV}
# candidate words to decompose
CANDS = [w for w in common
         if zipf_frequency(w, "en") >= WORD_FREQ_FLOOR and len(w) >= 4 and w in KV]
print(f"  {len(VALID)} valid roots, {len(CANDS)} candidate words")

# ---------------------------------------------------------------- decomposition
def find_root(stub):
    """Return a real standalone root for `stub`, trying light spelling repair."""
    variants = [stub]
    if stub.endswith("i"):                           # happi -> happy (reliable)
        variants.append(stub[:-1] + "y")
    # NOTE: deliberately NOT trying stub+"e" or de-doubling; those manufacture
    # fake roots (surg->surge, so "surgery" looks like ery+surge).
    for v in variants:
        if v in VALID:
            return v
    return None

def decompose(word, affixes, side):
    """Return (affix, root) for the longest matching affix, or None."""
    for af in affixes:                               # already longest-first
        if side == "prefix" and word.startswith(af) and len(word) - len(af) >= MIN_ROOT_LEN:
            root = find_root(word[len(af):])
            if root and root != word:
                return af, root
        if side == "suffix" and word.endswith(af) and len(word) - len(af) >= MIN_ROOT_LEN:
            root = find_root(word[: -len(af)])
            if root and root != word:
                return af, root
    return None

records = []   # dict per decomposable word
for w in CANDS:
    for side, affixes in (("prefix", PREFIXES), ("suffix", SUFFIXES)):
        hit = decompose(w, affixes, side)
        if hit:
            af, root = hit
            records.append(dict(word=w, root=root, affix=af, side=side,
                                zipf=zipf_frequency(w, "en")))

print(f"  {sum(r['side']=='prefix' for r in records)} prefix decompositions, "
      f"{sum(r['side']=='suffix' for r in records)} suffix decompositions")

# ---------------------------------------------------------------- learn offsets
def learn_offsets(recs):
    """Two-pass per-affix offset: pass 1 = mean; pass 2 = mean over the more
    compositional half, so drifted outliers don't poison the learned shift."""
    by_affix = {}
    for r in recs:
        by_affix.setdefault((r["side"], r["affix"]), []).append(r)
    offsets = {}
    for key, group in by_affix.items():
        diffs = [vec(r["word"]) - vec(r["root"]) for r in group]
        if len(diffs) < MIN_PAIRS:
            continue
        off1 = np.mean(diffs, axis=0)
        comps = [cos(vec(r["word"]), vec(r["root"]) + off1) for r in group]
        med = np.median(comps)
        keep = [d for d, c in zip(diffs, comps) if c >= med]
        offsets[key] = np.mean(keep, axis=0) if keep else off1
    return offsets

OFFSETS = learn_offsets(records)
print(f"  learned offsets for {len(OFFSETS)} affixes (>= {MIN_PAIRS} pairs each)")

# ---------------------------------------------------------------- score
scored = []
for r in records:
    key = (r["side"], r["affix"])
    if key not in OFFSETS:
        continue
    comp = cos(vec(r["word"]), vec(r["root"]) + OFFSETS[key])
    r["comp"] = comp
    scored.append(r)

# ---------------------------------------------------------------- validation
print("\n--- calibration on known cases (low comp = novel) ---")
val = {("remember","prefix"),("report","prefix"),("recover","prefix"),
       ("understand","prefix"),("business","suffix"),("fellowship","suffix"),
       ("rerun","prefix"),("reopen","prefix"),("happiness","suffix"),
       ("runner","suffix"),("hardship","suffix"),("treatment","suffix")}
idx = {(r["word"], r["side"]): r for r in scored}
for w, side in sorted(val):
    r = idx.get((w, side))
    if r:
        print(f"  {w:12s} = {r['affix']}+{r['root']:10s} comp={r['comp']:+.3f}")
    else:
        print(f"  {w:12s} ({side}) not decomposed under current rules")

# ---------------------------------------------------------------- aggregate
def summarize(side):
    rs = [r for r in scored if r["side"] == side]
    comps = np.array([r["comp"] for r in rs])
    out = dict(side=side, n=len(rs), mean=comps.mean(), median=np.median(comps))
    for t in THRESHOLDS:
        out[f"rate<{t}"] = float((comps < t).mean())
    return out, comps

print("\n--- prefix vs suffix ---")
ps, pc = summarize("prefix")
ss, sc = summarize("suffix")
for s in (ps, ss):
    line = f"  {s['side']:7s} n={s['n']:6d}  mean={s['mean']:+.3f}  median={s['median']:+.3f}  "
    line += "  ".join(f"novel{k.split('<')[1]}={s[k]:.1%}" for k in s if k.startswith("rate"))
    print(line)

U, p = mannwhitneyu(pc, sc, alternative="two-sided")
print(f"\n  Mann-Whitney U on compositionality: p={p:.2e} "
      f"(prefix {'lower=more novel' if pc.mean()<sc.mean() else 'higher'})")

# frequency control: compare novelty rate within zipf buckets
print("\n--- novelty rate (comp<0.15) controlled for word frequency ---")
print(f"  {'zipf bucket':14s}{'prefix n':>10s}{'prefix novel':>14s}{'suffix n':>10s}{'suffix novel':>14s}")
buckets = [(2.5,3.0),(3.0,3.5),(3.5,4.0),(4.0,4.5),(4.5,7.0)]
for lo, hi in buckets:
    pr = [r for r in scored if r["side"]=="prefix" and lo <= r["zipf"] < hi]
    su = [r for r in scored if r["side"]=="suffix" and lo <= r["zipf"] < hi]
    prr = np.mean([r["comp"]<0.15 for r in pr]) if pr else float("nan")
    sur = np.mean([r["comp"]<0.15 for r in su]) if su else float("nan")
    print(f"  [{lo:.1f},{hi:.1f}){'':4s}{len(pr):10d}{prr:14.1%}{len(su):10d}{sur:14.1%}")

# ---------------------------------------------------------------- per-affix + examples
def affix_table(side, top=12):
    rows = {}
    for r in scored:
        if r["side"] != side:
            continue
        rows.setdefault(r["affix"], []).append(r)
    out = []
    for af, group in rows.items():
        comps = np.array([g["comp"] for g in group])
        out.append((af, len(group), comps.mean(), float((comps < 0.15).mean())))
    out.sort(key=lambda x: x[3], reverse=True)
    print(f"\n--- most word-generating {side}es (by novelty rate, n>=15) ---")
    for af, n, m, rate in [o for o in out if o[1] >= 15][:top]:
        exs = sorted([r for r in scored if r["side"]==side and r["affix"]==af],
                     key=lambda r: r["comp"])[:4]
        exl = ", ".join(f"{e['word']}({e['comp']:+.2f})" for e in exs)
        print(f"  {af+'-':10s} n={n:4d}  novel={rate:5.1%}  mean={m:+.2f}   e.g. {exl}")

affix_table("prefix")
affix_table("suffix")

# ---------------------------------------------------------------- dump
with open("results.csv", "w", newline="") as f:
    wtr = csv.DictWriter(f, fieldnames=["word","root","affix","side","zipf","comp"])
    wtr.writeheader()
    for r in sorted(scored, key=lambda r: r["comp"]):
        wtr.writerow({k: r[k] for k in ["word","root","affix","side","zipf","comp"]})
print(f"\nwrote {len(scored)} scored words to results.csv (sorted most-novel first)")

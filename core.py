"""
Reusable compositionality core, shared by every experiment (English, other
embeddings, historical decades, other languages, LLM representations).

An "entry" is a dict with at least:
    word, side ('prefix'|'suffix'), affix, root, freq, zipf
Vectors come from any object supporting `w in kv` and `kv[w]` (gensim
KeyedVectors, or a thin wrapper around a dict of numpy arrays).

The metric:
    comp(word) = cosine( vec(word),  affix_offset(affix) + root_meaning )
high = predictable from parts (transparent); low = drifted / novel.
root_meaning = vec(root) for a free singleton root, else a leave-one-out
reconstruction from the root's morphological family.
"""
import numpy as np
from scipy.stats import mannwhitneyu

THRESHOLDS = (0.10, 0.15, 0.20)


def cos(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def prepare(entries, kv):
    """Keep entries whose word has a vector; tag root_free."""
    out = []
    for e in entries:
        if e["word"] in kv:
            e = dict(e)
            e["root_free"] = e["root"] in kv
            out.append(e)
    return out


def learn_offsets(entries, kv, min_pairs=8):
    by = {}
    for e in entries:
        if e["root_free"]:
            by.setdefault((e["side"], e["affix"]), []).append(e)
    offs = {}
    for key, grp in by.items():
        diffs = [kv[e["word"]] - kv[e["root"]] for e in grp]
        if len(diffs) < min_pairs:
            continue
        o1 = np.mean(diffs, axis=0)
        comps = [cos(kv[e["word"]], kv[e["root"]] + o1) for e in grp]
        med = np.median(comps)
        keep = [d for d, c in zip(diffs, comps) if c >= med]
        offs[key] = np.mean(keep, axis=0) if keep else o1
    return offs


def score(entries, kv, min_pairs=8):
    entries = prepare(entries, kv)
    offs = learn_offsets(entries, kv, min_pairs)
    family = {}
    for e in entries:
        family.setdefault(e["root"], []).append(e)

    def reconstruct(root, exclude):
        hats = []
        for s in family[root]:
            if s["word"] == exclude:
                continue
            k = (s["side"], s["affix"])
            if k in offs:
                hats.append(kv[s["word"]] - offs[k])
        return np.mean(hats, axis=0) if hats else None

    scored = []
    for e in entries:
        k = (e["side"], e["affix"])
        if k not in offs:
            continue
        R = reconstruct(e["root"], e["word"])
        method = "family"
        if R is None:
            if not e["root_free"]:
                continue
            R = kv[e["root"]]; method = "freeroot"
        e["comp"] = cos(kv[e["word"]], offs[k] + R)
        e["method"] = method
        scored.append(e)
    return scored


def _novelty(items, t, weighted):
    c = np.array([e["comp"] for e in items])
    w = np.array([e["freq"] for e in items]) if weighted else np.ones(len(items))
    if w.sum() == 0:
        return float("nan")
    return float(w[c < t].sum() / w.sum())


def summarize(scored, bands=((1.5, 2.5), (2.5, 3.0), (3.0, 3.5), (3.5, 4.0), (4.0, 7.0))):
    pref = [e for e in scored if e["side"] == "prefix"]
    suff = [e for e in scored if e["side"] == "suffix"]
    out = {"n_prefix": len(pref), "n_suffix": len(suff)}
    for t in THRESHOLDS:
        out[f"type_<{t}"] = {"prefix": _novelty(pref, t, False),
                             "suffix": _novelty(suff, t, False)}
        out[f"token_<{t}"] = {"prefix": _novelty(pref, t, True),
                              "suffix": _novelty(suff, t, True)}
    out["mean_comp"] = {"prefix": float(np.mean([e["comp"] for e in pref])),
                        "suffix": float(np.mean([e["comp"] for e in suff]))}
    out["boundroot_type_<0.15"] = {
        "prefix": _novelty([e for e in pref if not e["root_free"]], 0.15, False),
        "suffix": _novelty([e for e in suff if not e["root_free"]], 0.15, False)}
    if pref and suff:
        _, p = mannwhitneyu([e["comp"] for e in pref], [e["comp"] for e in suff],
                            alternative="two-sided")
        out["mannwhitney_p"] = float(p)
    out["by_freq_<0.15"] = []
    for lo, hi in bands:
        pr = [e for e in pref if lo <= e["zipf"] < hi]
        su = [e for e in suff if lo <= e["zipf"] < hi]
        out["by_freq_<0.15"].append({
            "band": f"{lo}-{hi}", "n_prefix": len(pr), "n_suffix": len(su),
            "prefix": _novelty(pr, 0.15, False), "suffix": _novelty(su, 0.15, False)})
    return out


def print_summary(name, s):
    print(f"\n==== {name} ====")
    print(f"  n: prefix={s['n_prefix']}  suffix={s['n_suffix']}")
    t = s["type_<0.15"]; tk = s["token_<0.15"]; b = s["boundroot_type_<0.15"]
    print(f"  type  novel<0.15:  prefix={t['prefix']:.1%}  suffix={t['suffix']:.1%}")
    print(f"  token novel<0.15:  prefix={tk['prefix']:.1%}  suffix={tk['suffix']:.1%}")
    print(f"  bound-root  <0.15: prefix={b['prefix']:.1%}  suffix={b['suffix']:.1%}")
    print(f"  mean comp: prefix={s['mean_comp']['prefix']:+.3f}  "
          f"suffix={s['mean_comp']['suffix']:+.3f}   p={s.get('mannwhitney_p', float('nan')):.1e}")
    print("  by frequency band (novel<0.15):")
    for r in s["by_freq_<0.15"]:
        print(f"    {r['band']:9s} prefix={r['prefix']:.1%} (n={r['n_prefix']:4d})"
              f"   suffix={r['suffix']:.1%} (n={r['n_suffix']:4d})")

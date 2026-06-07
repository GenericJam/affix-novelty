"""
Control for the nonce-word probe (nonce_test.py).

Confound: novel prefixed forms may score as non-compositional simply because the
affix offset is learned from real prefix words that are themselves opaque, so the
learned prefix offset points in a drift-contaminated direction. Fix: relearn each
affix's offset from only its MOST TRANSPARENT real words (top fraction by
compositionality), giving a clean "compositional shift", then re-score the novel
forms. Sweep the fraction.

  - gap COLLAPSES toward zero as the offset gets cleaner -> it was contamination.
  - gap PERSISTS -> GPT-2 genuinely represents novel prefixed forms as less
    compositional even relative to the clean prefix shift (an internalized bias).
"""

import numpy as np
from scipy.stats import mannwhitneyu

import core
import morpholex_entries
from nonce_test import deanisotropize, embed, novel_combos


def clean_offsets(real_scored, kv, frac):
    """Offset per affix from the top `frac` most-compositional free-root real words."""
    by = {}
    for e in real_scored:
        if e["root_free"]:
            by.setdefault((e["side"], e["affix"]), []).append(e)
    offs = {}
    for key, grp in by.items():
        grp = sorted(grp, key=lambda e: e["comp"], reverse=True)
        keep = grp[: max(2, int(len(grp) * frac))]
        if len(keep) < 2:
            continue
        offs[key] = np.mean([kv[e["word"]] - kv[e["root"]] for e in keep], axis=0)
    return offs


def main():
    real = morpholex_entries.load()
    novel = novel_combos()
    vocab = sorted(
        {e["word"] for e in real}
        | {e["root"] for e in real}
        | {n["word"] for n in novel}
        | {n["root"] for n in novel}
    )
    print(f"embedding {len(vocab)} strings with gpt2 ...")
    d = deanisotropize(embed(vocab))
    kv = type(
        "KV",
        (),
        {"d": d, "__contains__": lambda s, w: w in s.d, "__getitem__": lambda s, w: s.d[w]},
    )()

    real_scored = core.score(real, kv)  # standard two-pass offsets -> comp per real word

    def score_novel(offs, side):
        out = []
        for n in novel:
            if n["side"] != side:
                continue
            k = (n["side"], n["affix"])
            if k in offs and n["word"] in kv and n["root"] in kv:
                out.append(core.cos(kv[n["word"]], offs[k] + kv[n["root"]]))
        return np.array(out)

    print("\noffset learned from top-fraction most-transparent real words:")
    print(f"  {'frac':>6s}{'novel pfx':>11s}{'novel sfx':>11s}{'gap':>8s}{'p(pfx<sfx)':>12s}")
    for frac in (1.0, 0.5, 0.25, 0.1):
        offs = clean_offsets(real_scored, kv, frac)
        pc, sc = score_novel(offs, "prefix"), score_novel(offs, "suffix")
        _, p = mannwhitneyu(pc, sc, alternative="less")
        print(
            f"  {frac:6.2f}{pc.mean():11.3f}{sc.mean():11.3f}{sc.mean() - pc.mean():+8.3f}{p:12.1e}"
        )
    print("\n  gap shrinking toward 0 as frac drops => contamination explained it;")
    print("  gap persisting => genuine internalized bias on unseen forms.")


if __name__ == "__main__":
    main()

"""
LLM-track independent test: does GPT-2 GENERALIZE the prefix<suffix asymmetry to
NOVEL words it never saw, or only reproduce it per-word from training?

If the asymmetry is purely memorized, novel affixed combinations (which are all
genuinely compositional, e.g. "unsmall" = not small, "jumpness" = state of
jumping) should look equally compositional on both sides. If GPT-2 has an
internalized structural bias, it will represent the novel PREFIXED forms as less
compositional than the novel SUFFIXED forms, applying the pattern to words that
should not drift. A real generalization would be a shared-mechanism candidate
(beyond mere inheritance of the training distribution).

Offsets are learned in GPT-2 space from real MorphoLEX words, then applied to the
novel forms.
"""

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer
from wordfreq import word_frequency

import core
import morpholex_entries

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"

PREFIXES = ["re", "un", "over", "under", "mis", "dis", "out", "pre"]
SUFFIXES = ["ness", "less", "able", "er", "ish", "ful", "y", "ment"]
ROOTS = [
    "jump",
    "swim",
    "walk",
    "talk",
    "paint",
    "climb",
    "push",
    "pull",
    "kick",
    "throw",
    "catch",
    "build",
    "small",
    "large",
    "quick",
    "slow",
    "bright",
    "dark",
    "warm",
    "cold",
    "soft",
    "hard",
    "sharp",
    "clean",
    "dream",
    "guess",
    "laugh",
    "whisper",
    "stretch",
    "float",
    "wander",
    "gather",
    "scatter",
    "polish",
    "green",
    "blue",
    "round",
    "square",
    "smooth",
    "rough",
    "quiet",
    "loud",
    "sweet",
    "sour",
    "fresh",
    "tired",
    "book",
    "chair",
    "stone",
    "river",
    "cloud",
    "garden",
    "window",
    "mirror",
    "candle",
]


def novel_combos():
    out = []
    for r in ROOTS:
        for p in PREFIXES:
            w = p + r
            if word_frequency(w, "en") < 1e-7:  # genuinely non-existent
                out.append({"word": w, "side": "prefix", "affix": p, "root": r})
        for s in SUFFIXES:
            w = r + s
            if word_frequency(w, "en") < 1e-7:
                out.append({"word": w, "side": "suffix", "affix": s, "root": r})
    return out


@torch.no_grad()
def embed(words, batch=128):
    tok = AutoTokenizer.from_pretrained("gpt2")
    tok.pad_token = tok.eos_token
    model = AutoModel.from_pretrained("gpt2").to(DEVICE).eval()
    out = {}
    for i in range(0, len(words), batch):
        chunk = words[i : i + batch]
        enc = tok(chunk, return_tensors="pt", padding=True, return_special_tokens_mask=True).to(
            DEVICE
        )
        spec = enc.pop("special_tokens_mask")
        mask = (enc["attention_mask"] * (1 - spec)).unsqueeze(-1).float()
        h = model(**enc).last_hidden_state
        pooled = (h * mask).sum(1) / mask.sum(1).clamp(min=1)
        for w, v in zip(chunk, pooled, strict=False):
            out[w] = v.float().cpu().numpy()
    return out


def deanisotropize(d):
    words = list(d)
    M = np.stack([d[w] for w in words]).astype(np.float64)
    M = M - M.mean(0, keepdims=True)
    _, _, Vt = np.linalg.svd(M, full_matrices=False)
    M = M - np.outer(M @ Vt[0], Vt[0])
    return {w: M[i].astype(np.float32) for i, w in enumerate(words)}


def main():
    real = morpholex_entries.load()
    novel = novel_combos()
    print(
        f"{len(novel)} novel combos "
        f"({sum(n['side'] == 'prefix' for n in novel)} prefix, "
        f"{sum(n['side'] == 'suffix' for n in novel)} suffix)"
    )

    vocab = sorted(
        {e["word"] for e in real}
        | {e["root"] for e in real}
        | {n["word"] for n in novel}
        | {n["root"] for n in novel}
    )
    print(f"embedding {len(vocab)} strings with gpt2 ...")
    kv = type(
        "KV",
        (),
        {
            "d": deanisotropize(embed(vocab)),
            "__contains__": lambda s, w: w in s.d,
            "__getitem__": lambda s, w: s.d[w],
        },
    )()

    real_prepared = core.prepare(real, kv)
    offs = core.learn_offsets(real_prepared, kv)

    def score(items):
        out = []
        for n in items:
            k = (n["side"], n["affix"])
            if k in offs and n["word"] in kv and n["root"] in kv:
                out.append(core.cos(kv[n["word"]], offs[k] + kv[n["root"]]))
        return np.array(out)

    pc = score([n for n in novel if n["side"] == "prefix"])
    sc = score([n for n in novel if n["side"] == "suffix"])
    from scipy.stats import mannwhitneyu

    _, p = mannwhitneyu(pc, sc, alternative="less")
    print("\nNovel (never-seen) compositional forms, scored in GPT-2 space:")
    print(f"  novel prefix words: n={len(pc):4d}  mean_comp={pc.mean():+.3f}")
    print(f"  novel suffix words: n={len(sc):4d}  mean_comp={sc.mean():+.3f}")
    print(f"  prefix < suffix (model imposes drift on novel prefixed forms)?  p={p:.2e}")
    print("\n  Reading: a significantly lower prefix mean means GPT-2 generalizes the")
    print("  asymmetry to words it never saw (internalized bias, not memorization).")


if __name__ == "__main__":
    main()

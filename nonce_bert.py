"""
Bidirectional version of the nonce-word probe. Runs the novel-form test plus its
transparent-offset control for GPT-2 (autoregressive) and BERT (bidirectional),
with identical mean-pooled extraction.

Two questions at once:
  - Architecture independence: does BERT also generalize the prefix<suffix bias to
    words it never saw?
  - Tokenization: BERT uses WordPiece, GPT-2 byte-BPE. If both show it, the effect
    is not a quirk of one tokenizer.
"""

import numpy as np
from scipy.stats import mannwhitneyu

import core
import morpholex_entries
from bert_control import DictKV, deanisotropize, embed
from nonce_control import clean_offsets
from nonce_test import novel_combos

MODELS = ["gpt2", "bert-base-uncased"]


def main():
    real = morpholex_entries.load()
    novel = novel_combos()
    vocab = sorted(
        {e["word"] for e in real}
        | {e["root"] for e in real}
        | {n["word"] for n in novel}
        | {n["root"] for n in novel}
    )

    def score_novel(offs, kv, side):
        out = [
            core.cos(kv[n["word"]], offs[(n["side"], n["affix"])] + kv[n["root"]])
            for n in novel
            if n["side"] == side
            and (n["side"], n["affix"]) in offs
            and n["word"] in kv
            and n["root"] in kv
        ]
        return np.array(out)

    for model in MODELS:
        print(f"\nembedding {len(vocab)} strings with {model} ...")
        kv = DictKV(deanisotropize(embed(vocab, model), n_pcs=1))
        real_scored = core.score(real, kv)
        print(f"==== {model} : novel-form compositionality ====")
        print(f"  {'offset frac':>12s}{'novel pfx':>11s}{'novel sfx':>11s}{'gap':>8s}{'p':>11s}")
        for frac in (1.0, 0.25):
            offs = clean_offsets(real_scored, kv, frac)
            pc, sc = score_novel(offs, kv, "prefix"), score_novel(offs, kv, "suffix")
            _, p = mannwhitneyu(pc, sc, alternative="less")
            gap = sc.mean() - pc.mean()
            print(f"  {frac:12.2f}{pc.mean():11.3f}{sc.mean():11.3f}{gap:+8.3f}{p:11.1e}")
    print("\n  Both models showing a persistent gap => the generalizing bias is")
    print("  architecture-independent and not a single-tokenizer artifact.")


if __name__ == "__main__":
    main()

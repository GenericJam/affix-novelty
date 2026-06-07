"""
LLM test of the serial-position hypothesis. An autoregressive model reads a word
left to right and must traverse the prefix tokens before resolving the word. We
take each word's representation to be GPT-2's last-layer hidden state at the
final sub-token (the state after reading the whole word), then run the identical
compositionality metric. If a next-token predictor shows the same prefix<suffix
asymmetry, the serial-position mechanism is not specific to human memory.
"""

import json

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

import core
import morpholex_entries

MODEL = "gpt2"
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"

tok = AutoTokenizer.from_pretrained(MODEL)
tok.pad_token = tok.eos_token
model = AutoModel.from_pretrained(MODEL).to(DEVICE).eval()


@torch.no_grad()
def embed(words, batch=128):
    out = {}
    for i in range(0, len(words), batch):
        chunk = words[i : i + batch]
        enc = tok([" " + w for w in chunk], return_tensors="pt", padding=True).to(DEVICE)
        h = model(**enc).last_hidden_state  # (B, T, 768)
        last = enc["attention_mask"].sum(1) - 1  # final real token per row
        vecs = h[torch.arange(len(chunk)), last]  # (B, 768)
        for w, v in zip(chunk, vecs, strict=False):
            out[w] = v.float().cpu().numpy()
        if i % (batch * 10) == 0:
            print(f"  embedded {i + len(chunk)}/{len(words)}")
    return out


class DictKV:
    def __init__(self, d):
        self.d = d

    def __contains__(self, w):
        return w in self.d

    def __getitem__(self, w):
        return self.d[w]


def deanisotropize(d, n_pcs=1):
    """GPT-2 hidden states are strongly anisotropic (a dominant shared
    direction makes all cosines ~1). Standard fix: subtract the global mean and
    project out the top principal component(s). Note the affix offset is a
    difference of vectors and is unchanged by centering; only the final cosine
    is affected, which is the point."""
    words = list(d)
    M = np.stack([d[w] for w in words]).astype(np.float64)
    M = M - M.mean(0, keepdims=True)
    if n_pcs:
        _, _, Vt = np.linalg.svd(M, full_matrices=False)
        for i in range(n_pcs):
            v = Vt[i]
            M = M - np.outer(M @ v, v)
    return {w: M[i].astype(np.float32) for i, w in enumerate(words)}


def main():
    entries = morpholex_entries.load()
    vocab = sorted({e["word"] for e in entries} | {e["root"] for e in entries})
    print(f"embedding {len(vocab)} unique strings with {MODEL} on {DEVICE} ...")
    raw = embed(vocab)
    keep = None
    conditions = [
        ("raw (anisotropic)", raw),
        ("centered", deanisotropize(raw, n_pcs=0)),
        ("centered + top-1 PC removed", deanisotropize(raw, n_pcs=1)),
        ("centered + top-2 PCs removed", deanisotropize(raw, n_pcs=2)),
    ]
    keep = None
    for label, d in conditions:
        scored = core.score(entries, DictKV(d))
        s = core.summarize(scored)
        core.print_summary(f"{MODEL} last-token : {label}", s)
        if "top-1" in label:
            with open("stats_llm.json", "w") as f:
                json.dump(s, f, indent=2)
            keep = scored
    scored = keep
    for side in ("prefix", "suffix"):
        ex = sorted([e for e in scored if e["side"] == side], key=lambda e: e["comp"])[:8]
        print(f"  most-novel {side}: " + ", ".join(f"{e['word']}({e['comp']:+.2f})" for e in ex))
    print("\nwrote stats_llm.json")


if __name__ == "__main__":
    main()

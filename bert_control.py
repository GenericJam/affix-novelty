"""
BERT control for the LLM result.

The prefix<suffix asymmetry already exists in static (non-directional) embeddings
(Lazaridou et al. 2013), so showing it in GPT-2 does not by itself prove the
autoregressive left-to-right interpretation. This control compares an
AUTOREGRESSIVE model (GPT-2) with a BIDIRECTIONAL one (BERT) using IDENTICAL
extraction (mean-pooled sub-token last-layer hidden states, same anisotropy
correction), so architecture is the only difference.

  - If BERT shows a WEAKER/absent asymmetry than GPT-2 -> autoregressive order
    contributes -> processing-order interpretation supported.
  - If BERT shows it AS STRONGLY -> it is distributional, not processing-order ->
    drop the autoregressive interpretation.
"""
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
import core, morpholex_entries

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
MODELS = ["gpt2", "bert-base-uncased"]


@torch.no_grad()
def embed(words, model_name, batch=128):
    tok = AutoTokenizer.from_pretrained(model_name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModel.from_pretrained(model_name).to(DEVICE).eval()
    out = {}
    for i in range(0, len(words), batch):
        chunk = words[i:i + batch]
        enc = tok(chunk, return_tensors="pt", padding=True,
                  return_special_tokens_mask=True).to(DEVICE)
        spec = enc.pop("special_tokens_mask")
        mask = (enc["attention_mask"] * (1 - spec)).unsqueeze(-1).float()  # (B,T,1)
        h = model(**enc).last_hidden_state                                 # (B,T,H)
        pooled = (h * mask).sum(1) / mask.sum(1).clamp(min=1)              # mean-pool real word tokens
        for w, v in zip(chunk, pooled):
            out[w] = v.float().cpu().numpy()
    return out


def deanisotropize(d, n_pcs=1):
    words = list(d)
    M = np.stack([d[w] for w in words]).astype(np.float64)
    M = M - M.mean(0, keepdims=True)
    for i in range(n_pcs):
        _, _, Vt = np.linalg.svd(M, full_matrices=False)
        v = Vt[0]; M = M - np.outer(M @ v, v)
    return {w: M[i].astype(np.float32) for i, w in enumerate(words)}


class DictKV:
    def __init__(self, d): self.d = d
    def __contains__(self, w): return w in self.d
    def __getitem__(self, w): return self.d[w]


def asymmetry(scored):
    p = np.array([e["comp"] for e in scored if e["side"] == "prefix"])
    s = np.array([e["comp"] for e in scored if e["side"] == "suffix"])
    from scipy.stats import mannwhitneyu
    _, pv = mannwhitneyu(p, s, alternative="two-sided")
    return dict(prefix_mean=p.mean(), suffix_mean=s.mean(), gap=s.mean() - p.mean(),
                prefix_nov=float((p < 0.15).mean()), suffix_nov=float((s < 0.15).mean()),
                p=pv)


def main():
    entries = morpholex_entries.load()
    vocab = sorted({e["word"] for e in entries} | {e["root"] for e in entries})
    print(f"{len(vocab)} strings; comparing {MODELS} (mean-pooled, centered + top-1 PC)\n")
    print(f"{'model':20s}{'pfx_mean':>9s}{'sfx_mean':>9s}{'gap':>8s}"
          f"{'pfx_nov':>9s}{'sfx_nov':>9s}{'ratio':>7s}{'p':>10s}")
    rows = {}
    for m in MODELS:
        raw = embed(vocab, m)
        d = deanisotropize(raw, n_pcs=1)
        scored = core.score(entries, DictKV(d))
        a = asymmetry(scored)
        ratio = a["prefix_nov"] / a["suffix_nov"] if a["suffix_nov"] else float("nan")
        rows[m] = a
        print(f"{m:20s}{a['prefix_mean']:+9.3f}{a['suffix_mean']:+9.3f}{a['gap']:+8.3f}"
              f"{a['prefix_nov']:9.1%}{a['suffix_nov']:9.1%}{ratio:7.2f}{a['p']:10.1e}")
    import json
    json.dump({m: {k: float(v) for k, v in a.items()} for m, a in rows.items()},
              open("stats_bert_control.json", "w"), indent=2)
    print("\nInterpretation: a SMALLER gap/ratio in bert-base-uncased than in gpt2 supports")
    print("the autoregressive-order account; comparable values support a distributional account.")


if __name__ == "__main__":
    main()

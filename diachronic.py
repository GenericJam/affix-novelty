"""
Diachronic test: did prefixed words drift MORE over historical time than
suffixed words? Uses HistWords per-decade SGNS embeddings (fiction subset).

Two measures:
  A. Within-decade compositionality trajectory (no alignment needed): run the
     core metric inside each decade's own space, track prefix vs suffix mean
     compositionality across 1800-1990.
  B. Semantic change magnitude (needs alignment): orthogonal Procrustes between
     an early and a late decade on shared vocab, then 1-cos(early, late) per
     word, compared prefix vs suffix.
"""

import io
import pickle
import zipfile

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import core
import morpholex_entries

ZIP = "histwords/eng-fiction-all_sgns.zip"
DECADES = list(range(1800, 2000, 10))


class KV:
    """gensim-like wrapper over a decade matrix; zero-row words count as absent."""

    def __init__(self, mat, vocab):
        self.mat = mat
        self.idx = {w: i for i, w in enumerate(vocab)}
        self.norm = np.linalg.norm(mat, axis=1)

    def __contains__(self, w):
        i = self.idx.get(w)
        return i is not None and self.norm[i] > 0

    def __getitem__(self, w):
        return self.mat[self.idx[w]]


def _load_vocab(f):
    vocab = pickle.load(f, encoding="latin1")  # HistWords pickles are Python 2
    return [w.decode("utf-8", "replace") if isinstance(w, bytes) else w for w in vocab]


def main():
    zf = zipfile.ZipFile(ZIP)
    names = zf.namelist()
    # detect path prefix inside the archive
    sample = next(n for n in names if n.endswith("-w.npy"))
    prefix = sample[: sample.index("-w.npy") - 4]  # strip 'YYYY-w.npy'
    print(f"archive layout prefix: '{prefix}' (e.g. {sample})")

    def load(year):
        with zf.open(f"{prefix}{year}-w.npy") as f:
            mat = np.load(io.BytesIO(f.read()), allow_pickle=False)
        with zf.open(f"{prefix}{year}-vocab.pkl") as f:
            vocab = _load_vocab(f)
        return KV(mat, vocab)

    entries = morpholex_entries.load()
    print(f"{len(entries)} English entries\n")

    traj = {"year": [], "prefix": [], "suffix": [], "np": [], "ns": []}
    decade_kv = {}
    for y in DECADES:
        try:
            kv = load(y)
        except KeyError:
            continue
        decade_kv[y] = kv
        scored = core.score(entries, kv)
        pref = [e["comp"] for e in scored if e["side"] == "prefix"]
        suff = [e["comp"] for e in scored if e["side"] == "suffix"]
        if len(pref) < 50 or len(suff) < 50:
            print(f"{y}: too few ({len(pref)}/{len(suff)}), skip")
            continue
        traj["year"].append(y)
        traj["prefix"].append(np.mean(pref))
        traj["suffix"].append(np.mean(suff))
        traj["np"].append(len(pref))
        traj["ns"].append(len(suff))
        pm, sm = float(np.mean(pref)), float(np.mean(suff))
        print(
            f"{y}: prefix mean={pm:+.3f} (n={len(pref)})  "
            f"suffix mean={sm:+.3f} (n={len(suff)})  gap={sm - pm:+.3f}"
        )

    # Figure A: compositionality trajectory
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(traj["year"], traj["prefix"], "o-", color="#c0392b", label="prefix words")
    ax.plot(traj["year"], traj["suffix"], "s-", color="#2980b9", label="suffix words")
    ax.set_xlabel("decade")
    ax.set_ylabel("mean compositionality (higher = transparent)")
    ax.set_title("The prefix-suffix compositionality gap is stable across the 20th century")
    ax.legend()
    fig.tight_layout()
    fig.savefig("fig4_diachronic.png", dpi=130)

    # Measure B: aligned semantic-change magnitude, early vs late
    early, late = traj["year"][0], traj["year"][-1]
    ke, kl = decade_kv[early], decade_kv[late]
    shared = [w for w in ke.idx if w in ke and w in kl]
    A = np.array([ke[w] for w in shared])
    B = np.array([kl[w] for w in shared])
    # orthogonal Procrustes: R aligns A -> B
    U, _, Vt = np.linalg.svd(A.T @ B)
    R = U @ Vt

    def change(word):
        if word in ke and word in kl:
            a = ke[word] @ R
            b = kl[word]
            return 1 - core.cos(a, b)
        return None

    scored_late = core.score(entries, kl)
    pc = [change(e["word"]) for e in scored_late if e["side"] == "prefix"]
    sc = [change(e["word"]) for e in scored_late if e["side"] == "suffix"]
    pc = [x for x in pc if x is not None]
    sc = [x for x in sc if x is not None]
    from scipy.stats import mannwhitneyu

    _, p = mannwhitneyu(pc, sc, alternative="greater")
    print(f"\nsemantic change {early}->{late} (1-cos, aligned):")
    print(f"  prefix words: mean change={np.mean(pc):.3f} (n={len(pc)})")
    print(f"  suffix words: mean change={np.mean(sc):.3f} (n={len(sc)})")
    print(f"  prefix > suffix?  Mann-Whitney p={p:.2e}")
    print("\nwrote fig4_diachronic.png")


if __name__ == "__main__":
    main()

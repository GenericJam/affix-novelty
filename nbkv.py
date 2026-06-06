"""gensim-like loader for a per-language Numberbatch .npz."""
import numpy as np


class NBKV:
    def __init__(self, lang):
        d = np.load(f"numberbatch/nb_{lang}.npz", allow_pickle=True)
        self.mat = d["mat"]
        self.idx = {w: i for i, w in enumerate(d["vocab"])}

    def __contains__(self, w):
        return w in self.idx

    def __getitem__(self, w):
        return self.mat[self.idx[w]]

    def __len__(self):
        return len(self.idx)

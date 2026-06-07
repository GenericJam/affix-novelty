"""Reproduce the headline figures and stats from results_morpholex.csv."""

import csv
import json

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

PREFIX_C, SUFFIX_C = "#c0392b", "#2980b9"
T = 0.15  # "novel" threshold

rows = []
with open("results_morpholex.csv") as f:
    for r in csv.DictReader(f):
        r["comp"] = float(r["comp"])
        r["zipf"] = float(r["zipf"])
        r["freq"] = float(r["freq"])
        r["root_free"] = r["root_free"] == "True"
        rows.append(r)
pref = [r for r in rows if r["side"] == "prefix"]
suff = [r for r in rows if r["side"] == "suffix"]


def nov(items, weighted=False):
    c = np.array([r["comp"] for r in items])
    w = np.array([r["freq"] for r in items]) if weighted else np.ones(len(items))
    return float((w[c < T].sum()) / w.sum())


stats = {
    "n_prefix": len(pref),
    "n_suffix": len(suff),
    "type_novelty": {"prefix": nov(pref), "suffix": nov(suff)},
    "token_novelty": {"prefix": nov(pref, True), "suffix": nov(suff, True)},
    "mean_comp": {
        "prefix": float(np.mean([r["comp"] for r in pref])),
        "suffix": float(np.mean([r["comp"] for r in suff])),
    },
    "boundroot_novelty": {
        "prefix": nov([r for r in pref if not r["root_free"]]),
        "suffix": nov([r for r in suff if not r["root_free"]]),
    },
}
with open("stats.json", "w") as f:
    json.dump(stats, f, indent=2)
print(json.dumps(stats, indent=2))

# --- Figure 1: compositionality distributions ------------------------------
fig, ax = plt.subplots(figsize=(8, 4.5))
bins = np.linspace(-0.5, 1.0, 46)
ax.hist(
    [r["comp"] for r in pref],
    bins=bins,
    density=True,
    alpha=0.55,
    color=PREFIX_C,
    label=f"prefix (n={len(pref)})",
)
ax.hist(
    [r["comp"] for r in suff],
    bins=bins,
    density=True,
    alpha=0.55,
    color=SUFFIX_C,
    label=f"suffix (n={len(suff)})",
)
ax.axvline(T, color="k", ls="--", lw=1, label=f"novel threshold ({T})")
ax.set_xlabel("compositionality  (low = meaning NOT predictable from parts = novel)")
ax.set_ylabel("density")
ax.set_title("Prefix-derived words are more often non-compositional than suffix-derived")
ax.legend()
fig.tight_layout()
fig.savefig("fig1_distributions.png", dpi=130)

# --- Figure 2: novelty rate by frequency band ------------------------------
bands = [(1.5, 2.5), (2.5, 3.0), (3.0, 3.5), (3.5, 4.0), (4.0, 7.0)]
labels = ["1.5-2.5", "2.5-3.0", "3.0-3.5", "3.5-4.0", "4.0+"]
pr = [nov([r for r in pref if lo <= r["zipf"] < hi]) for lo, hi in bands]
sr = [nov([r for r in suff if lo <= r["zipf"] < hi]) for lo, hi in bands]
x = np.arange(len(bands))
w = 0.38
fig, ax = plt.subplots(figsize=(8, 4.5))
ax.bar(x - w / 2, pr, w, color=PREFIX_C, label="prefix")
ax.bar(x + w / 2, sr, w, color=SUFFIX_C, label="suffix")
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.set_xlabel("word frequency (Zipf band)")
ax.set_ylabel(f"novelty rate (comp < {T})")
ax.set_title("The prefix advantage grows with frequency (common prefix words lexicalize)")
ax.legend()
fig.tight_layout()
fig.savefig("fig2_frequency.png", dpi=130)

# --- Figure 3: type / token / bound-root summary ---------------------------
cats = ["type", "token\n(usage-weighted)", "bound-root\nonly"]
pv = [
    stats["type_novelty"]["prefix"],
    stats["token_novelty"]["prefix"],
    stats["boundroot_novelty"]["prefix"],
]
sv = [
    stats["type_novelty"]["suffix"],
    stats["token_novelty"]["suffix"],
    stats["boundroot_novelty"]["suffix"],
]
x = np.arange(len(cats))
fig, ax = plt.subplots(figsize=(7.5, 4.5))
b1 = ax.bar(x - w / 2, pv, w, color=PREFIX_C, label="prefix")
b2 = ax.bar(x + w / 2, sv, w, color=SUFFIX_C, label="suffix")
for b in list(b1) + list(b2):
    ax.text(
        b.get_x() + b.get_width() / 2,
        b.get_height() + 0.005,
        f"{b.get_height():.0%}",
        ha="center",
        va="bottom",
        fontsize=9,
    )
ax.set_xticks(x)
ax.set_xticklabels(cats)
ax.set_ylabel(f"novelty rate (comp < {T})")
ax.set_title("Prefixes generate non-compositional words ~1.3-2.7x as often as suffixes")
ax.legend()
fig.tight_layout()
fig.savefig("fig3_summary.png", dpi=130)
print("\nwrote fig1_distributions.png, fig2_frequency.png, fig3_summary.png, stats.json")

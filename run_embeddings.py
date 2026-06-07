"""Run the English analysis across multiple embeddings (robustness check)."""

import json
import sys

import gensim.downloader as api

import core
import morpholex_entries

EMBEDDINGS = sys.argv[1:] or ["glove-wiki-gigaword-300", "word2vec-google-news-300"]

entries = morpholex_entries.load()
print(f"loaded {len(entries)} English entries")

results = {}
for name in EMBEDDINGS:
    print(f"\nloading {name} ...")
    kv = api.load(name)
    scored = core.score(entries, kv)
    s = core.summarize(scored)
    core.print_summary(name, s)
    results[name] = s

with open("stats_embeddings.json", "w") as f:
    json.dump(results, f, indent=2)
print("\nwrote stats_embeddings.json")

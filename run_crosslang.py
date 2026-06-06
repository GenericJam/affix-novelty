"""Cross-linguistic test on Numberbatch word vectors.

Step 0 validates the instrument: run the English MorphoLEX entries through
Numberbatch-en. If the prefix<suffix asymmetry we found with GloVe/word2vec
survives here, Numberbatch is trustworthy for languages where we lack a validated
morphology database. (Numberbatch retrofitting would, if anything, bias toward
transparency, so surviving it is a conservative check.)

Then run Spanish / French / Italian / Portuguese with orthographic decomposition.
"""
import json
import core, morpholex_entries, ortho_entries, affixes
from nbkv import NBKV

results = {}

print("### Step 0: validate Numberbatch on English (MorphoLEX entries) ###")
kv_en = NBKV("en")
print(f"  nb_en vocab: {len(kv_en)}")
en_entries = morpholex_entries.load()
scored = core.score(en_entries, kv_en)
s = core.summarize(scored)
core.print_summary("English / Numberbatch / MorphoLEX entries", s)
results["en_morpholex_numberbatch"] = s

print("\n### Romance languages: orthographic decomposition ###")
for lang in ["es", "fr", "it", "pt"]:
    kv = NBKV(lang)
    entries = ortho_entries.build(lang, kv, affixes.PREFIXES[lang], affixes.SUFFIXES[lang])
    np_ = sum(e["side"] == "prefix" for e in entries)
    ns_ = sum(e["side"] == "suffix" for e in entries)
    print(f"\n[{lang}] vocab={len(kv)}  decomposed: {np_} prefix, {ns_} suffix")
    scored = core.score(entries, kv)
    s = core.summarize(scored)
    core.print_summary(f"{lang} / Numberbatch / orthographic", s)
    # a few example most-novel words each side for sanity
    for side in ("prefix", "suffix"):
        ex = sorted([e for e in scored if e["side"] == side], key=lambda e: e["comp"])[:6]
        print(f"    most-novel {side}: " +
              ", ".join(f"{e['word']}({e['affix']}+{e['root']},{e['comp']:+.2f})" for e in ex))
    results[lang] = s

json.dump(results, open("stats_crosslang.json", "w"), indent=2)
print("\nwrote stats_crosslang.json")

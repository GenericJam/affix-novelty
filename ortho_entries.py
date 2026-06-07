"""Orthographic affix decomposition for any language with a wordfreq list and a
vector vocab. Root must be a real standalone word (in the vocab, above a
frequency floor). Used for the cross-linguistic tests where no validated
morphological database is available.

Limitations (stated honestly): catches only FREE-root derivations (misses
bound-root drift like Spanish recibir/percibir), and orthographic matching
produces some folk-etymology false splits. The offset metric self-corrects the
systematic part, and we compare populations, so this gives a directional answer.
"""

from wordfreq import top_n_list, word_frequency, zipf_frequency


def build(
    lang,
    kv,
    prefixes,
    suffixes,
    n_words=50000,
    word_floor=2.0,
    root_floor=2.8,
    min_root=4,
    min_affix=2,
):
    common = [w for w in top_n_list(lang, n_words) if w.isalpha()]
    valid = {w for w in common if zipf_frequency(w, lang) >= root_floor and w in kv}
    cands = [w for w in common if zipf_frequency(w, lang) >= word_floor and len(w) >= 5 and w in kv]
    pref = sorted({p for p in prefixes if len(p) >= min_affix}, key=len, reverse=True)
    suff = sorted({s for s in suffixes if len(s) >= min_affix}, key=len, reverse=True)

    def root_of(stub):
        return stub if stub in valid else None

    entries = []
    for w in cands:
        for af in pref:  # longest affix first
            if w.startswith(af) and len(w) - len(af) >= min_root:
                r = root_of(w[len(af) :])
                if r and r != w:
                    entries.append(
                        {
                            "word": w,
                            "side": "prefix",
                            "affix": af,
                            "root": r,
                            "freq": word_frequency(w, lang),
                            "zipf": zipf_frequency(w, lang),
                        }
                    )
                    break
        for af in suff:
            if w.endswith(af) and len(w) - len(af) >= min_root:
                r = root_of(w[: -len(af)])
                if r and r != w:
                    entries.append(
                        {
                            "word": w,
                            "side": "suffix",
                            "affix": af,
                            "root": r,
                            "freq": word_frequency(w, lang),
                            "zipf": zipf_frequency(w, lang),
                        }
                    )
                    break
    return entries

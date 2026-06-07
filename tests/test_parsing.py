"""Tests for the morphology parsers / decomposition logic."""

import numpy as np

import morpholex_entries
import ortho_entries


def test_morpholex_parse_prefix_root_suffix():
    assert morpholex_entries._parse("{<re<(ceive)}") == (["re"], ["ceive"], [])
    assert morpholex_entries._parse("{(algorithm)}>ic>") == ([], ["algorithm"], ["ic"])
    assert morpholex_entries._parse("<un<{(limit)}") == (["un"], ["limit"], [])
    p, r, s = morpholex_entries._parse("{<co<(duct)}>ion>")
    assert p == ["co"] and r == ["duct"] and s == ["ion"]


class FakeKV:
    def __init__(self, words):
        self.d = {w: np.ones(3) for w in words}

    def __contains__(self, w):
        return w in self.d

    def __getitem__(self, w):
        return self.d[w]


def test_ortho_build_decomposes_free_root_only():
    # rebuild = re + build, fearless = fear + less (both free roots, >=4 chars)
    kv = FakeKV({"rebuild", "build", "fearless", "fear"})
    entries = ortho_entries.build("en", kv, prefixes=["re"], suffixes=["less"])
    got = {(e["word"], e["side"], e["affix"], e["root"]) for e in entries}
    assert ("rebuild", "prefix", "re", "build") in got
    assert ("fearless", "suffix", "less", "fear") in got


def test_ortho_build_rejects_when_root_not_a_word():
    # "result" should NOT split as re+sult because "sult" is not in the vocab
    kv = FakeKV({"result"})
    entries = ortho_entries.build("en", kv, prefixes=["re"], suffixes=[])
    assert entries == []

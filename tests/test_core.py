"""Tests for the shared compositionality core. Vectors are hand-built so the
expected geometry is known exactly. Offsets need >=2 pairs here (min_pairs=2)."""

import numpy as np
import pytest

import core


class FakeKV:
    def __init__(self, d):
        self.d = {k: np.array(v, dtype=float) for k, v in d.items()}

    def __contains__(self, w):
        return w in self.d

    def __getitem__(self, w):
        return self.d[w]


def entry(word, side, affix, root, freq=1.0, zipf=3.0):
    return {
        "word": word,
        "side": side,
        "affix": affix,
        "root": root,
        "freq": freq,
        "zipf": zipf,
    }


# ----------------------------------------------------------------- cos
def test_cos_identical_orthogonal_opposite():
    a = np.array([1.0, 0.0])
    assert core.cos(a, a) == pytest.approx(1.0)
    assert core.cos(a, np.array([0.0, 1.0])) == pytest.approx(0.0)
    assert core.cos(a, np.array([-1.0, 0.0])) == pytest.approx(-1.0)


# ----------------------------------------------------------------- learn_offsets
def test_learn_offsets_is_mean_shift():
    kv = FakeKV(
        {
            "run": [1, 0, 0, 0],
            "rerun": [1, 1, 0, 0],  # run + [0,1,0,0]
            "do": [0, 0, 1, 0],
            "redo": [0, 1, 1, 0],  # do + [0,1,0,0]
        }
    )
    entries = core.prepare(
        [entry("rerun", "prefix", "re", "run"), entry("redo", "prefix", "re", "do")], kv
    )
    offs = core.learn_offsets(entries, kv, min_pairs=2)
    assert ("prefix", "re") in offs
    np.testing.assert_allclose(offs[("prefix", "re")], [0, 1, 0, 0], atol=1e-9)


def test_learn_offsets_respects_min_pairs():
    kv = FakeKV({"run": [1, 0], "rerun": [1, 1]})
    entries = core.prepare([entry("rerun", "prefix", "re", "run")], kv)
    assert core.learn_offsets(entries, kv, min_pairs=2) == {}


# ----------------------------------------------------------------- prepare
def test_prepare_drops_unembedded_and_tags_root_free():
    kv = FakeKV({"rerun": [1, 1], "run": [1, 0], "rex": [2, 2]})
    out = core.prepare(
        [
            entry("rerun", "prefix", "re", "run"),  # both present -> root_free True
            entry("rex", "prefix", "re", "ceive"),  # word present, root absent -> root_free False
            entry("ghost", "prefix", "re", "run"),  # word absent -> dropped
        ],
        kv,
    )
    by = {e["word"]: e for e in out}
    assert set(by) == {"rerun", "rex"}
    assert by["rerun"]["root_free"] is True
    assert by["rex"]["root_free"] is False


# ----------------------------------------------------------------- score
def _training_kv():
    # affix "re" offset = [0,1,0,0]; three transparent free-root pairs to learn it
    return {
        "run": [1, 0, 0, 0],
        "rerun": [1, 1, 0, 0],
        "do": [0, 0, 1, 0],
        "redo": [0, 1, 1, 0],
        "use": [0, 0, 0, 1],
        "reuse": [0, 1, 0, 1],
    }


def _training_entries():
    return [
        entry("rerun", "prefix", "re", "run"),
        entry("redo", "prefix", "re", "do"),
        entry("reuse", "prefix", "re", "use"),
    ]


def test_score_transparent_is_high_drifted_is_low():
    d = _training_kv()
    d["member"] = [0, 0, 1, 1]
    d["remember"] = [1, 0, 0, 0]  # unrelated to member + offset
    kv = FakeKV(d)
    entries = [*_training_entries(), entry("remember", "prefix", "re", "member")]
    scored = {e["word"]: e for e in core.score(entries, kv, min_pairs=2)}
    assert scored["rerun"]["comp"] == pytest.approx(1.0, abs=1e-6)
    assert scored["rerun"]["method"] == "freeroot"  # root "run" has no siblings
    assert scored["remember"]["comp"] < 0.2


def test_score_family_reconstruction_loo():
    # bound root "x" (absent from kv); coherent family rex/dex sharing root r0
    re_off, de_off, r0 = [0, 1, 0, 0], [0, 0, 1, 0], [1, 0, 0, 0]
    d = _training_kv()
    # add transparent pairs so "de" also gets an offset
    d.update(
        {"part": [0, 0, 0, 1], "depart": [0, 0, 1, 1], "fy": [1, 1, 0, 0], "defy": [1, 1, 1, 0]}
    )
    d["rex"] = list(np.array(r0) + np.array(re_off))  # [1,1,0,0]
    d["dex"] = list(np.array(r0) + np.array(de_off))  # [1,0,1,0]
    kv = FakeKV(d)
    entries = [
        *_training_entries(),
        entry("depart", "prefix", "de", "part"),
        entry("defy", "prefix", "de", "fy"),
        entry("rex", "prefix", "re", "x"),  # x not in kv -> bound root
        entry("dex", "prefix", "de", "x"),
    ]
    scored = {e["word"]: e for e in core.score(entries, kv, min_pairs=2)}
    # rex reconstructed from sibling dex (LOO, not itself) -> coherent -> high comp
    assert scored["rex"]["method"] == "family"
    assert scored["rex"]["comp"] == pytest.approx(1.0, abs=1e-6)


def test_score_skips_bound_singleton_without_offset():
    kv = FakeKV(_training_kv() | {"zorp": [9, 9, 9, 9]})
    # affix "zz" has no training pairs -> no offset -> entry skipped
    entries = [*_training_entries(), entry("zorp", "prefix", "zz", "absent")]
    words = {e["word"] for e in core.score(entries, kv, min_pairs=2)}
    assert "zorp" not in words


# ----------------------------------------------------------------- summarize
def test_summarize_type_and_token_novelty():
    def row(word, side, comp, freq):
        return {
            "word": word,
            "side": side,
            "comp": comp,
            "root_free": True,
            "freq": freq,
            "zipf": 3.0,
        }

    scored = [
        row("a", "prefix", 0.0, 10.0),
        row("b", "prefix", 0.9, 1.0),
        row("c", "suffix", 0.9, 1.0),
        row("d", "suffix", 0.9, 1.0),
    ]
    s = core.summarize(scored)
    assert s["n_prefix"] == 2 and s["n_suffix"] == 2
    # prefix: 1 of 2 below 0.15 -> type 0.5; token weights drift by freq 10/11
    assert s["type_<0.15"]["prefix"] == pytest.approx(0.5)
    assert s["token_<0.15"]["prefix"] == pytest.approx(10 / 11)
    assert s["type_<0.15"]["suffix"] == pytest.approx(0.0)
    assert "mannwhitney_p" in s

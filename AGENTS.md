# AGENTS.md — orientation for working on affix-novelty

A distributional study of when affixed words lose compositional meaning. Python.
Read this before changing anything.

## What this is

We measure the semantic non-compositionality ("drift") of affixed words with word
embeddings, to ask what predicts lexicalization: affix position, grammatical
function, referentiality, frequency. The research log is `notes.md` (chronological,
includes dead ends and reversals); the novelty/prior-work assessment is
`LITERATURE.md`; the paper draft is `PAPER.md`.

## Two-track discipline (important)

Keep the HUMAN track and the LLM track separate, and never merge their evidence.
The LLMs (GPT-2, BERT) are trained on human text, so they are downstream of the
human results: an LLM reproducing a human pattern is inheritance, not independent
corroboration. The LLM track is only interesting where it DIVERGES from the human
distributional baseline (see `bert_control.py`, `nonce_test.py`).

## Repo layout

| File | Role |
|---|---|
| `core.py` | the shared compositionality metric (offsets, family reconstruction, summarize). Everything imports this. |
| `morpholex_entries.py`, `ortho_entries.py` | build decomposed entries (validated English / orthographic any-language) |
| `nbkv.py`, `numberbatch_extract.py` | per-language Numberbatch vectors |
| `run_embeddings.py`, `diachronic.py`, `run_crosslang.py`, `agglutinative.py`, `counterfactual.py`, `arabic_probe.py`, `function_vs_position.py`, `concreteness_test.py`, `llm_rep.py`, `bert_control.py`, `nonce_test.py` | one experiment each |
| `figures.py` | regenerate figures + stats from `results_*.csv` |
| `tests/` | pytest suite for `core.py` and the parsers |

Large data (MorphoLEX, HistWords, Numberbatch, Brysbaert) is gitignored; fetch with
`./fetch_data.sh` then `python numberbatch_extract.py`.

## Pre-commit policy

A committed hook runs ruff + pytest before every commit. Activate it once per clone:

```bash
git config core.hooksPath .githooks
```

The hook (`.githooks/pre-commit`) runs, and you should run before committing:

```bash
.venv/bin/ruff format .          # apply formatting
.venv/bin/ruff check .           # lint (must be clean; --fix for the easy ones)
.venv/bin/python -m pytest       # full suite must pass
```

ruff config is in `pyproject.toml` (line length 100; E/W/F/I/UP/B/SIM/C4/RUF). The
RUF unicode-ambiguity rules are off because the data legitimately contains Turkish
and Arabic script.

## Tests cover everything

Every behavior in `core.py` and the parsers gets a test. The metric is the heart of
the project; a bug there silently corrupts every result. Tests use hand-built
vectors (`FakeKV`) so the expected geometry is exact and they run in ~1s with no
model downloads. When you change the metric or a parser, add or update coverage in
the same commit. Experiment scripts are not unit-tested (they need multi-GB models);
verify them by running and sanity-checking against `notes.md`.

## Conventions (from mob, as they apply)

- Terse. The user reads diffs; don't recap changes in chat.
- Comments explain WHY (invariants, surprising behavior), never WHAT. No narrator
  or step comments. No "this module provides..." docstrings.
- No premature abstraction. Three similar lines beat a half-baked helper.
- Validate at boundaries (data files, CLI args), trust internal callers.
- Don't add scope beyond what was asked.
- Report results honestly: a confounded or null result is logged as such in
  `notes.md`, not dressed up. This project has several (diachronic null,
  agglutinative inconclusive, nonce-test confound) and that is the point.

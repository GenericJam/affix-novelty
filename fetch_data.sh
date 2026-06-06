#!/usr/bin/env bash
# Download the MorphoLEX English morphological database (~7 MB).
# Source: https://github.com/hugomailhot/MorphoLex-en  (check its license).
set -euo pipefail
URL="https://github.com/hugomailhot/MorphoLex-en/raw/master/MorphoLEX_en.xlsx"
curl -fSL -o MorphoLEX_en.xlsx "$URL"
echo "saved MorphoLEX_en.xlsx"

# --- diachronic: HistWords decade embeddings (fiction subset, ~400 MB) ---
mkdir -p histwords
curl -fSL -o histwords/eng-fiction-all_sgns.zip \
  "http://snap.stanford.edu/historical_embeddings/eng-fiction-all_sgns.zip"

# --- cross-linguistic: ConceptNet Numberbatch multilingual word vectors (~3.2 GB) ---
mkdir -p numberbatch
curl -fSL -o numberbatch/numberbatch-19.08.txt.gz \
  "https://conceptnet.s3.amazonaws.com/downloads/2019/numberbatch/numberbatch-19.08.txt.gz"
echo "now run: python numberbatch_extract.py   (writes per-language nb_*.npz)"

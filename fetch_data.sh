#!/usr/bin/env bash
# Download the MorphoLEX English morphological database (~7 MB).
# Source: https://github.com/hugomailhot/MorphoLex-en  (check its license).
set -euo pipefail
URL="https://github.com/hugomailhot/MorphoLex-en/raw/master/MorphoLEX_en.xlsx"
curl -fSL -o MorphoLEX_en.xlsx "$URL"
echo "saved MorphoLEX_en.xlsx"

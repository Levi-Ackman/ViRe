#!/bin/bash
# Render every sample of every benchmark and cache its frozen CLIP embedding under ./emb_VLM/<DATASET>/<split>/.
set -e
for d in APAVA ADFTD TDBRAIN PTB PTB-XL MIMIC; do
    echo "==== $d ===="
    bash scripts/get_vlm_emb/$d.sh
done

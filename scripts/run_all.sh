#!/bin/bash
# Train and evaluate ViRe on all six benchmarks (five seeds each); logs are written to ./logs/<DATASET>/.
set -e
for d in APAVA ADFTD TDBRAIN PTB PTB-XL MIMIC; do
    echo "==== $d ===="
    bash scripts/$d.sh
done

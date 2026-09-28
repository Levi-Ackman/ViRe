#!/usr/bin/env bash
set -euo pipefail

# Rendering + CLIP encoding is embarrassingly parallel: set the visible GPUs and nproc_per_node
# to the number of GPUs you have (e.g. CUDA_VISIBLE_DEVICES=0 and --nproc_per_node=1 on one GPU).
export CUDA_VISIBLE_DEVICES=0,1,2,3
NPROC=${NPROC:-4}

run_dist () {
    python -m torch.distributed.run --standalone --nproc_per_node=$NPROC "$@"
}

run_dist ./Gen_VLM/save_emb_ddp.py \
  --data PTB-XL \
  --root_path ./dataset/PTB-XL/ \
  --divide all \
  --batch_size 128 \
  --num_workers 32 \
  --fs 250 \
  --amp \
  --skip_existing

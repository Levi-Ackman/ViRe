#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES=0,1,2,3

run_dist () {
    python -m torch.distributed.run --standalone --nproc_per_node=4 "$@"
}

run_dist ./Gen_VLM/save_emb_ddp.py \
  --data ADFTD \
  --root_path ./dataset/ADFTD/ \
  --divide all \
  --batch_size 128 \
  --num_workers 32 \
  --fs 256 \
  --amp \
  --skip_existing

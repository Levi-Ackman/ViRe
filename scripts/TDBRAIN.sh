#!/bin/bash
export CUDA_VISIBLE_DEVICES=0

mkdir -p ./logs/TDBRAIN
log_dir="./logs/TDBRAIN"

model_name=ViRe
data_path="./dataset/TDBRAIN/"
data_name="TDBRAIN"

bss=(128)
lrs=(1e-4)
t_layers=(6)
c_layers=(0)

dropouts=(0.)
d_models=(128)
patch_lens=(8)
aug='none,frequency0.4,drop0.6'

for bs in "${bss[@]}"; do
    for lr in "${lrs[@]}"; do
        for t_layer in "${t_layers[@]}"; do
            for c_layer in "${c_layers[@]}"; do
                for dropout in "${dropouts[@]}"; do
                    for d_model in "${d_models[@]}"; do
                        for patch_len in "${patch_lens[@]}"; do
                                python -u run.py \
                                    --root_path $data_path \
                                    --model $model_name \
                                    --data $data_name \
                                    --t_layer $t_layer \
                                    --c_layer $c_layer \
                                    --batch_size $bs \
                                    --d_model $d_model \
                                    --dropout $dropout \
                                    --patch_len $patch_len\
                                    --augmentations $aug \
                                    --seed 56 \
                                    --itr 5\
                                    --lradj constant \
                                    --learning_rate $lr \
                                    --train_epochs 64 \
                                    --patience 64 > "${log_dir}/bs${bs}_lr${lr}_tl${t_layer}_cl${c_layer}_llm${llm_layer}_dp${dropout}_dm${d_model}_pl${patch_len}_aug${aug}.log"
                            
                        done
                    done
                done
            done
        done
    done
done

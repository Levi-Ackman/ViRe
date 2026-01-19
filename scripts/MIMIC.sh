#!/bin/bash
export CUDA_VISIBLE_DEVICES=0

mkdir -p ./logs/MIMIC
log_dir="./logs/MIMIC"

model_name=ViRe
data_path="./dataset/MIMIC/"
data_name="MIMIC"

bss=(128)
lrs=(1e-4)
t_layers=(6)
c_layers=(6)

dropouts=(0.)
d_models=(128)
patch_lens=(6)
aug='none,mask0.2,drop0.3'

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
                                    --lradj constant \
                                    --seed 51\
                                    --itr 5 \
                                    --learning_rate $lr \
                                    --train_epochs 10 \
                                    --patience 10 > "${log_dir}/bs${bs}_lr${lr}_tl${t_layer}_cl${c_layer}_llm${llm_layer}_dp${dropout}_dm${d_model}_pl${patch_len}_aug${aug}.log"
                            
                        done
                    done
                done
            done
        done
    done
done

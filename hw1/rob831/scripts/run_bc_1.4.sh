#!/bin/bash

# Environment and directories
ENV="HalfCheetah-v2"
POLICY_DIR="rob831/policies/experts"
DATA_DIR="rob831/expert_data"

# List of learning rates
learning_rates=("5e-5" "5e-4" "5e-3" "5e-2" "5e-1")

# Loop through each learning rate and run the experiment
for lr in "${learning_rates[@]}"; do
    echo "========================================"
    echo "Running experiment for learning rate: $lr"
    echo "========================================"

    python rob831/scripts/run_hw1.py \
        --expert_policy_file ${POLICY_DIR}/HalfCheetah.pkl \
        --env_name ${ENV} \
        --exp_name bc_HalfCheetah_lr_${lr} \
        --n_iter 1 \
        --expert_data ${DATA_DIR}/expert_data_${ENV}.pkl \
        --video_log_freq -1 \
        --eval_batch_size 5000 \
        -lr ${lr}

    echo "========================================"
    echo "Finished experiment for learning rate: $lr"
    echo "========================================"
    echo ""
done
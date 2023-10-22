# python rob831/scripts/run_hw1.py \
# 	--expert_policy_file rob831/policies/experts/Ant.pkl \
# 	--env_name Ant-v2 --exp_name bc_ant --n_iter 1 \
# 	--expert_data rob831/expert_data/expert_data_Ant-v2.pkl \
# 	--video_log_freq -1 \
# 	--eval_batch_size 5000

#!/bin/bash


# List of environments
environments=("Ant-v2" "HalfCheetah-v2" "Hopper-v2" "Humanoid-v2" "Walker2d-v2")

# Base directory paths
POLICY_DIR="rob831/policies/experts"
DATA_DIR="rob831/expert_data"

# Loop through each environment and run the experiment
for env in "${environments[@]}"; do
    # Extract the name for the policy file
    policy_name=$(echo $env | cut -d'-' -f 1)

    echo "========================================"
    echo "Running experiment for environment: $env"
    echo "========================================"

    python rob831/scripts/run_hw1.py \
        --expert_policy_file ${POLICY_DIR}/${policy_name}.pkl \
        --env_name ${env} \
        --exp_name bc_${policy_name} \
        --n_iter 1 \
        --expert_data ${DATA_DIR}/expert_data_${env}.pkl \
        --video_log_freq -1 \
        --eval_batch_size 5000

    echo "========================================"
    echo "Finished experiment for environment: $env"
    echo "========================================"
    echo ""
done
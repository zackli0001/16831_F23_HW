#!/bin/bash

# Define the lambda values
lambdas=(0 0.95 0.99 1)

# Counter for simultaneous processes
count=0

# Loop over all lambda values
for lambda in "${lambdas[@]}"; do
    # Run the command in the background
    python rob831/scripts/run_hw2.py \
    --env_name Hopper-v4 --ep_len 1000 \
    --discount 0.99 -n 300 -l 2 -s 32 -b 2000 -lr 0.001 \
    --reward_to_go --nn_baseline --action_noise_std 0.5 --gae_lambda $lambda \
    --exp_name q5_b2000_r0.001_lambda$lambda &

    # Increment the counter
    ((count++))

    # If 3 processes are running, wait for them to finish
    if ((count % 3 == 0)); then
        wait
    fi
done

# Wait for any remaining processes to finish
wait
echo "Experiment 5 is done!"
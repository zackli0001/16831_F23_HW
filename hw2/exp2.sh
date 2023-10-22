#!/bin/bash

# Define arrays of values you want to try for b* and r*

# Reach training goal of 1000 within 100 iterations but not stablized
# batch_sizes=(900)
# learning_rates=(0.06)

batch_sizes=(20000)
learning_rates=(0.01)

# Counter for simultaneous processes
count=0

# Loop over all combinations of batch size and learning rate
for b in "${batch_sizes[@]}"; do
    for r in "${learning_rates[@]}"; do
        # Run the command in the background
        python rob831/scripts/run_hw2.py --env_name InvertedPendulum-v4 \
        --ep_len 1000 --discount 0.9 -n 100 -l 2 -s 64 -b $b -lr $r -rtg \
        --exp_name q2_b${b}_r${r} &

        # Increment the counter
        ((count++))

        # If 3 processes are running, wait for them to finish
        if ((count % 3 == 0)); then
            wait
        fi
    done
done

# Wait for any remaining processes to finish
wait
echo "Experiment 2 is done!"
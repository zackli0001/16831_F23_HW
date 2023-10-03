#!/bin/bash

# Experiment 1: q1_sb_no_rtg_dsa
echo "======================"
echo "Running experiment: q1_sb_no_rtg_dsa"
echo "Settings: --env_name CartPole-v0 -n 100 -b 1000 -dsa"
python rob831/scripts/run_hw2.py --env_name CartPole-v0 -n 100 -b 1000 -dsa --exp_name q1_sb_no_rtg_dsa

# Experiment 2: q1_sb_rtg_dsa
echo "======================"
echo "Running experiment: q1_sb_rtg_dsa"
echo "Settings: --env_name CartPole-v0 -n 100 -b 1000 -rtg -dsa"
python rob831/scripts/run_hw2.py --env_name CartPole-v0 -n 100 -b 1000 -rtg -dsa --exp_name q1_sb_rtg_dsa

# Experiment 3: q1_sb_rtg_na
echo "======================"
echo "Running experiment: q1_sb_rtg_na"
echo "Settings: --env_name CartPole-v0 -n 100 -b 1000 -rtg"
python rob831/scripts/run_hw2.py --env_name CartPole-v0 -n 100 -b 1000 -rtg --exp_name q1_sb_rtg_na

# Experiment 4: q1_lb_no_rtg_dsa
echo "======================"
echo "Running experiment: q1_lb_no_rtg_dsa"
echo "Settings: --env_name CartPole-v0 -n 100 -b 5000 -dsa"
python rob831/scripts/run_hw2.py --env_name CartPole-v0 -n 100 -b 5000 -dsa --exp_name q1_lb_no_rtg_dsa

# Experiment 5: q1_lb_rtg_dsa
echo "======================"
echo "Running experiment: q1_lb_rtg_dsa"
echo "Settings: --env_name CartPole-v0 -n 100 -b 5000 -rtg -dsa"
python rob831/scripts/run_hw2.py --env_name CartPole-v0 -n 100 -b 5000 -rtg -dsa --exp_name q1_lb_rtg_dsa

# Experiment 6: q1_lb_rtg_na
echo "======================"
echo "Running experiment: q1_lb_rtg_na"
echo "Settings: --env_name CartPole-v0 -n 100 -b 5000 -rtg"
python rob831/scripts/run_hw2.py --env_name CartPole-v0 -n 100 -b 5000 -rtg --exp_name q1_lb_rtg_na

echo "======================"
echo "All experiments completed!"

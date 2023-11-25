
python cs16831/hw4_part1/scripts/run_hw4_mb.py --exp_name q5_cheetah_random \
--env_name 'cheetah-hw4_part1-v0' --mpc_horizon 15 --add_sl_noise --num_agent_train_steps_per_iter 1500 \
--batch_size_initial 5000 --batch_size 5000 --n_iter 5 --video_log_freq -1 \
--mpc_action_sampling_strategy 'random' &

python cs16831/hw4_part1/scripts/run_hw4_mb.py --exp_name q5_cheetah_cem_2 \
--env_name 'cheetah-hw4_part1-v0' --mpc_horizon 15 --add_sl_noise \
--num_agent_train_steps_per_iter 1500 --batch_size_initial 5000 \
--batch_size 5000 --n_iter 5 --video_log_freq -1 --mpc_action_sampling_strategy 'cem' \
--cem_iterations 2 &

python cs16831/hw4_part1/scripts/run_hw4_mb.py --exp_name q5_cheetah_cem_4 \
--env_name 'cheetah-hw4_part1-v0' --mpc_horizon 15 --add_sl_noise \
--num_agent_train_steps_per_iter 1500 --batch_size_initial 5000 \
--batch_size 5000 --n_iter 5 --video_log_freq -1 --mpc_action_sampling_strategy 'cem' \
--cem_iterations 4
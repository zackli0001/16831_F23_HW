python rob831/scripts/run_hw1.py \
    --expert_policy_file rob831/policies/experts/Ant.pkl \
    --env_name Ant-v2 --exp_name dagger_ant --n_iter 10 \
    --do_dagger --expert_data rob831/expert_data/expert_data_Ant-v2.pkl \
	--video_log_freq -1 \
    --eval_batch_size 5000 

python rob831/scripts/run_hw1.py \
    --expert_policy_file rob831/policies/experts/Humanoid.pkl \
    --env_name Humanoid-v2 --exp_name dagger_Humanoid --n_iter 10 \
    --do_dagger --expert_data rob831/expert_data/expert_data_Humanoid-v2.pkl \
	--video_log_freq -1 \
    --eval_batch_size 5000 
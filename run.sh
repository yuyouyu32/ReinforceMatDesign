#!/bin/bash

cd ./src
nohup python -m train --model ddpg --batch_size 256 --episodes 2500 --save_path ../ckpts/ddpg/ --start_timesteps 512 --log_episodes 10 --use_per --use_trust > ../logs/ddpg_train.log 2>&1 & 
nohup python -m train --model ddpg --batch_size 256 --episodes 2500 --save_path ../ckpts/ddpg_wo_per/ --start_timesteps 512 --log_episodes 10 --use_per> ../logs/ddpg_wo_per_train.log 2>&1 &
nohup python -m train --model ddpg --batch_size 256 --episodes 2500 --save_path ../ckpts/ddpg_ZrCuNiAl/ --start_timesteps 512 --log_episodes 10 --explore_base_index 0 --use_per --use_trust > ../logs/ddpg_ZrCuNiAl_train.log 2>&1 &

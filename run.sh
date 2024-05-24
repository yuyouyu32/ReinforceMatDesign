#!/bin/bash

cd ./src
nohup python -m train --model ddpg --batch_size 256 --episodes 5000 --save_path ../ckpts/ddpg --start_timesteps 1000 --log_episodes 10 --use_per > ../logs/ddpg_train.log 2>&1 && 
nohup python -m train --model ddpg --batch_size 256 --episodes 5000 --save_path ../ckpts/ddpg_wo_per --start_timesteps 1000 --log_episodes 10 > ../logs/ddpg_wo_per_train.log 2>&1 &

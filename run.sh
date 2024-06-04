#!/bin/bash

cd ./src
nohup python -m train --model ddpg --batch_size 256 --episodes 5000 --save_path ../ckpts/ddpg_seed32/ --start_timesteps 1000 --log_episodes 10 --eval_steps 1000 --use_per --use_trust > ../logs/ddpg_train_seed32.log 2>&1 & 

#!/bin/bash

cd ./src
nohup python -m train --model td3 --batch_size 512 --episodes 5000 --save_path ../ckpts/td3_seed21/ --start_timesteps 1000 --log_episodes 10 --eval_steps 512 --use_per --use_trust > ../logs/td3_train_seed21.log 2>&1 & 
nohup python -m train --model ppo --batch_size 64 --episodes 5000 --save_path ../ckpts/ppo_seed21/ --start_timesteps 0 --log_episodes 10 --eval_steps 512 > ../logs/td3_train_seed21.log 2>&1 & 
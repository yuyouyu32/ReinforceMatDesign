import os
import random
from typing import Union

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from config import ExploreBases, MaxStep, N_Action, N_State, Seed, logging

# Set seed for reproducibility, should be set before importing other modules
torch.manual_seed(Seed)
torch.cuda.manual_seed(Seed)
torch.cuda.manual_seed_all(Seed)
np.random.seed(Seed)
random.seed(Seed)

from env.enviroment import Enviroment
from RLs.BaseAgent import BaseAgent
from RLs.PPO import PPOAgent
from RLs.TD3 import TD3Agent
from RLs.utils import moving_average

logger = logging.getLogger(__name__)

class Trainer:
    def __init__(self, agent: Union[BaseAgent, PPOAgent, TD3Agent],
        batch_size: int, episodes: int, save_path: str, start_timesteps: int = 500, log_episodes: int = 10, eval_steps: int = 1000):
        self.agent = agent
        self.env: Enviroment = self.agent.env
        self.batch_size = batch_size
        self.episodes = episodes
        self.start_timesteps = start_timesteps
        self.save_path = save_path
        self.log_episodes = log_episodes
        self.save_episodes = episodes // 5
        self.eval_steps = eval_steps
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)
        self.log_dir = os.path.join(self.save_path, "logs")
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)
        self.writer = SummaryWriter(log_dir=self.log_dir)
        
    def ddpg_train(self, state):
        action = self.agent.select_action(state)
        next_state, reward, done = self.env.step(state, action)
        self.agent.store_experience(state, action, reward, next_state, done)
        
        return next_state, reward, done
    
    def ppo_train(self, state):
        action, action_log_prob = self.agent.select_action(state)
        next_state, reward, done = self.env.step(state, action)

        state_tensor = torch.tensor(state, dtype=torch.float32).to(self.agent.device)
        next_state_tensor = torch.tensor(next_state, dtype=torch.float32).to(self.agent.device)

        state_value = self.agent.critic(state_tensor).item()
        next_state_value = self.agent.critic(next_state_tensor).item()

        transition = (state, action, action_log_prob, reward, done, state_value, next_state_value)
        self.agent.store_experience(transition)
        return next_state, reward, done
        
    def train(self):
        total_steps = 0
        done_record = []
        for episode in tqdm(range(self.episodes)):
            random_base_element = np.random.choice(list(self.env.init_base_matrix.keys()))
            state = self.env.reset_by_constraint(*self.env.init_base_matrix[random_base_element])
            done = False
            episode_step = 0
            episode_rewards = []
            episode_c_loss, episode_a_loss = [], []

            while (not done) and (episode_step <= MaxStep):
                if self.agent.name in {"PPO"}:
                    next_state, reward, done = self.ppo_train(state)
                elif self.agent.name in {"TD3"}:
                    next_state, reward, done = self.ddpg_train(state)
                else:
                    raise ValueError("Unknown agent name")
                state = next_state
                episode_rewards.append(reward)
                total_steps += 1
                episode_step += 1
                if done:
                    if reward < 0:
                        state = self.env.reset_by_constraint(*self.env.init_base_matrix[random_base_element])
                        done = False
                    else:
                        done_record.append(1)

                if self.agent.name in {"TD3"} and total_steps >= self.start_timesteps:
                    c_loss, a_loss = self.agent.train_step(self.batch_size)
                    episode_c_loss.append(c_loss)
                    episode_a_loss.append(a_loss)
                    train_steps = total_steps - self.start_timesteps
                    # if train_steps % self.eval_steps == 0:
                    #     logger.info(f"Train: {train_steps} steps, start eval...")
                    #     eval_reward, eval_conv_reward = self.agent.evaluate_policy(episodes=5)
                    #     self.writer.add_scalar('Reward/Eval', eval_reward, train_steps)
                    #     self.writer.add_scalar('Reward/Eval_con10', eval_conv_reward, train_steps)
                    #     logger.info(f"Eval: {train_steps} steps, Ave Reward: {round(eval_reward, 2)}, Conv Reward: {round(eval_conv_reward, 2)}")
            if not done:
                done_record.append(0)  
            if self.agent.name in {"PPO"}:
                average_c_loss, average_a_loss = self.agent.train_step(self.batch_size)       
            elif self.agent.name in {"TD3"}:
                average_c_loss = np.mean(episode_c_loss)
                average_a_loss = np.mean(episode_a_loss)
                
            if total_steps >= self.start_timesteps:
                self.writer.add_scalar('Loss/Critic', average_c_loss, episode)
                self.writer.add_scalar('Loss/Actor', average_a_loss, episode)
                
            average_reward = np.mean(episode_rewards)
            conv_reward = moving_average(episode_rewards, min(10, episode_step))
    
            # Add to TensorBoard
            self.writer.add_scalar('Reward/Train', average_reward, episode)
            self.writer.add_scalar('Reward/Train_con10', conv_reward[-1], episode)


            if episode % self.log_episodes == 0:
                logger.info(f"Train Episode: {episode + 1}/{self.episodes}, Ave Reward: {round(average_reward, 2)}, Conv Reward: {round(conv_reward[-1], 2)}")
                if total_steps >= self.start_timesteps:
                    logger.info(f"Train Episode: {episode + 1}/{self.episodes} Critic Loss: {average_c_loss}, Actor Loss: {average_a_loss}")

            if episode % self.save_episodes == 0 and total_steps >= self.start_timesteps:
                self.save_by_episode(episode)
                self.env.save_bmgs(os.path.join(self.save_path, "new_BMGs.xlsx"))
        
        self.writer.close()
        self.save_by_episode(episode)
        self.env.save_bmgs(os.path.join(self.save_path, "new_BMGs.xlsx"))
        np.save(os.path.join(self.save_path, "done_record.npy"), np.array(done_record))
        logger.info(f"Finished Train: {episode + 1}")
        
        

    def save_by_episode(self, episode):
        save_path = os.path.join(self.save_path, f"episode_{episode}")
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        self.agent.save_model(self.agent.actor, save_path + f"/{self.agent.name}_actor.pth")
        self.agent.save_model(self.agent.critic, save_path + f"/{self.agent.name}_critic.pth")
    
    def save(self):
        self.agent.save_model(self.agent.actor, self.save_path + f"/{self.agent.name}_actor.pth")
        self.agent.save_model(self.agent.critic, self.save_path + f"/{self.agent.name}_critic.pth")

    def load(self, critic_path, actor_path):
        self.agent.load_model(self.agent.actor, actor_path)
        self.agent.load_model(self.agent.critic, critic_path)
        self.agent.load_model(self.agent.actor_target, actor_path)
        self.agent.load_model(self.agent.critic_target, critic_path)

    def save_logs(self, reward_record, c_loss_record, a_loss_record):
        torch.save(reward_record, self.save_path + "reward_record.pth")
        torch.save(c_loss_record, self.save_path + "c_loss_record.pth")
        torch.save(a_loss_record, self.save_path + "a_loss_record.pth")
        

# Example usage:
def unit_test():
    agent = TD3Agent(N_State, N_Action, use_per=True)
    trainer = Trainer(agent, batch_size=64, episodes=1000, save_path='../ckpts/TD3', start_timesteps= 500, log_episodes= 10)
    trainer.train()

# python -m RLs.Trainer
if __name__ == '__main__':
    unit_test()
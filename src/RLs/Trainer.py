import os
import random
from typing import Optional

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
from RLs.DDPG import DDPGAgent
from RLs.utils import moving_average

logger = logging.getLogger(__name__)

class Trainer:
    def __init__(self, agent: BaseAgent, batch_size: int, episodes: int, save_path: str, start_timesteps: int = 500, log_episodes: int = 10, eval_steps: int = 1000):
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
        
    def reset_state(self, explore_base_index):
        if explore_base_index is None:
            state = self.env.reset()
        else:
            state = self.env.reset_by_constraint(*ExploreBases[explore_base_index])
        return state

    def train(self, explore_base_index: Optional[int] = None):
        total_steps = 0

        for episode in tqdm(range(self.episodes)):
            state = self.reset_state(explore_base_index)
            done = False
            episode_step = 0
            episode_rewards = []
            episode_c_loss, episode_a_loss = 0, 0

            while not done and episode_step <= MaxStep:
                action = self.agent.select_action(state)
                next_state, reward, done = self.env.step(state, action)
                        
                self.agent.store_experience(state, action, reward, next_state, done)

                state = next_state
                episode_rewards.append(reward)
                total_steps += 1
                episode_step += 1

                if total_steps >= self.start_timesteps:
                    c_loss, a_loss = self.agent.train_step(self.batch_size)
                    episode_c_loss += c_loss
                    episode_a_loss += a_loss
                    train_steps = total_steps - self.start_timesteps
                    if train_steps % self.eval_steps == 0:
                        logger.info(f"Train: {train_steps} steps, start eval...")
                        eval_reward, eval_conv_reward = self.agent.evaluate_policy(episodes=5)
                        self.writer.add_scalar('Reward/Eval', eval_reward, train_steps)
                        self.writer.add_scalar('Reward/Eval_con10', eval_conv_reward, train_steps)
                        logger.info(f"Eval: {train_steps} steps, Ave Reward: {round(eval_reward, 2)}, Conv Reward: {round(eval_conv_reward, 2)}")
                    
                if done and reward in {-1, -0.5}:
                    state = self.reset_state(explore_base_index)
                    done = False
                        
            if total_steps >= self.start_timesteps:
                average_c_loss = episode_c_loss / episode_step
                average_a_loss = episode_a_loss / episode_step

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
            
        logger.info(f"Finished Train: {train_steps} steps, start final eval...")
        eval_reward, eval_conv_reward = self.agent.evaluate_policy(episodes=5)
        self.writer.add_scalar('Reward/Eval', eval_reward, train_steps)
        self.writer.add_scalar('Reward/Eval_con10', eval_conv_reward, train_steps)
        logger.info(f"Eval: {train_steps} steps, Ave Reward: {round(eval_reward, 2)}, Conv Reward: {round(eval_conv_reward, 2)}")

        self.writer.close()
        self.env.save_bmgs(os.path.join(self.save_path, "new_BMGs.xlsx"))

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
    agent = DDPGAgent(N_State, N_Action, use_per=True)
    trainer = Trainer(agent, batch_size=64, episodes=1000, save_path='../ckpts/ddpg', start_timesteps= 500, log_episodes= 10)
    trainer.train()

# python -m RLs.Trainer
if __name__ == '__main__':
    unit_test()
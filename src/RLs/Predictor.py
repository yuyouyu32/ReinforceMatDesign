import os
import random
from collections import deque
from typing import Optional, Union

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from config import ExploreBases, MaxStep, N_Action, N_State, Seed, logging
from env.enviroment import Enviroment
from RLs.BaseAgent import BaseAgent
from RLs.PPO import PPOAgent
from RLs.TD3 import TD3Agent
from RLs.utils import moving_average

logger = logging.getLogger(__name__)
torch.manual_seed(Seed)
torch.cuda.manual_seed(Seed)
torch.cuda.manual_seed_all(Seed)
np.random.seed(Seed)
random.seed(Seed)

class Predictor:
    def __init__(self, agent: Union[BaseAgent, PPOAgent, TD3Agent], episodes: int, save_path: str, log_episodes: int = 10):
        self.agent = agent
        self.env: Enviroment = self.agent.env
        self.episodes = episodes
        self.save_path = save_path
        self.log_episodes = log_episodes
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)
        self.log_dir = os.path.join(self.save_path, "logs")
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)
        self.writer = SummaryWriter(log_dir=self.log_dir)
        self.window_size = 10
        self.reward_record = []

    def reset_state(self, explore_base_index):
        if explore_base_index is None:
            random_base_element = np.random.choice(list(self.env.init_base_matrix.keys()))
            state = self.env.reset_by_constraint(*self.env.init_base_matrix[random_base_element])
        else:
            mandatory_elements, optional_elements, k = ExploreBases[explore_base_index]
            state = self.env.reset_by_constraint(mandatory_elements, optional_elements, k, replace_flag=False, min_optional_len=1)
        return state

    def predict(self, explore_base_index: Optional[int] = None):
        total_steps = 0

        for episode in tqdm(range(self.episodes)):
            state = self.reset_state(explore_base_index)
            done = False
            episode_step = 0
            episode_reward = 0

            while not done and episode_step <= MaxStep:
                action = self.agent.select_action(state, explore=False)
                next_state, reward, done = self.env.step(state, action)

                state = next_state
                episode_reward += reward
                total_steps += 1
                episode_step += 1

                if done:
                    if reward < 0:
                        state = self.reset_state(explore_base_index)
                    done = False
                    

            average_reward = episode_reward / episode_step
            self.reward_record.append(average_reward)

            self.writer.add_scalar("Reward", average_reward, episode)
            if episode % self.log_episodes == 0:
                logger.info(f"Episode: {episode + 1}/{self.episodes}, Ave Reward: {round(average_reward, 2)}")

        self.env.save_bmgs(os.path.join(self.save_path, "new_BMGs.xlsx"))
        self.save_logs(self.reward_record)
        self.writer.close()

    def load(self, critic_path, actor_path):
        self.agent.load_model(self.agent.actor, actor_path)
        self.agent.load_model(self.agent.critic, critic_path)
        if hasattr(self.agent, "actor_target"):
            self.agent.load_model(self.agent.actor_target, actor_path)
            self.agent.load_model(self.agent.critic_target, critic_path)

    def save_logs(self, reward_record):
        torch.save(reward_record, self.save_path + "reward_record.pth")


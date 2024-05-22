from abc import ABC, abstractmethod

import numpy as np
import torch
import torch.nn as nn

from config import *
from env.enviroment import Enviroment
from exp.PERBuffer import PrioritizedReplayBuffer
from exp.ReplayBuffer import ReplayBuffer


class BaseAgent(ABC):
    def __init__(self, use_per: bool = False) -> None:
        super(BaseAgent, self).__init__()
        self.use_per = use_per
        if self.use_per:
            self.buffer = PrioritizedReplayBuffer(PoolSize)
        else:
            self.buffer = ReplayBuffer(PoolSize)
        self.env = Enviroment()
        
    @abstractmethod
    def select_action(self, state: np.ndarray, explore: bool = True) -> np.ndarray:
        pass
    
    @abstractmethod
    def train_step(self, batch_size: int) -> float:
        pass
    
    def store_experience(self, s, action, reward, s_, done) -> None:
        """Store experience in the replay buffer."""
        self.buffer.add(s, action, reward, s_, done)
    
    def sample_experiences(self, batch_size: int):
        """Sample a batch of experiences from the replay buffer."""
        return self.buffer.sample(batch_size)
    
    def update_priorities(self, idxes, priorities):
        """Update priorities of experiences in the buffer."""
        if self.use_per:
            self.buffer.update_priorities(idxes, priorities)
    
    def save_model(self, model: nn.Module, path: str) -> None:
        """Save the model parameters."""
        torch.save(model.state_dict(), path)
    
    def load_model(self, model: nn.Module, path: str) -> None:
        """Load the model parameters."""
        model.load_state_dict(torch.load(path))
    
    def evaluate_policy(self, episodes: int = 10) -> float:
        """Evaluate the current policy."""
        total_reward = 0.0
        for _ in range(episodes):
            state = self.env.reset()
            done = False
            while not done:
                action = self.select_action(state, explore=False)
                next_state, reward, done = self.env.step(state, action)
                total_reward += reward
                state = next_state
        return total_reward / episodes
        

    
def unit_test():
    agent = BaseAgent()

# python -m RLs.BaseAgent
if __name__ == '__main__':
    unit_test()
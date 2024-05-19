from abc import ABC, abstractmethod

import pandas as pd

from config import *
from exp.PERBuffer import PrioritizedReplayBuffer
from exp.ReplayBuffer import ReplayBuffer
from env.enviroment import Enviroment



class BaseAgent(ABC):
    def __init__(self, use_per: bool = False) -> None:
        super(BaseAgent, self).__init__()
        self.use_per = use_per
        if self.use_per:
            self.buffer = PrioritizedReplayBuffer(PoolSize)
        else:
            self.buffer = ReplayBuffer(PoolSize)
        self.env = Enviroment()


    
def unit_test():
    agent = BaseAgent()

# python -m RLs.BaseAgent
if __name__ == '__main__':
    unit_test()
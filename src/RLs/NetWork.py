import torch
import torch.nn as nn
from torch.distributions.normal import Normal
import numpy as np
from RLs.utils import activate_A_func

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class DeterActorNet(nn.Module):
    def __init__(self, s_shape, a_shape):
        super(DeterActorNet, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(s_shape, 256),
            nn.BatchNorm1d(256),  
            nn.LeakyReLU(),
            nn.Linear(256, 256),
            nn.BatchNorm1d(256),  
            nn.LeakyReLU(),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),  
            nn.LeakyReLU(),
            nn.Linear(128, a_shape),
            nn.Tanh()
        )
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x, k):
        x = self.fc(x)
        a = activate_A_func(x, k)
        return a


class DoubleQNet(nn.Module):
    def __init__(self, s_shape, a_shape):
        super(DoubleQNet, self).__init__()
        self.fc_s = nn.Sequential(
            nn.Linear(s_shape, 256),
            nn.BatchNorm1d(256),  
            nn.LeakyReLU(),
            nn.Linear(256, 512),
            nn.BatchNorm1d(512),  
            nn.LeakyReLU(),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),  
            nn.LeakyReLU(),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),  
            nn.LeakyReLU()
        )
        self.fc_a = nn.Sequential(
            nn.Linear(a_shape, 256),
            nn.BatchNorm1d(256),  
            nn.LeakyReLU(),
            nn.Linear(256, 512),
            nn.BatchNorm1d(512),  
            nn.LeakyReLU(),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),  
            nn.LeakyReLU(),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),  
            nn.LeakyReLU()
        )
        self.output1 = nn.Sequential(
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),  
            nn.LeakyReLU(),
            nn.Linear(128, 32),
            nn.BatchNorm1d(32),  
            nn.LeakyReLU(),
            nn.Linear(32, 1)
        )
        self.output2 = nn.Sequential(
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),  
            nn.LeakyReLU(),
            nn.Linear(128, 32),
            nn.BatchNorm1d(32),  
            nn.LeakyReLU(),
            nn.Linear(32, 1)
        )
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x, a):
        h1 = self.fc_s(x)
        h2 = self.fc_a(a)
        cat = torch.cat([h1, h2], dim=1)
        q1 = self.output1(cat)
        q2 = self.output2(cat)
        return q1, q2
    
class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(QNetwork, self).__init__()
        self.fc_s = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.LeakyReLU(),
            nn.Linear(256, 512),
            nn.LeakyReLU(),
            nn.Linear(512, 256),
            nn.LeakyReLU()
        )
        self.fc_a = nn.Sequential(
            nn.Linear(action_dim, 256),
            nn.LeakyReLU(),
            nn.Linear(256, 512),
            nn.LeakyReLU(),
            nn.Linear(512, 256),
            nn.LeakyReLU()
        )
        self.fc_cat = nn.Sequential(
            nn.Linear(512, 256),
            nn.LeakyReLU(),
            nn.Linear(256, 1)
        )
        self._initialize_weights()

    def forward(self, state, action):
        state_out = self.fc_s(state)
        action_out = self.fc_a(action)
        cat = torch.cat([state_out, action_out], dim=1)
        q_value = self.fc_cat(cat)
        return q_value
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
class PolicyActorNet(nn.Module):
    def __init__(self, s_shape, a_shape, action_scale):
        super(PolicyActorNet, self).__init__()
        self.shared = nn.Sequential(
            nn.Linear(s_shape, 128),
            nn.LeakyReLU(),
            nn.Linear(128, 256),
            nn.LeakyReLU(),
            nn.Linear(256, 128),
            nn.LeakyReLU(),
            nn.Linear(128, 64),
            nn.LeakyReLU()
        )
        self.fc_mean = nn.Linear(64, a_shape)
        self.fc_std = nn.Linear(64, a_shape)
        self.action_scale = action_scale

    def forward(self, x, k):
        shared = self.shared(x)
        mean = self.fc_mean(shared)
        std = torch.exp(self.fc_std(shared))
        normal_dist = Normal(mean, std)
        action = normal_dist.sample()
        activated_action = activate_A_func(action, k)
        log_prob = normal_dist.log_prob(action).sum(-1)  # Adjust if necessary based on activation
        return activated_action, log_prob
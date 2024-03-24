import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class ActorNet(nn.Module):
    def __init__(self, s_shape, a_shape, action_scale):
        super(ActorNet, self).__init__()
        self.fc1 = nn.Linear(s_shape, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc_mu = nn.Linear(64, a_shape)
        self.action_scale = action_scale

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        mu = torch.tanh(self.fc_mu(x)) * torch.tensor(np.array(self.action_scale)).float().to(device)
        return mu

class QNet(nn.Module):
    def __init__(self, s_shape, a_shape):
        super(QNet, self).__init__()
        self.fc_s = nn.Linear(s_shape, 64)
        self.fc_a = nn.Linear(a_shape, 64)
        self.fc_q = nn.Linear(128, 32)
        self.fc_out = nn.Linear(32, 1)

    def forward(self, x, a):
        h1 = F.relu(self.fc_s(x))
        h2 = F.relu(self.fc_a(a))
        cat = torch.cat([h1, h2], dim=1)
        q = F.relu(self.fc_q(cat))
        q = self.fc_out(q)
        return q
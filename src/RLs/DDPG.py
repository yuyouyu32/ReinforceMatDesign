import torch
import torch.nn as nn
import torch.optim as optim

from config import A_Scale
from RLs.BaseAgent import BaseAgent
from RLs.NetWork import DeterActorNet, DoubleQNet, device
import numpy as np


class DDPGAgent(BaseAgent):
    def __init__(self, state_dim, action_dim, use_per: bool = False):
        super(DDPGAgent, self).__init__(use_per)
        self.name = 'ddpg'
        self.actor = DeterActorNet(state_dim, action_dim, A_Scale).to(device)
        self.actor_target = DeterActorNet(state_dim, action_dim, A_Scale).to(device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=1e-4)

        self.critic = DoubleQNet(state_dim, action_dim).to(device)
        self.critic_target = DoubleQNet(state_dim, action_dim).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=1e-3)
        self.device = device
        self.max_action = A_Scale
        self.discount = 0.99
        self.tau = 0.005
        self.epsilon = 1.0
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.1

    def select_action(self, state: np.ndarray, explore: bool=True):
        if explore and np.random.rand() < self.epsilon:
            action = self.env.get_random_legal_action(state)
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        else:
            state = torch.FloatTensor(state.reshape(1, -1)).to(self.device)
            k = torch.count_nonzero(state, dim=1).item()  # Count non-zero elements in the state tensor
            k_tensor = torch.tensor([k], dtype=torch.int).to(self.device)  # Convert k to a tensor and move to device
            action = self.actor(state, k_tensor).cpu().data.numpy().flatten()

        return action


    def train_step(self, batch_size):
        if not self.use_per:
            states, actions, rewards, next_states, dones = self.sample_experiences(batch_size)
            weights, batch_idxes = np.ones_like(rewards), None
        else:
            states, actions, rewards, next_states, dones, weights, batch_idxes = self.sample_experiences(batch_size)
            weights = np.sqrt(weights)
            weights = torch.FloatTensor(weights).to(device)
        states = torch.FloatTensor(states).to(device)
        actions = torch.FloatTensor(actions).to(device)
        rewards = torch.FloatTensor(rewards).to(device)
        next_states = torch.FloatTensor(next_states).to(device)
        dones = torch.FloatTensor(dones).to(device)
        with torch.no_grad():
            k = torch.count_nonzero(next_states, dim=1)  # Count non-zero elements in the state tensor
            next_actions = self.actor_target(next_states, k)
            target_q1, target_q2 = self.critic_target(next_states, next_actions)
            target_q = torch.min(target_q1, target_q2)
            rewards = rewards.view(-1, 1)
            dones = dones.view(-1, 1)
            target_q = rewards + (1 - dones) * self.discount * target_q

        current_q1, current_q2 = self.critic(states, actions)
        # critic_loss = nn.MSELoss()(current_q1, target_q) + nn.MSELoss()(current_q2, target_q)
        TD_Error_q1, TD_Error_q2 = torch.abs(current_q1 - target_q), torch.abs(current_q2 - target_q)
        weighted_TD_errors_1, weighted_TD_errors_2 = weights * TD_Error_q1, weights * TD_Error_q2
        zero_tensor = torch.zeros(weighted_TD_errors_1.shape).to(device)
        critic_loss = nn.functional.smooth_l1_loss(weighted_TD_errors_1, zero_tensor) + nn.functional.smooth_l1_loss(weighted_TD_errors_2, zero_tensor)
        
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Update the actor network using the deterministic policy gradient
        k = torch.count_nonzero(states, dim=1)  # Count non-zero elements in the state tensor
        actor_loss = -self.critic(states, self.actor(states, k))[0].mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        
        if self.use_per:
            new_priorities = (TD_Error_q1 + TD_Error_q2).cpu().data.numpy()
            self.buffer.update_priorities(batch_idxes, new_priorities)

        # Update the target networks
        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        return critic_loss.item(), actor_loss.item()
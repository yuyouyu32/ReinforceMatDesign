import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal

from RLs.BaseAgent import BaseAgent
from RLs.NetWork import PPOActorNetwork, PPOCriticNetwork, device
from RLs.utils import activate_A_func


class PPOAgent(BaseAgent):
    def __init__(self, state_dim, action_dim, lr=3e-4, gamma=0.99, clip_eps=0.2, update_epochs=10, batch_size=64):
        super(PPOAgent, self).__init__(use_per=False, use_trust=False)
        self.name = 'PPO'
        self.actor = PPOActorNetwork(state_dim, action_dim).to(device)
        self.critic = PPOCriticNetwork(state_dim).to(device)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr)
        self.gamma = gamma
        self.clip_eps = clip_eps
        self.update_epochs = update_epochs
        self.batch_size = batch_size
        self.mse_loss = nn.MSELoss()
        self.memory = []

    def select_action(self, state):
        state = state / sum(state)
        state = torch.FloatTensor(state.reshape(1, -1)).to(self.device)
        k = torch.count_nonzero(state, dim=1).item()  # Count non-zero elements in the state tensor
        k_tensor = torch.tensor([k], dtype=torch.int).to(self.device)  # Convert k to a tensor and move to device
        
        with torch.no_grad():
            action_mean, action_std = self.actor(state)
        dist = Normal(action_mean, action_std)
        action = dist.sample()
        action_log_prob = dist.log_prob(action)
        action = torch.clamp(action, -1, 1)
        action = activate_A_func(state, k_tensor)
        return action.cpu().data.numpy().flatten(), action_log_prob.cpu().data.numpy().flatten()

    def store_experience(self, transition):
        self.memory.append(transition)

    def compute_advantages(self, rewards, values, next_values, dones):
        advantages = []
        advantage = 0
        for r, v, nv, d in zip(rewards[::-1], values[::-1], next_values[::-1], dones[::-1]):
            td_error = r + self.gamma * nv * (1 - d) - v
            advantage = td_error + self.gamma * advantage * (1 - d)
            advantages.insert(0, advantage)
        return advantages

    def train_step(self):
        states, actions, log_probs_old, rewards, dones, values = zip(*self.memory)

        states = torch.tensor(states, dtype=torch.float32)
        actions = torch.tensor(actions, dtype=torch.float32)
        log_probs_old = torch.tensor(log_probs_old, dtype=torch.float32)
        rewards = torch.tensor(rewards, dtype=torch.float32)
        dones = torch.tensor(dones, dtype=torch.float32)
        values = torch.tensor(values, dtype=torch.float32)

        next_values = values[1:].tolist() + [0]
        advantages = self.compute_advantages(rewards.tolist(), values[:-1].tolist(), next_values, dones.tolist())
        advantages = torch.tensor(advantages, dtype=torch.float32)
        returns = advantages + values[:-1]

        dataset_size = len(states)
        indices = np.arange(dataset_size)

        for _ in range(self.update_epochs):
            np.random.shuffle(indices)
            for start in range(0, dataset_size, self.batch_size):
                end = start + self.batch_size
                batch_indices = indices[start:end]
                
                # Process the last batch
                if len(batch_indices) < self.batch_size:
                    batch_indices = indices[start:dataset_size]

                batch_states = states[batch_indices]
                batch_actions = actions[batch_indices]
                batch_log_probs_old = log_probs_old[batch_indices]
                batch_returns = returns[batch_indices]
                batch_advantages = advantages[batch_indices]

                # Update Actor Network
                action_mean, action_std = self.actor(batch_states)
                dist = Normal(action_mean, action_std)
                log_probs = dist.log_prob(batch_actions)
                ratio = torch.exp(log_probs - batch_log_probs_old)

                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps) * batch_advantages
                policy_loss = -torch.min(surr1, surr2).mean()

                self.actor_optimizer.zero_grad()
                policy_loss.backward()
                self.actor_optimizer.step()

                # Update Critic Network
                state_values = self.critic(batch_states)
                value_loss = self.mse_loss(state_values, batch_returns.unsqueeze(1))

                self.critic_optimizer.zero_grad()
                value_loss.backward()
                self.critic_optimizer.step()

        self.memory = []


    def train(self, env, max_episodes):
        for episode in range(max_episodes):
            state = env.reset()
            done = False
            while not done:
                action, action_log_prob = self.select_action(state)
                next_state, reward, done, _ = env.step(action)

                state_tensor = torch.tensor(state, dtype=torch.float32)
                next_state_tensor = torch.tensor(next_state, dtype=torch.float32)
                reward_tensor = torch.tensor([reward], dtype=torch.float32)
                done_tensor = torch.tensor([done], dtype=torch.float32)

                state_value = self.critic(state_tensor).item()
                next_state_value = self.critic(next_state_tensor).item()

                transition = (state, action, action_log_prob, reward, done, state_value)
                self.store_experience(transition)

                state = next_state

            self.update()

def unit_test():
    from config import N_Action, N_State
    agent = PPOAgent(N_State, N_Action)
    s = agent.env.reset()
    for i in range(100):
        a, prob = agent.select_action(s)
        print(a, sum(a))
        print(prob)
        input('wait')

# python -m RLs.PPO
if __name__ == '__main__':
    unit_test()
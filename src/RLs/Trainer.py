import os
from tqdm import tqdm
import torch
from torch.utils.tensorboard import SummaryWriter
from BMGs import BMGs
from config import MaxStep, N_Action, N_State, logging, ExploreBases
from env.enviroment import Enviroment
from RLs.BaseAgent import BaseAgent
from RLs.DDPG import DDPGAgent

logger = logging.getLogger(__name__)

class Trainer:
    def __init__(self, agent: BaseAgent, batch_size: int, episodes: int, save_path: str, start_timesteps: int = 500, log_episodes: int = 10):
        self.agent = agent
        self.env: Enviroment = self.agent.env
        self.batch_size = batch_size
        self.episodes = episodes
        self.start_timesteps = start_timesteps
        self.save_path = save_path
        self.log_episodes = log_episodes
        self.save_episodes = episodes // 5
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)
        self.log_dir = os.path.join(self.save_path, "logs")
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)
        self.writer = SummaryWriter(log_dir=self.log_dir)

    def train(self):
        total_steps = 0
        reward_record = []
        c_loss_record = []
        a_loss_record = []
        for episode in tqdm(range(self.episodes)):
            # state = self.env.reset_by_constraint(**ExploreBases[0])
            state = self.env.reset()
            done = False
            episode_step = 0
            episode_reward = 0
            episode_c_loss, episode_a_loss = 0, 0
            while not done and episode_step <= MaxStep:

                action = self.agent.select_action(state)
                next_state, reward, done = self.env.step(state, action)
                if done and reward != -10:
                    logger.info(f"Find New BMGs: {BMGs(next_state).bmg_s}")
                    self.env.save_bmgs(os.path.join(self.save_path, "new_BMGs.xlsx"))
                self.agent.store_experience(state, action, reward, next_state, done)

                state = next_state
                episode_reward += reward
                total_steps += 1
                episode_step += 1

                if total_steps >= self.start_timesteps:
                    c_loss, a_loss = self.agent.train_step(self.batch_size)
                    episode_c_loss += c_loss
                    episode_a_loss += a_loss
            if total_steps >= self.start_timesteps:
                average_reward = episode_reward / episode_step
                average_c_loss = episode_c_loss / episode_step
                average_a_loss = episode_a_loss / episode_step
                reward_record.append(average_reward)
                c_loss_record.append(average_c_loss)
                a_loss_record.append(average_a_loss)
                # Add to TensorBoard
                self.writer.add_scalar('Reward/Average', average_reward, episode)
                self.writer.add_scalar('Loss/Critic', average_c_loss, episode)
                self.writer.add_scalar('Loss/Actor', average_a_loss, episode)
            if episode % self.log_episodes == 0 and episode != 0 and total_steps >= self.start_timesteps:
                logger.info(f"Episode: {episode + 1}/{self.episodes}, Ave Reward: {round(average_reward, 2)}")
                logger.info(f"Critic Loss: {average_c_loss}, Actor Loss: {average_a_loss}")
            if episode % self.save_episodes == 0 and episode != 0 and total_steps >= self.start_timesteps:
                self.save_by_episode(episode)

        self.save_logs(reward_record, c_loss_record, a_loss_record)
        self.writer.close()

    def save_by_episode(self, episode):
        self.agent.save_model(self.agent.actor, self.save_path + f"{self.agent.name}_actor.{episode}.pth")
        self.agent.save_model(self.agent.critic, self.save_path + f"{self.agent.name}_critic.{episode}.pth")
    
    def save(self):
        self.agent.save_model(self.agent.actor, self.save_path + f"{self.agent.name}_actor.pth")
        self.agent.save_model(self.agent.critic, self.save_path + f"{self.agent.name}_critic.pth")

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
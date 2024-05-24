import click
from RLs.DDPG import DDPGAgent
from RLs.Trainer import Trainer
from config import logging, N_Action, N_State

logger = logging.getLogger(__name__)

@click.command()
@click.option('--model', default='ddpg', help='RL model to train.')
@click.option('--use_per', is_flag=True, default=False, help='Use Prioritized Experience Replay.')
@click.option('--batch_size', default=64, help='Batch size for training.')
@click.option('--episodes', default=1000, help='Number of episodes to train.')
@click.option('--save_path', default='../ckpts/ddpg', help='Path to save the trained model.')
@click.option('--start_timesteps', default=500, help='Timesteps to start training.')
@click.option('--log_episodes', default=10, help='Log every n episodes.')

def train(model, use_per, batch_size, episodes, save_path, start_timesteps, log_episodes):
    """Train a RLs agent with given parameters."""
    if model == 'ddpg':
        agent = DDPGAgent(N_State, N_Action, use_per=use_per)
    trainer = Trainer(agent, batch_size=batch_size, episodes=episodes, save_path=save_path, start_timesteps=start_timesteps, log_episodes=log_episodes)
    trainer.train()


# nohup python -m train --model ddpg --batch_size 256 --episodes 5000 --save_path ../ckpts/ddpg --start_timesteps 1000 --log_episodes 10 --use_per > ../logs/ddpg_train.log 2>&1 &
# nohup python -m train --model ddpg --batch_size 256 --episodes 5000 --save_path ../ckpts/ddpg_wo_per --start_timesteps 1000 --log_episodes 10 > ../logs/ddpg_wo_per_train.log 2>&1 &
if __name__ == '__main__':
    train()

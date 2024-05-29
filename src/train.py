import click
from RLs.DDPG import DDPGAgent
from RLs.Trainer import Trainer
from config import logging, N_Action, N_State

logger = logging.getLogger(__name__)

@click.command()
@click.option('--model', default='ddpg', help='RL model to train.')
@click.option('--use_per', is_flag=True, default=False, help='Use Prioritized Experience Replay.')
@click.option('--use_trust', is_flag=True, default=False, help='Use Trusted Experience Replay.')
@click.option('--batch_size', default=256, help='Batch size for training.')
@click.option('--episodes', default=5000, help='Number of episodes to train.')
@click.option('--save_path', default='../ckpts/ddpg', help='Path to save the trained model.')
@click.option('--start_timesteps', default=512, help='Timesteps to start training.')
@click.option('--log_episodes', default=10, help='Log every n episodes.')
@click.option('--explore_base_index', default=None, help='Index of the base to explore.')

def train(model, use_per, use_trust, batch_size, episodes, save_path, start_timesteps, log_episodes, explore_base_index):
    """Train a RLs agent with given parameters."""
    if model == 'ddpg':
        agent = DDPGAgent(N_State, N_Action, use_per=use_per, use_trust=use_trust)
    trainer = Trainer(agent, batch_size=batch_size, episodes=episodes, save_path=save_path, start_timesteps=start_timesteps, log_episodes=log_episodes)
    if explore_base_index:
        explore_base_index = int(explore_base_index)
        logger.info(f"Start training {model} agent with {episodes} episodes, env reset by explore base index: {explore_base_index}.")
    else:
        explore_base_index = None
        logger.info(f"Start training {model} agent with {episodes} episodes and random env reset method.")
    trainer.train(explore_base_index=explore_base_index)

if __name__ == '__main__':
    train()

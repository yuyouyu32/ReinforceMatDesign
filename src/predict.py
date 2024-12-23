import os

import click

from config import N_Action, N_State, logging
from RLs.DDPG import DDPGAgent
from RLs.DQN import DQNAgent
from RLs.PPO import PPOAgent
from RLs.Predictor import Predictor
from RLs.SAC import SACAgent
from RLs.TD3 import TD3Agent

logger = logging.getLogger(__name__)

@click.command()
@click.option('--model', default='td3', help='RL model agent.')
@click.option('--c_pth', default='../ckpts/td3/', help='Critic model to predict.')
@click.option('--a_pth', default='../ckpts/td3/', help='Actor model to predict.')
@click.option('--episodes', default=1500, help='Number of episodes to predict.')
@click.option('--save_path', default='../designs/td3', help='Path to save the designed results.')
@click.option('--log_episodes', default=10, help='Log every n episodes.')
@click.option('--explore_base_index', default=None, help='Index of the base to explore.')

def predict(model, c_pth, a_pth, episodes, save_path, log_episodes, explore_base_index):
    """Train a RLs agent with given parameters."""
    if model == 'td3':
        agent = TD3Agent(N_State, N_Action)
    elif model == 'ppo':
        agent = PPOAgent(N_State, N_Action)
    elif model == 'dqn':
        agent = DQNAgent(N_State, N_Action)
    elif model == 'ddpg':
        agent = DDPGAgent(N_State, N_Action)
    elif model == 'sac':
        agent = SACAgent(N_State, N_Action)
    else:
        raise ValueError(f"Model {model} not supported.")
    predictor = Predictor(agent=agent, episodes=episodes, save_path=save_path, log_episodes=log_episodes)
    predictor.load(c_pth, a_pth)
    predictor.agent.epsilon = predictor.agent.epsilon_min
    if explore_base_index is not None:
        if explore_base_index.isdigit():
            explore_base_index = int(explore_base_index)
            logger.info(f"Start predicting {model} agent with {episodes} episodes, env reset by explore base index: {explore_base_index}.")
            predictor.predict(explore_base_index=explore_base_index)
        elif explore_base_index.lower() == 'all':
            logger.info(f"Start predicting {model} agent with {episodes} episodes, env reset by all explore bases.")
            for base_matrix in list(predictor.env.init_base_matrix.keys()):
                logger.info(f"Start predicting {model} agent with {episodes} episodes, env reset by explore base: {base_matrix}.")
                predictor.save_path = os.path.join(save_path, base_matrix)
                if not os.path.exists(predictor.save_path):
                    os.makedirs(predictor.save_path)
                predictor.predict(explore_base_index=base_matrix)
                done_ratio = len(predictor.env.new_bmgs) / episodes
                logger.info(f"Base Matrix: {base_matrix}, Done Ratio: {done_ratio}")
                predictor.env.new_bmgs = []
    else:
        explore_base_index = None
        logger.info(f"Start predicting {model} agent with {episodes} episodes and random env reset method.")
        predictor.predict(explore_base_index=explore_base_index)

# nohup python -m predict --model td3 --c_pth "../ckpts/td3_seed32/episode_100072/TD3_critic.pth" --a_pth "../ckpts/td3_seed32/episode_100072/TD3_actor.pth" --episodes 1500 --save_path ../designs/td3_1500 --log_episodes 10 --explore_base_index 0 > ../logs/td3_1500_predict.log 2>&1 &
# nohup python -m predict --model td3 --c_pth "../ckpts/td3_seed32/episode_100072/TD3_critic.pth" --a_pth "../ckpts/td3_seed32/episode_100072/TD3_actor.pth" --episodes 100 --save_path ../designs/td3_all_base --log_episodes 1000 --explore_base_index all > ../logs/td3_all_base_predict.log 2>&1 &
if __name__ == '__main__':
    predict()

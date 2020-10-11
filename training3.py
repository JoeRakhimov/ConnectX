import gym
from kaggle_environments import make, evaluate

import os
import numpy as np
import torch as th
from torch import nn as nn
import torch.nn.functional as F
from numpy.random import choice

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import load_results
from stable_baselines3.common.torch_layers import NatureCNN
from stable_baselines3.common.policies import ActorCriticPolicy, ActorCriticCnnPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

from common import get_win_percentages_and_score
from connect4gym3 import board_flip, SaveBestModelCallback


def minimax_agent(obs, config):
    ################################
    # Imports and helper functions #
    ################################

    import numpy as np
    import random

    # Calculates score if agent drops piece in selected column
    def score_move(grid, col, mark, config):
        next_grid = drop_piece(grid, col, mark, config)
        score = get_heuristic(next_grid, mark, config)
        return score

    # Helper function for score_move: gets board at next step if agent drops piece in selected column
    def drop_piece(grid, col, mark, config):
        next_grid = grid.copy()
        for row in range(config.rows - 1, -1, -1):
            if next_grid[row][col] == 0:
                break
        next_grid[row][col] = mark
        return next_grid

    # Helper function for score_move: calculates value of heuristic for grid
    def get_heuristic(grid, mark, config):
        num_twos = count_windows(grid, 2, mark, config)
        num_threes = count_windows(grid, 3, mark, config)
        num_fours = count_windows(grid, 4, mark, config)
        num_twos_opp = count_windows(grid, 2, mark % 2 + 1, config)
        num_threes_opp = count_windows(grid, 3, mark % 2 + 1, config)
        score = num_fours * 10000 + num_threes * 10 + num_twos - num_twos_opp * 100 - num_threes_opp * 1000
        return score

    # Helper function for get_heuristic: checks if window satisfies heuristic conditions
    def check_window(window, num_discs, piece, config):
        return (window.count(piece) == num_discs and window.count(0) == config.inarow - num_discs)

    # Helper function for get_heuristic: counts number of windows satisfying specified heuristic conditions
    def count_windows(grid, num_discs, piece, config):
        num_windows = 0
        # horizontal
        for row in range(config.rows):
            for col in range(config.columns - (config.inarow - 1)):
                window = list(grid[row, col:col + config.inarow])
                if check_window(window, num_discs, piece, config):
                    num_windows += 1
        # vertical
        for row in range(config.rows - (config.inarow - 1)):
            for col in range(config.columns):
                window = list(grid[row:row + config.inarow, col])
                if check_window(window, num_discs, piece, config):
                    num_windows += 1
        # positive diagonal
        for row in range(config.rows - (config.inarow - 1)):
            for col in range(config.columns - (config.inarow - 1)):
                window = list(grid[range(row, row + config.inarow), range(col, col + config.inarow)])
                if check_window(window, num_discs, piece, config):
                    num_windows += 1
        # negative diagonal
        for row in range(config.inarow - 1, config.rows):
            for col in range(config.columns - (config.inarow - 1)):
                window = list(grid[range(row, row - config.inarow, -1), range(col, col + config.inarow)])
                if check_window(window, num_discs, piece, config):
                    num_windows += 1
        return num_windows

    #########################
    # Agent makes selection #
    #########################

    # Get list of valid moves
    valid_moves = [c for c in range(config.columns) if obs.board[c] == 0]
    # Convert the board to a 2D grid
    grid = np.asarray(obs.board).reshape(config.rows, config.columns)
    # Use the heuristic to assign a score to each possible board in the next turn
    scores = dict(zip(valid_moves, [score_move(grid, col, obs.mark, config) for col in valid_moves]))
    # Get a list of columns (moves) that maximize the heuristic
    max_cols = [key for key in scores.keys() if scores[key] == max(scores.values())]
    # Select at random from the maximize

class ConnectFourGym():
    #     def __init__(self, agent2="random"):
    def __init__(self, opponent_pool=np.asarray(['random']), distribution='even'):
        self.ks_env = make("connectx", debug=True)
        #         self.env = self.ks_env.train([None, agent2])
        self.rows = self.ks_env.configuration.rows
        self.columns = self.ks_env.configuration.columns
        # Learn about spaces here: http://gym.openai.com/docs/#spaces
        self.action_space = gym.spaces.Discrete(self.columns)
        self.observation_space = gym.spaces.Box(low=0, high=1,
                                                shape=(1, self.rows, self.columns), dtype=np.float)
        # Tuple corresponding to the min and max possible rewards
        self.reward_range = (-10, 1)
        # StableBaselines throws error if these are not defined
        self.spec = None
        self.metadata = None
        self.last_action = -1
        self.iter = 0
        self.opponent_pool = opponent_pool
        self.distribution = distribution
        self.init_env()

    def init_env(self):
        if self.distribution == 'even':
            distribution = [1.0 / len(self.opponent_pool)] * len(self.opponent_pool)
        else:
            distribution = self.distribution
        opponent = choice(self.opponent_pool, 1, p=distribution)[0]
        self.env = self.ks_env.train([None, opponent])

    #         if self.iter % 2:
    #             self.env = self.ks_env.train([None, opponent])
    #         else:
    #             self.env = self.ks_env.train([opponent, None])

    def reset(self):
        self.iter += 1
        self.init_env()
        self.obs = self.env.reset()
        self.last_action = -1
        return board_flip(self.obs.mark, np.array(self.obs['board']).reshape(1, self.rows, self.columns) / 2)

    def change_reward(self, old_reward, done):
        if old_reward == 1:  # The agent won the game
            return 1
        elif done:  # The opponent won the game
            return -1
        else:  # Reward 1/42
            return 1 / (self.rows * self.columns)

    def step(self, action):
        # Check if agent's move is valid
        is_valid = (self.obs['board'][int(action)] == 0)
        if is_valid:  # Play the move
            self.obs, old_reward, done, _ = self.env.step(int(action))
            reward = self.change_reward(old_reward, done)
        else:  # End the game and penalize agent
            reward, done, _ = -10, True, {}
        return board_flip(self.obs.mark,
                          np.array(self.obs['board']).reshape(1, self.rows, self.columns) / 2), reward, done, _


env = ConnectFourGym([minimax_agent, 'random'])
# env = ConnectFourGym(['random'])
env

# # Create directory for logging training information
# log_dir = "log/"
# os.makedirs(log_dir, exist_ok=True)
#
# # Logging progress
# env = Monitor(env, log_dir, allow_early_resets=True)
# env

vec_env = DummyVecEnv([lambda: env])
vec_env


class Net(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 512):
        super(Net, self).__init__(observation_space, features_dim)
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
        self.fc3 = nn.Linear(384, features_dim)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = nn.Flatten()(x)
        x = F.relu(self.fc3(x))
        return x


policy_kwargs = {
    'activation_fn': th.nn.ReLU,
    'net_arch': [64, dict(pi=[32, 16], vf=[32, 16])],
    'features_extractor_class': Net,
}
learner = PPO('MlpPolicy', vec_env, policy_kwargs=policy_kwargs)

eval_callback = SaveBestModelCallback('RDaneelConnect4_', 1000, ['random', minimax_agent])

learner.learn(total_timesteps=200_000, callback=eval_callback)

# df = load_results(log_dir)['r']
# df.rolling(window=1000).mean().plot()
# df.tail(1000).mean()

def testagent(obs, config):
    import numpy as np
    obs = np.array(obs['board']).reshape(1, config.rows, config.columns) / 2
    action, _ = learner.predict(obs)
    return int(action)

get_win_percentages_and_score(agent1=testagent, agent2=testagent)

agent_path = 'submission.py'

submission_beginning = '''def agent(obs, config):
    import numpy as np
    import torch as th
    from torch import nn as nn
    import torch.nn.functional as F
    from torch import tensor

    class Net(nn.Module):
        def __init__(self):
            super(Net, self).__init__()
            self.conv1 = nn.Conv2d(1, 32, kernel_size=3)
            self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
            self.fc3 = nn.Linear(384, 512)
            self.shared1 = nn.Linear(512, 64)
            self.policy1 = nn.Linear(64, 32)
            self.policy2 = nn.Linear(32, 16)
            self.action = nn.Linear(16, 7)

        def forward(self, x):
            x = F.relu(self.conv1(x))
            x = F.relu(self.conv2(x))
            x = nn.Flatten()(x)
            x = F.relu(self.fc3(x))
            x = F.relu(self.shared1(x))
            x = F.relu(self.policy1(x))
            x = F.relu(self.policy2(x))
            x = self.action(x)
            x = x.argmax()
            return x


'''

with open(agent_path, mode='w+') as file:
    # file.write(f'\n    data = {learner.policy._get_data()}\n')
    file.write(submission_beginning)

th.set_printoptions(profile="full")

state_dict = learner.policy.to('cpu').state_dict()
state_dict = {
    'conv1.weight': state_dict['features_extractor.conv1.weight'],
    'conv1.bias': state_dict['features_extractor.conv1.bias'],
    'conv2.weight': state_dict['features_extractor.conv2.weight'],
    'conv2.bias': state_dict['features_extractor.conv2.bias'],
    'fc3.weight': state_dict['features_extractor.fc3.weight'],
    'fc3.bias': state_dict['features_extractor.fc3.bias'],

    'shared1.weight': state_dict['mlp_extractor.shared_net.0.weight'],
    'shared1.bias': state_dict['mlp_extractor.shared_net.0.bias'],

    'policy1.weight': state_dict['mlp_extractor.policy_net.0.weight'],
    'policy1.bias': state_dict['mlp_extractor.policy_net.0.bias'],
    'policy2.weight': state_dict['mlp_extractor.policy_net.2.weight'],
    'policy2.bias': state_dict['mlp_extractor.policy_net.2.bias'],

    'action.weight': state_dict['action_net.weight'],
    'action.bias': state_dict['action_net.bias'],
}

with open(agent_path, mode='a') as file:
    # file.write(f'\n    data = {learner.policy._get_data()}\n')
    file.write(f'    state_dict = {state_dict}\n')

submission_ending = '''    model = Net()
    model = model.float()
    model.load_state_dict(state_dict)
    model = model.to('cpu')
    model = model.eval()
    obs = tensor(obs['board']).reshape(1, 1, config.rows, config.columns).float()
    obs = obs / 2
    action = model(obs)
    return int(action)'''

with open(agent_path, mode='a') as file:
    # file.write(f'\n    data = {learner.policy._get_data()}\n')
    file.write(submission_ending)

# load submission.py
f = open(agent_path)
source = f.read()
exec(source)

# get_win_percentages_and_score(agent1=agent, agent2="random")


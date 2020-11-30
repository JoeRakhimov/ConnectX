import gym

import torch as th
from stable_baselines3.common.utils import set_random_seed
from torch import nn as nn
import torch.nn.functional as F

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

from agents import agentc1, agentc2, agentc3, agentc5, matrix_agent, rule_based_agent
from common import get_win_percentages_and_score
from connect4gym3 import SaveBestModelCallback, ConnectFourGym
import multiprocessing

from typing import Callable

if __name__ == '__main__':

    iterations = 100_000

    agents = ['random', matrix_agent, rule_based_agent, agentc1, agentc2, agentc3, agentc5]

    def make_env(seed: int = 0) -> Callable:
        """
        Utility function for multiprocessed env.

        :param env_id: (str) the environment ID
        :param num_env: (int) the number of environment you wish to have in subprocesses
        :param seed: (int) the inital seed for RNG
        :param rank: (int) index of the subprocess
        :return: (Callable)
        """
        def _init() -> gym.Env:
            env = ConnectFourGym(agents)
            return env

        set_random_seed(seed)
        return _init

    # env = ConnectFourGym(agents, id=env_id)
    # env
    #
    # vec_env = DummyVecEnv([lambda: env])
    # vec_env

    num_cpu = multiprocessing.cpu_count()  # Number of processes to use

    # Create the vectorized environment
    vec_env = SubprocVecEnv([make_env(i) for i in range(num_cpu)])

    # 1
    # https://www.kaggle.com/toshikazuwatanabe/connect4-make-submission-with-stable-baselines3
    # class Net(BaseFeaturesExtractor):
    #
    #     def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 512):
    #
    #         super(Net, self).__init__(observation_space, features_dim)
    #         print(observation_space)
    #         self.conv1 = nn.Conv2d(3, 32, kernel_size=3)
    #         self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
    #         self.fc3 = nn.Linear(384, features_dim)
    #
    #     def forward(self, x):
    #         x = F.relu(F.batch_norm(self.conv1(x), running_mean=None, running_var=None, training=True))
    #         x = F.relu(F.batch_norm(self.conv2(x), running_mean=None, running_var=None, training=True))
    #         x = nn.Flatten()(x)
    #         x = F.relu(self.fc3(x))
    #         x = F.dropout(x)
    #
    #         return x

        # 2
        # https://github.com/PaddlePaddle/PARL/blob/0915559a1dd1b9de74ddd2b261e2a4accd0cd96a/benchmark/torch/AlphaZero/submission_template.py#L423
    # class Net(BaseFeaturesExtractor):
    #
    #     def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 512):
    #         super(Net, self).__init__(observation_space, features_dim)
    #
    #         self.conv1 = nn.Conv2d(1, 64, 3, stride=1, padding=1)
    #         self.conv2 = nn.Conv2d(64, 64, 3, stride=1, padding=1)
    #         self.conv3 = nn.Conv2d(64, 64, 3, stride=1)
    #         self.conv4 = nn.Conv2d(64, 64, 3, stride=1)
    #
    #         self.bn1 = nn.BatchNorm2d(64)
    #         self.bn2 = nn.BatchNorm2d(64)
    #         self.bn3 = nn.BatchNorm2d(64)
    #         self.bn4 = nn.BatchNorm2d(64)
    #
    #         self.fc1 = nn.Linear(64 * (7 - 4) * (6 - 4), 128)
    #         self.fc_bn1 = nn.BatchNorm1d(128)
    #
    #         self.fc2 = nn.Linear(128, 64)
    #         self.fc_bn2 = nn.BatchNorm1d(64)
    #
    #         self.fc3 = nn.Linear(64, features_dim)
    #
    #         self.fc4 = nn.Linear(64, 1)
    #
    #     def forward(self, s):
    #         #                                                            s: batch_size x board_x x board_y
    #         s = s.view(-1, 1, 7, 6)  # batch_size x 1 x board_x x board_y
    #         s = F.relu(self.bn1(self.conv1(s)))  # batch_size x num_channels x board_x x board_y
    #         s = F.relu(self.bn2(self.conv2(s)))  # batch_size x num_channels x board_x x board_y
    #         s = F.relu(self.bn3(self.conv3(s)))  # batch_size x num_channels x (board_x-2) x (board_y-2)
    #         s = F.relu(self.bn4(self.conv4(s)))  # batch_size x num_channels x (board_x-4) x (board_y-4)
    #         s = s.view(-1,64 * (7 - 4) * (6 - 4))
    #
    #         s = F.dropout(
    #             F.relu(self.fc_bn1(self.fc1(s))),
    #             p=0.3,
    #             training=self.training)  # batch_size x 128
    #         s = F.dropout(
    #             F.relu(self.fc_bn2(self.fc2(s))),
    #             p=0.3,
    #             training=self.training)  # batch_size x 64
    #
    #         pi = self.fc3(s)  # batch_size x action_size
    #         # v = self.fc4(s)  # batch_size x 1
    #
    #         # return F.log_softmax(pi), th.tanh(v)
    #         return F.log_softmax(pi)

    # 3
    # https://github.com/boettiger-lab/gym_wildfire/blob/26127f389566ab920107bfbc65d56bcc16199d1a/examples/sb3/ppo.py
    # class Net(BaseFeaturesExtractor):
    #     def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 256):
    #         super(Net, self).__init__(observation_space, features_dim)
    #         n_input_channels = observation_space.shape[0]
    #         self.cnn = nn.Sequential(
    #             nn.Conv2d(n_input_channels, 32, kernel_size=3, stride=1, padding=0),
    #             nn.ReLU(),
    #             nn.Conv2d(32, 64, kernel_size=2, stride=1, padding=0),
    #             nn.ReLU(),
    #             nn.Flatten(),
    #         )
    #         with th.no_grad():
    #             n_flatten = self.cnn(
    #                 th.as_tensor(observation_space.sample()[None]).float()
    #             ).shape[1]
    #         self.linear = nn.Sequential(nn.Linear(n_flatten, features_dim), nn.ReLU())
    #
    #     def forward(self, observations: th.Tensor) -> th.Tensor:
    #         return self.linear(self.cnn(observations))

    # 4
    # https://github.com/greentfrapp/snake/blob/45a9dce92c092b5c65412a666892c71360524772/train_vanilla.py
    class Net(BaseFeaturesExtractor):
        def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 256):
            super(Net, self).__init__(observation_space, features_dim)
            # We assume CxHxW images (channels first)
            # Re-ordering will be done by pre-preprocessing or wrapper
            n_input_channels = observation_space.shape[0]
            self.cnn = nn.Sequential(
                nn.Conv2d(n_input_channels, 32, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=0),
                nn.ReLU(),
                nn.Flatten(),
            )

            # Compute shape by doing one forward pass
            with th.no_grad():
                n_flatten = self.cnn(th.as_tensor(observation_space.sample()[None]).float()).shape[1]

            self.linear = nn.Sequential(nn.Linear(n_flatten, features_dim), nn.ReLU())

        def forward(self, observations: th.Tensor) -> th.Tensor:
            return self.linear(self.cnn(observations))

    # 5
    # https://github.com/denisergashbaev/rl_playground/blob/f4a80c3f0e4ace462b40a87b881bffd0387e7759/algs/dqn/reference/network.py
    # class Net(BaseFeaturesExtractor):
    #     """
    #     :param observation_space: (gym.Space)
    #     :param features_dim: (int) Number of features extracted.
    #         This corresponds to the number of unit for the last layer.
    #     """
    #
    #     def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 128):
    #         super(Net, self).__init__(observation_space, features_dim)
    #         # We assume CxHxW images (channels first)
    #         # Re-ordering will be done by pre-preprocessing or wrapper
    #         in_channels = observation_space.shape[0]
    #         self.cnn = nn.Sequential(
    #             nn.Conv2d(in_channels, 32, kernel_size=1, stride=1),
    #             nn.ReLU(),
    #             nn.Conv2d(32, 64, kernel_size=3, stride=1),
    #             nn.ReLU(),
    #             nn.Conv2d(64, 64, kernel_size=3, stride=1),
    #             nn.ReLU(),
    #             nn.Conv2d(64, 128, kernel_size=2, stride=1),
    #             nn.ReLU(),
    #             nn.Flatten(),
    #         )
    #
    #         # Compute shape by doing one forward pass
    #         with th.no_grad():
    #             n_flatten = self.cnn(th.as_tensor(observation_space.sample()[None]).float()).shape[1]
    #
    #         self.linear = nn.Sequential(nn.Linear(n_flatten, features_dim), nn.ReLU())
    #
    #     def forward(self, observations: th.Tensor) -> th.Tensor:
    #         return self.linear(self.cnn(observations))

    # device = th.device("cuda" if th.cuda.is_available() else "cpu")  # PyTorch v0.4.0
    # model = Net(None).to(device)
    # summary(model, (1, 6, 7))

    policy_kwargs = {
        'activation_fn': th.nn.ReLU,
        'net_arch': [64, dict(pi=[32, 16], vf=[32, 16])],
        'features_extractor_class': Net,
    }

    learner = PPO('MlpPolicy', vec_env, policy_kwargs=policy_kwargs)
    # learner = PPO('MlpPolicy', vec_env)

    eval_callback = SaveBestModelCallback('RDaneelConnect4_', 1000, agents)

    learner.learn(total_timesteps=iterations, callback=eval_callback)

    def testagent(obs, config):
        import numpy as np
        obs = np.array(obs['board']).reshape(1, config.rows, config.columns) / 2
        action, _ = learner.predict(obs)
        return int(action)

    get_win_percentages_and_score(agent1=testagent, agent2='random')

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
            x = F.relu(F.batch_norm(self.conv1(x), running_mean=None, running_var=None, training=True))
            x = F.relu(F.batch_norm(self.conv2(x), running_mean=None, running_var=None, training=True))
            x = nn.Flatten()(x)
            x = F.relu(self.fc3(x))
            x = F.dropout(x)
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
        file.write(f'state_dict = {state_dict}\n')

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

from deep_rl import *
import matplotlib.pyplot as plt
import torch
from tqdm import trange, tqdm
import random
import numpy as np

select_device(0)

class NatureConvBody(nn.Module):
    def __init__(self, in_channels=4):
        super(NatureConvBody, self).__init__()
        self.feature_dim = 512
        self.conv1 = layer_init(nn.Conv2d(in_channels, 32, kernel_size=3, stride=2))
        self.conv2 = layer_init(nn.Conv2d(32, 64, kernel_size=3, stride=2))
#         self.conv3 = layer_init(nn.Conv2d(64, 64, kernel_size=3, stride=2))
        self.fc4 = layer_init(nn.Linear(9 * 9 * 64, self.feature_dim))

    def forward(self, x):
        y = F.relu(self.conv1(x))
        y = F.relu(self.conv2(y))
#         y = F.relu(self.conv3(y))
        y = y.view(y.size(0), -1)
        y = F.relu(self.fc4(y))
        return y

def dqn_feature(**kwargs):
    generate_tag(kwargs)
    kwargs.setdefault('log_level', 0)
    config = Config()
    config.merge(kwargs)

    config.task_fn = lambda: Task(config.game)
    config.eval_env = config.task_fn()
    # config.action_dim = 3

    config.optimizer_fn = lambda params: torch.optim.RMSprop(
        params, lr=0.001, centered=True)
#     config.network_fn = lambda: VanillaNet(config.action_dim, FCBody(config.state_dim, hidden_units=(43,)))
    config.network_fn = lambda: VanillaNet(config.action_dim, NatureConvBody(in_channels=3))
#     print(config.action_dim)
    config.replay_fn = lambda: Replay(memory_size=int(2e5), batch_size=32)
    # config.replay_fn = lambda: AsyncReplay(memory_size=int(2e5), batch_size=32)
    config.batch_size = 32
    config.state_normalizer = ImageNormalizer()
    config.reward_normalizer = SignNormalizer()

    config.random_action_prob = LinearSchedule(1.0, 0.01, 5e5)
    config.discount = 0.99
    config.target_network_update_freq = 5000
    config.exploration_steps = 10000
    # config.double_q = True
    config.double_q = False
    config.sgd_update_frequency = 4
    config.gradient_clip = 5
    config.eval_interval = int(5e10)
    config.max_steps = 1e6
    config.async_actor = False
    agent = DQNAgent(config)
    #run_steps function below
    config = agent.config
    agent_name = agent.__class__.__name__
    t0 = time.time()
    for i in tqdm(range(int(config.max_steps))):
        if config.save_interval and not agent.total_steps % config.save_interval:
            agent.save('data/%s-%s-%d' % (agent_name, config.tag, agent.total_steps))
        if config.log_interval and not agent.total_steps % config.log_interval:
            t0 = time.time()
        if config.eval_interval and not agent.total_steps % 5000:
#             agent.eval_episodes()
            print(agent.total_steps)
            pass
        if config.max_steps and agent.total_steps >= config.max_steps:
            return agent
            break
        agent.step()
        agent.switch_task()
    return agent

game = 'MiniGrid-Empty-5x5-v0'
agent = dqn_feature(game=game)
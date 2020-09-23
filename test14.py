from deep_rl import *
import matplotlib.pyplot as plt
import torch
from tqdm import trange, tqdm
import random
import numpy as np

FEATURE_DIMS = 512
VERSION = 'psi'

def dqn_feature_v2(weights, **kwargs):
    generate_tag(kwargs)
    kwargs.setdefault('log_level', 0)
    config = Config()
    config.merge(kwargs)

    config.task_fn = lambda: Task(config.game)
    config.eval_env = config.task_fn()

    config.optimizer_fn = lambda params: torch.optim.RMSprop(
        params, lr=0.0005, alpha=0.95, eps=0.01, centered=True)
    if(VERSION == 'phi'):
        # For psi version
        config.network_fn = lambda: SRNetNature_v2_phi(output_dim=config.action_dim, feature_dim=FEATURE_DIMS, hidden_units_psi2q=(2048,512))
    else:
        # For psi version
        config.network_fn = lambda: SRNetNature_v2_psi(output_dim=config.action_dim, feature_dim=FEATURE_DIMS, hidden_units_sr=(512*4,), hidden_units_psi2q=(2048,512))

    config.replay_fn = lambda: Replay(memory_size=int(2e5), batch_size=10)

    config.random_action_prob = LinearSchedule(1.0, 0.1, 3e6)
    config.discount = 0.99
    config.target_network_update_freq = 2000
    config.exploration_steps = 50000
    # config.double_q = True
    config.double_q = False
    config.sgd_update_frequency = 4
    config.gradient_clip = 5
    config.eval_interval = int(5e3)
    config.max_steps = 5e6
    config.async_actor = False

    agent = DQNAgent_v2(config)
    #run_steps function below
    config = agent.config
    agent_name = agent.__class__.__name__
    if(weights is not None):
        print(agent.network.load_state_dict(weights, strict=False))
    t0 = time.time()
    while True:
        if config.save_interval and not agent.total_steps % config.save_interval:
            agent.save('data/%s-%s-%d' % (agent_name, config.tag, agent.total_steps))
        if config.log_interval and not agent.total_steps % config.log_interval:
            agent.logger.info('steps %d, %.2f steps/s' % (agent.total_steps, config.log_interval / (time.time() - t0)))
            t0 = time.time()
        if config.eval_interval and not agent.total_steps % config.eval_interval:
            print(agent.total_steps)
            # agent.eval_episodes()
            pass
        if config.max_steps and agent.total_steps >= config.max_steps:
            return agent
            break
        # import pdb; pdb.set_trace()
        agent.step()
        agent.switch_task()
        
    return agent

GAME = 'BoxingNoFrameskip-v0'
READFILE = 'storage/41-avdsr-trained-boxing-' + str(FEATURE_DIMS) + '.weights'

weights = torch.load(READFILE).state_dict()

if(VERSION == 'phi'):
    # For phi version
    to_remove = ['decoder.0.weight', 'decoder.0.bias', 'decoder.2.weight', 'decoder.2.bias', 'decoder.4.weight', 'decoder.4.bias', 'decoder.6.weight', 'decoder.6.bias', 'layers_sr.0.weight', 'layers_sr.0.bias', 'layers_sr.1.weight', 'layers_sr.1.bias', 'psi2q.layers.0.weight', 'psi2q.layers.0.bias']
else:
    # For psi version
    to_remove = ['decoder.0.weight', 'decoder.0.bias', 'decoder.2.weight', 'decoder.2.bias', 'decoder.4.weight', 'decoder.4.bias', 'decoder.6.weight', 'decoder.6.bias', 'psi2q.layers.0.weight', 'psi2q.layers.0.bias']

for key in to_remove:
    weights.pop(key)

select_device(0)
# agent = dsr_feature_init(game=GAME, freeze=2, weights=weights)
agent = dqn_feature_v2(game=GAME, weights=weights, is_wb=True)
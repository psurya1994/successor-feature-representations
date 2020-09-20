from deep_rl import *
import matplotlib.pyplot as plt
import torch
from tqdm import trange, tqdm
import random
import numpy as np



def dsr_feature_init(weights,**kwargs):
    generate_tag(kwargs)
    kwargs.setdefault('log_level', 0)
    config = Config()
    config.merge(kwargs)

    config.task_fn = lambda: Task(config.game)
    config.eval_env = config.task_fn()
    config.c = 1

    config.optimizer_fn = lambda params: torch.optim.RMSprop(params, 0.001/2)
    config.network_fn = lambda: SRNetImage(7, config=1, hidden_units_psi2q=(1024,512))
    config.replay_fn = lambda: Replay(memory_size=int(1e5), batch_size=10)

    config.random_action_prob = LinearSchedule(1.0, 0.1, 1e5)
    config.discount = 0.99
    config.target_network_update_freq = 200
    config.exploration_steps = 0
    config.double_q = False
    config.sgd_update_frequency = 4
    config.gradient_clip = 5
    config.eval_interval = int(5e3)
    config.max_steps = 2e5
    config.async_actor = False
    
    agent = DSRAgent_v2(config)
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
#             agent.logger.info('steps %d, %.2f steps/s' % (agent.total_steps, config.log_interval / (time.time() - t0)))
            t0 = time.time()
        if config.eval_interval and not agent.total_steps % config.eval_interval:
            print(agent.total_steps)
#             agent.eval_episodes()
            pass
        if config.max_steps and agent.total_steps >= config.max_steps:
            return agent
            break
#         import pdb; pdb.set_trace()
        agent.step()
        agent.switch_task()
        
    return agent

GAME = 'MiniGrid-Empty-5x5-v0'
READFILE = 'storage/40-avdsr-trained-minigrid.weights'

weights = torch.load(READFILE).state_dict()
# Remove psi2q weights, I'm using a deep net for psi2q (weights here are for linear untrained)
weights.pop('psi2q.layers.0.weight')
weights.pop('psi2q.layers.0.bias')
select_device(0)
agent = dsr_feature_init(game=GAME, freeze=2, weights=weights)
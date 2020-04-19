import os
import sys
import gym
import random
import numpy as np
import json
import torch
import torch.optim as optim
import torch.nn.functional as F
from model import QNet
from memory import Memory
from tensorboardX import SummaryWriter
import time
import pdb
from baselines.common.vec_env.shmem_vec_env import ShmemVecEnv

evalution_mode = True
cartpole_test = False
need_ui = False

if (not cartpole_test):
    import gym_arc    
    sys.path.append('../gym_arc/envs/')

from config import env_name, initial_exploration, batch_size, update_target, goal_score, log_interval, device, replay_memory_capacity, lr

def get_action(state, target_net, epsilon, env):
    if np.random.rand() <= epsilon:
        return env.action_space.sample()
    else:
        return target_net.get_action(state)

def update_target_model(online_net, target_net):
    target_net.load_state_dict(online_net.state_dict())

def main():
    # cartpole test
    if (cartpole_test):
        envs_fun = [lambda: gym.make('CartPole-v0')]
        envs_fun = np.tile(envs_fun, 3)
        envs = ShmemVecEnv(envs_fun)
        dummy_env = envs_fun[0]()
    else:
        INPUT_FILE = '../data/05f2a901.json'
        with open(INPUT_FILE,'r') as f:
            puzzle = json.load(f)

        envs_fun = [lambda: gym.make('arc-v0', input=task['input'], output=task['output'], need_ui=need_ui) for task in puzzle['train'] ]
        #pdb.set_trace()
        envs_fun = envs_fun[0:1]
        envs = ShmemVecEnv(envs_fun)
        dummy_env = envs_fun[0]()

    env_num = len(envs_fun)
    torch.manual_seed(500)

    num_inputs = dummy_env.observation_space.shape[0]
    num_actions = dummy_env.action_space.n
    print('state size:', num_inputs)
    print('action size:', num_actions)

    online_net = QNet(num_inputs, num_actions, cartpole_test, evalution_mode)
    target_net = QNet(num_inputs, num_actions, cartpole_test, evalution_mode)

    if (evalution_mode):
        online_net = torch.load('../result/arc0.model')
        target_net = torch.load('../result/arc0.model')

    update_target_model(online_net, target_net)

    optimizer = optim.Adam(online_net.parameters(), lr=lr)
    writer = SummaryWriter('logs')

    online_net.to(device)
    target_net.to(device)
    online_net.train()
    target_net.train()
    memory = Memory(replay_memory_capacity)

    score = 0
    epsilon = 1.0
    steps = 0
    loss = 0

    states = envs.reset()
        
    try:
        while True:
            if (need_ui):
                envs.render()
            steps += 1

            global initial_exploration
            if (initial_exploration > 0):
                initial_exploration -= 1

            actions = []

            for state in states:
                state = torch.Tensor(state).to(device)
                state = state.unsqueeze(0)
                action = get_action(state, target_net, 0 if evalution_mode else epsilon, dummy_env)
                if (evalution_mode):
                    print(action)
                actions.append(action)

            next_states, rewards, dones, info = envs.step(actions)
            #print(rewards)

            masks = np.zeros(envs.num_envs)
            for i in range(envs.num_envs):
                masks[i] = 0 if dones[i] else 1

            for i in range(envs.num_envs):
                #print(rewards[i])
                action_one_hot = np.zeros(dummy_env.action_space.n)
                action_one_hot[actions[i]] = 1
                memory.push(states[i], next_states[i], action_one_hot, rewards[i], masks[i])

            #score += reward
            states = next_states

            if not evalution_mode and steps > initial_exploration:
                epsilon -= 0.00003
                epsilon = max(epsilon, 0.1)

                batch = memory.sample(batch_size)
                loss = QNet.train_model(online_net, target_net, optimizer, batch, device)

                if steps % update_target == 0:
                    update_target_model(online_net, target_net)

            if (steps > 1028):
                states = envs.reset()
                steps = 0
                print('new epsisode ------------------------------------------')

    except KeyboardInterrupt:
        print('save model')
        torch.save(target_net, '../result/arc.model')
        sys.exit(0)


if __name__=="__main__":
    main()

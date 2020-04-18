import os
import sys
import gym
import random
import numpy as np
import json
import gym_arc
import time
import pdb
from baselines.common.vec_env.shmem_vec_env import ShmemVecEnv

import torch
import torch.optim as optim
import torch.nn.functional as F
from model import QNet
from memory import Memory
from tensorboardX import SummaryWriter

from config import env_name, initial_exploration, batch_size, update_target, goal_score, log_interval, device, replay_memory_capacity, lr

def get_action(state, target_net, epsilon, env):
    if np.random.rand() <= epsilon:
        return env.action_space.sample()
    else:
        return target_net.get_action(state)

def update_target_model(online_net, target_net):
    # Target <- Net
    target_net.load_state_dict(online_net.state_dict())


def main():
    INPUT_FILE = '../data/05f2a901.json'
    with open(INPUT_FILE,'r') as f:
        puzzle = json.load(f)

    puzzle_input = puzzle['train'][0]['input']
    puzzle_output = puzzle['train'][0]['output']
    dummy_env = gym.make('arc-v0', input=puzzle_input, output=puzzle_output, need_ui=False)

    need_ui = True
    envs = []

    

    # for task in puzzle['train']:
    #     e = gym.make('arc-v0', input=task['input'], output=task['output'], need_ui=need_ui)
    #     e.seed(500)
    #     envs.append(e)

    #pdb.set_trace()
    envs = [lambda: gym.make('arc-v0', input=task['input'], output=task['output'], need_ui=need_ui) for task in puzzle['train'] ]

    #envs = envs[0:1]

    envs = ShmemVecEnv(envs)
    #pdb.set_trace()

    #env = gym.make(env_name)
    #env.seed(500)
    torch.manual_seed(500)

    num_inputs = dummy_env.observation_space.shape[0]
    num_actions = dummy_env.action_space.n
    
    print('state size:', num_inputs)
    print('action size:', num_actions)

    online_net = QNet(num_inputs, num_actions)
    target_net = QNet(num_inputs, num_actions)
    update_target_model(online_net, target_net)

    optimizer = optim.Adam(online_net.parameters(), lr=lr)
    writer = SummaryWriter('logs')

    online_net.to(device)
    target_net.to(device)
    online_net.train()
    target_net.train()
    memory = Memory(replay_memory_capacity)
    running_score = 0
    epsilon = 1.0
    steps = 0
    loss = 0

    for e in range(3000):
        
        done = False

        score = 0
        #pdb.set_trace()
        states = envs.reset()
        #state = state.flatten()
        
        

        while not done:
            #envs.render()
            steps += 1

            actions = []

            for state in states:
                state = torch.Tensor(state).to(device)
                state = state.unsqueeze(0)
                action = get_action(state, target_net, epsilon, dummy_env)
                actions.append(action)

            #print(actions)

            #action = 9
            #pdb.set_trace()
            next_states, rewards, dones, _ = envs.step(actions)
            masks = np.zeros(len(rewards))

            for i in range(len(rewards)):
                if (rewards[i] < 0):
                    dones[i] = True
                masks[i] = 0 if done else 1

            #if (reward  < 0): done = True

            #next_state = next_state.flatten()
            # next_state = torch.Tensor(next_state)
            # next_state = next_state.unsqueeze(0)

            #mask = 0 if done else 1
            #reward = reward if not done or score == 499 else -1

            # if (reward > 0):
            #     print(reward)
            

            for i in range(len(rewards)):
                action_one_hot = np.zeros(dummy_env.action_space.n)
                action_one_hot[actions[i]] = 1
                memory.push(states[i], next_states[i], action_one_hot, rewards[i], masks[i])

            #score += reward
            state = next_states

            if steps > initial_exploration:
                epsilon -= 0.00003
                epsilon = max(epsilon, 0.1)

                batch = memory.sample(batch_size)
                #pdb.set_trace()
                loss = QNet.train_model(online_net, target_net, optimizer, batch)

                if steps % update_target == 0:
                    update_target_model(online_net, target_net)

        #score = score if score == 500.0 else score + 1
        #running_score = 0.99 * running_score + 0.01 * score
        # if e % 1 == 0:
        #     print('{} episode | score: {:.2f} | epsilon: {:.2f}'.format(
        #         e, score, epsilon))
        #     writer.add_scalar('log/score', float(score), e)
        #     writer.add_scalar('log/loss', float(loss), e)

        if running_score > goal_score:
            break

        #time.sleep(0.01)


if __name__=="__main__":
    main()

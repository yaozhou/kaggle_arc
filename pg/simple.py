#!/bin/python
# -*- coding: UTF-8 -*-

import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical
from torch.optim import Adam
import numpy as np
import gym
from gym.spaces import Discrete, Box
import sys
import json
import pdb
import os
import gym_arc
import time
sys.path.append('./gym_arc/envs/')
from config import LR, EPOCHS, batch_size, RENDER, MODEL, INPUT_FILE, STEPS_LIMIT, DECAY, FILE_ID, device, action_model

# soft-ac, ppo
# 怎样优化重复动作，无意义动作
# 找最核心特征
# 适配所有题目，看是否泛化

def generate_model(feature_num, action_num):
    hidden_size = 82

    net = nn.Sequential(
        nn.Linear(feature_num, hidden_size),
        nn.Dropout(0.3),
        nn.ReLU(),
        nn.Linear(hidden_size, hidden_size),
        nn.Dropout(0.3),
        nn.ReLU(),
        nn.Linear(hidden_size, hidden_size),
        nn.Dropout(0.3),
        nn.ReLU(),
        nn.Linear(hidden_size, action_num),
        nn.Softmax()
    )

    for layer in net:
        if isinstance(layer, nn.Linear):
            nn.init.xavier_uniform(layer.weight)
    
    return net

def get_categorical(model, obs):
    model.eval()
    probs = model(obs)
    return Categorical(probs=probs)

def get_action(model, obs):
    return get_categorical(model, obs).sample().item()

def compute_loss(model, obs, action, weights):
    logp = get_categorical(model, obs).log_prob(action)
    return -(logp * weights).mean()

def train():
    with open(INPUT_FILE, 'r') as f:
        puzzle = json.load(f)

    envs = []
    env_idx = 0

    for i in range(len(puzzle['train'])):
        p = puzzle['train'][i]
        e = gym.make('arc-v0', input=p['input'], output=p['output'], need_ui=RENDER, action_mode='single')
        envs.append(e)

    #envs = envs[:1]
    #pdb.set_trace()

    test_input = puzzle['test'][0]['input']
    test_output = puzzle['test'][0]['output']
    env_test = gym.make('arc-v0', input=test_input, output=test_output, need_ui=RENDER, action_mode='single')

    n_feature = env_test.observation_space.shape[0]
    n_acts = env_test.action_space.n
    print('feature(%d) actions(%d)' % (n_feature, n_acts))

    #
    if (MODEL == ''):
        net = generate_model(n_feature, n_acts)
        net.train()
    else:
        net = torch.load(MODEL)
        envs.append(env_test)
        print(envs)
        net.eval()

    net.to(device)

    optimizer = Adam(net.parameters(), lr=LR)

    for epoch_idx in range(EPOCHS):
        batch_obs = []
        batch_acts = []
        batch_weights = []
        rewards = []

        env = envs[0]
        obs = env.reset()
        steps = 0

        while True:
            steps += 1

            if RENDER: env.render()

            act = get_action(net, torch.as_tensor(obs, dtype = torch.float32).to(device))
            next_obs, r, done, info = env.step(act)
            
            if (steps >= STEPS_LIMIT):
                done = True

            batch_acts.append(act)
            batch_obs.append(obs)
            rewards.append(r)

            obs = next_obs

            #time.sleep(5)

            if done:
                print('epoch_idx:%4d %100s     ----> %2.1f\t env:%2d' % (epoch_idx, info['steps'][:20], info['total_reward'], env_idx))

                if (len(envs) > 1):
                    env_idx = (env_idx + 1) % len(envs)
                    env = envs[env_idx]
                
                returns = [0] * len(rewards)
                acc_rewards = 0
                for i in reversed(range(len(rewards))):
                    acc_rewards = rewards[i] + DECAY * acc_rewards
                    returns[i] = acc_rewards
                batch_weights += returns

                obs, done, rewards, steps = env.reset(), False, [], 0

                if len(batch_obs) > batch_size:
                    break
        
        if (MODEL == ''):
            net.train()
            optimizer.zero_grad()
            print('----------------------------- learning --------------------------')
            loss = compute_loss(net,
                                    torch.as_tensor(batch_obs, dtype = torch.float32).to(device),
                                    torch.as_tensor(batch_acts, dtype = torch.int32).to(device),
                                    torch.as_tensor(batch_weights, dtype = torch.float32).to(device)
                                    )

            loss.backward()
            optimizer.step()

            if (epoch_idx % 20 == 0):
                torch.save(net, ('./result/%s_%4d.model' % (FILE_ID, epoch_idx)))

if __name__ == '__main__':
    train()
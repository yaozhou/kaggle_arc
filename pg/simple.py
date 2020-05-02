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
import cv2
import gym_arc
sys.path.append('./gym_arc/envs/')

# soft-ac, ppo
# 怎样优化重复动作，无意义动作

LR = 0.001
EPOCHS = 5000
batch_size = 5012
RENDER = False
MODEL = ''
INPUT_FILE = './data/0d3d703e.json'
STEPS_LIMIT = 800
DECAY = 0.9
FILE_ID = os.path.basename(INPUT_FILE).split('.')[0]

def generate_model(feature_num, action_num):
    hidden_size = 82

    net = nn.Sequential(
        nn.Linear(feature_num, hidden_size),
        nn.ReLU(),
        nn.Linear(hidden_size, hidden_size),
        nn.ReLU(),
        nn.Linear(hidden_size, hidden_size),
        nn.ReLU(),
        nn.Linear(hidden_size, action_num),
        nn.Softmax()
    )

    for layer in net:
        if isinstance(layer, nn.Linear):
            nn.init.xavier_uniform(layer.weight)
    
    return net

def get_categorical(model, obs):
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

    for p in puzzle['train']:
        e = gym.make('arc-v0', input=p['input'], output=p['output'], need_ui=RENDER)
        envs.append(e)

    #envs = envs[:1]

    test_input = puzzle['test'][0]['input']
    env_test = gym.make('arc-v0', input=test_input, output=test_input, need_ui=RENDER)

    n_feature = env_test.observation_space.shape[0]
    n_acts = env_test.action_space.n
    print('feature(%d) actions(%d)' % (n_feature, n_acts))

    if (MODEL == ''):
        net = generate_model(n_feature, n_acts)
    else:
        net = torch.load(MODEL)

    optimizer = Adam(net.parameters(), lr=LR)

    for epoch_idx in range(EPOCHS):
        batch_obs = []
        batch_acts = []
        batch_weights = []
        rewards = []

        env = envs[0]
        obs = env.reset()
        #actions = []
        steps = 0

        while True:
            steps += 1

            if RENDER: env.render()

            act = get_action(net, torch.as_tensor(obs, dtype = torch.float32))
            next_obs, r, done, info = env.step(act)
            #print(r)

            #if (steps == 0):
                #print('epoch(%d) env_idx(%d) step(%d) action(%d)' % (epoch_idx, env_idx, steps, act))
                #print(net(torch.as_tensor(obs, dtype = torch.float32)).detach().numpy())
            
            if (steps >= STEPS_LIMIT):
                done = True

            #actions.append(act)
            batch_acts.append(act)
            batch_obs.append(obs)
            rewards.append(r)

            obs = next_obs

            if done:
                #rint(actions[:STEPS_LIMIT])
                print('env:%2d %40s     ----> %5s' % (env_idx, info['steps'][:30], info['total_reward']))

                if (len(envs) > 1):
                    env_idx = (env_idx + 1) % 3
                    env = envs[env_idx]
                    #print('change env (%d)' % env_idx)

                
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
            optimizer.zero_grad()
            print('----------------------------- learning --------------------------')
            loss = compute_loss(net,
                                    torch.as_tensor(batch_obs, dtype = torch.float32),
                                    torch.as_tensor(batch_acts, dtype = torch.int32),
                                    torch.as_tensor(batch_weights, dtype = torch.float32)
                                    )

            loss.backward()
            optimizer.step()

            if (epoch_idx % 20 == 0):
                torch.save(net, ('./result/%s_%d.model' % (FILE_ID, epoch_idx)))

if __name__ == '__main__':
    train()
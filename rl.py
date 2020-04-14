import gym
import gym_arc
import pdb
import json
import time
import keyboard
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque
import random

EPSILON = 0.9
TARGET_REPLACE_ITER = 100
MEMORY_CAPACITY = 2560
BATCH_SIZE = 32
LR = 0.01
GAMMA = 0.999

class Net(nn.Module):
    def __init__(self, width, height, action_num):
        super(Net, self).__init__()

        self.layer = nn.Sequential(
            nn.Linear(width * height, 512),
            nn.ReLU(True),
            nn.Linear(512, 256),
            nn.ReLU(True),
            nn.Linear(256, action_num)
        )

    def forward(self, x):
        x = self.layer(x)

        return x

class DQN(object):
    EPSILON = 0.9

    def __init__(self, obs_shape, action_num):
        self.width = obs_shape[0]
        self.height = obs_shape[1]
        self.eval_net = Net(obs_shape[0], obs_shape[1], action_num)
        self.target_net = Net(obs_shape[0], obs_shape[1], action_num)
        self.action_num = action_num
        
        self.learn_step = 0
        self.memory = deque()
        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=LR)
        self.loss_func = nn.MSELoss()

    def choose_action(self, x):
        x = x.flatten()
        x = torch.unsqueeze(torch.FloatTensor(x), 0)
        if np.random.uniform() < 0.9:
            actions_value = self.eval_net.forward(x)
            action = torch.max(actions_value, 1)[1].data.numpy()[0]
        else:
            action = np.random.randint(0, self.action_num)
        return action

    def store_transition(self, s0, a, r, s1, done):
        s0 = s0.flatten()
        s1 = s1.flatten()
        self.memory.append((s0, a, r, s1, done))
        if (len(self.memory) > MEMORY_CAPACITY):
            self.memory.popleft()

    def learn(self):
        if self.learn_step % TARGET_REPLACE_ITER == 0:
            self.target_net.load_state_dict(self.eval_net.state_dict())
        self.learn_step += 1

        minibatch = random.sample(self.memory, BATCH_SIZE)

        s0_batch = np.array([d[0] for d in minibatch])
        a_batch = np.expand_dims(np.array([d[1] for d in minibatch]), axis=1)   # shape (batch, 1)
        r_batch = np.array([d[2] for d in minibatch])
        s1_batch = np.array([d[3] for d in minibatch])


        q_eval = self.eval_net(torch.FloatTensor(s0_batch)).gather(1, torch.tensor(a_batch))
        q_next = self.target_net(torch.FloatTensor(s1_batch)).detach()
        q_target = torch.FloatTensor(r_batch) + GAMMA * q_next.max(1)[0]
        loss = self.loss_func(q_eval.squeeze(1), q_target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()



action = None

def keyboard_hook(x):
    global action
    if (x.event_type != 'down' or x.name < '0' or x.name > '9'): return

    action = int(x.name)

INPUT_FILE = './data/05f2a901.json'
with open(INPUT_FILE,'r') as f:
    puzzle = json.load(f)
puzzle_input = puzzle['train'][0]['input']
puzzle_output = puzzle['train'][0]['output']

env = gym.make('arc-v0', input=puzzle_input, output=puzzle_output, need_ui=False)
print(env.observation_space.shape)
print(env.action_space.n)
#keyboard.hook(keyboard_hook)

dqn = DQN(env.observation_space.shape, env.action_space.n)
#s = env.reset()

for i_episode in range(400):
    s = env.reset()
    steps = 0
    while True:
        if (steps > 5120):
            break

        #env.render()
        a = dqn.choose_action(s)
        #a = env.action_space.sample()

        s1, r, done, info = env.step(a)
        if (r > 0):
            print('reward %f' % r)

        dqn.store_transition(s, a, r, s1, done)

        if (len(dqn.memory) == MEMORY_CAPACITY):
            dqn.learn()

        if done:
            print('episode success!')
            break

torch.save(dqn.target_net, './result/arc.model')
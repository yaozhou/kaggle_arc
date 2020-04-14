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
# import torchvision.transforms as T
from collections import deque
import random

EPSILON = 0.9
TARGET_REPLACE_ITER = 100
MEMORY_CAPACITY = 128
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

        # self.fc1 = nn.Linear(height * width, 500)
        # self.fc1.weight.data.normal_(0, 0.1)

        # self.fc2 = nn.Linear(width, 100)
        # self.fc2.weight.data.normal_(0, 0.1)

        # self.out = nn.Linear(100, action_num)
        # self.out.weight.data.normal_(0, 0.1)

    def forward(self, x):
        # x = self.fc1(x)
        # x = F.relu(x)
        # x = self.fc2(x)
        # x = F.relu(x)
        # x = self.out(x)
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

        #self.learn_step_counter = 0     # 用于 target 更新计时
        #self.memory_counter = 0         # 记忆库记数
        #self.memory = np.zeros(MEMORY_CAPACITY)     # 初始化记忆库
        
        self.learn_step = 0
        self.memory = deque()
        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=LR)    # torch 的优化器
        self.loss_func = nn.MSELoss()   # 误差公式

    def choose_action(self, x):
        #pdb.set_trace()
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
        #pdb.set_trace()
        if self.learn_step % TARGET_REPLACE_ITER == 0:
            self.target_net.load_state_dict(self.eval_net.state_dict())
        self.learn_step += 1

        minibatch = random.sample(self.memory, BATCH_SIZE)

        s0_batch = np.array([d[0] for d in minibatch])
        a_batch = np.expand_dims(np.array([d[1] for d in minibatch]), axis=1)   # shape (batch, 1)
        r_batch = np.array([d[2] for d in minibatch])
        s1_batch = np.array([d[3] for d in minibatch])

        #pdb.set_trace()

        q_eval = self.eval_net(torch.FloatTensor(s0_batch)).gather(1, torch.tensor(a_batch))
        q_next = self.target_net(torch.FloatTensor(s1_batch)).detach()
        q_target = torch.FloatTensor(r_batch) + GAMMA * q_next.max(1)[0]
        loss = self.loss_func(q_eval.squeeze(1), q_target)
        #pdb.set_trace()
        #print('loss: %f' % loss)

        # 计算, 更新 eval net
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()



action = None

def keyboard_hook(x):
    global action
    if (x.event_type != 'down' or x.name < '0' or x.name > '9'): return

    #print(x)
    action = int(x.name)

INPUT_FILE = '/Users/yao/develop/ARC/data/training/05f2a901.json'
with open(INPUT_FILE,'r') as f:
    puzzle = json.load(f)
puzzle_input = puzzle['train'][0]['input']
puzzle_output = puzzle['train'][0]['output']

env = gym.make('arc-v0', input=puzzle_input, output=puzzle_output)
print(env.observation_space.shape)
print(env.action_space.n)
keyboard.hook(keyboard_hook)




#pdb.set_trace()



# obs = env.reset()
# print(obs)

# while True:
#     env.render()
#     if (action != None):
#         print('action : ' + str(action))
#         obs, reward, done, info = env.step(action)
#         print(obs)
#         print(reward)
#         print(done)
#         action = None
#     time.sleep(0.1)


# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

dqn = DQN(env.observation_space.shape, env.action_space.n)
s = env.reset()
#pdb.set_trace()
# a = dqn.choose_action(s.flatten())

for i_episode in range(400):
    s = env.reset()
    while True:
        env.render()
        a = dqn.choose_action(s)
        #a = env.action_space.sample()
        #print('action %d' % a)

        s1, r, done, info = env.step(a)
        if (r > 0):
            print('reward %f' % r)

        dqn.store_transition(s, a, r, s1, done)

        if (len(dqn.memory) == MEMORY_CAPACITY):
            #print('learning')
            dqn.learn()

        if done:
            print('episode success!')
            break

        # 修改 reward, 使 DQN 快速学习
        # x, x_dot, theta, theta_dot = s_
        # r1 = (env.x_threshold - abs(x)) / env.x_threshold - 0.8
        # r2 = (env.theta_threshold_radians - abs(theta)) / env.theta_threshold_radians - 0.5
        # r = r1 + r2

        # 存记忆
        #dqn.store_transition(s, a, r, s1)

        #if dqn.memory_counter > MEMORY_CAPACITY:
        #    dqn.learn() # 记忆库满了就进行学习

        if done:    # 如果回合结束, 进入下回合
            break

        #s = s_
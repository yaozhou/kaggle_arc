import gym
from gym import error, spaces, utils
from gym.utils import seeding
import numpy as np
import game
import json

class MultiTaskARCEnv(gym.Env):
    metadata = {'render.modes': ['human']}  
    def __init__(self, **kwargs):
        self.json_file = kwargs['data']
        self.need_ui = kwargs['need_ui']
        self.engines = []

        with open(self.json_file,'r') as f:
            puzzles = json.load(f)

        for puzzle in puzzles['train']:
            engine = game.GameEngine(puzzle['input'], puzzle['output'], self.need_ui)
            self.engines.append(engine)

        self.action_space = spaces.Discrete(self.engines[0].action_n)
        self.observation_space = spaces.Box(low=0, high=255,
                                        shape=(game.MAX_SHAPLE_NUM * game.MAX_FEATURE_NUM, 1), 
                                        dtype=np.float)

    def step(self, action):
        obs_batch = reward_batch = done_batch = info_batch = []
        for a in action:
            for engine in self.engines:
                obs, reward, done, info = engine.do_action(a)
                obs_batch.append(obs)
                reward_batch.append(reward)
                done_batch.append(done)
                info_batch.append(info)

        return obs_batch, reward_batch, done_batch, info_batch

    def reset(self):
        obs_batch = []
        for engine in self.engines:
            obs.append(engine.reset())

        return obs_batch

    def render(self, mode='human', close=False):
        for engine in self.engines:
            self.engine.draw_game()
 
class ARCEnv(gym.Env):
    metadata = {'render.modes': ['human']}   
    def __init__(self, **kwargs):
        #print(kwargs)
        self.input = kwargs['input']
        self.output = kwargs['output'] 
        self.need_ui = kwargs['need_ui']
        self.reward = 0

        self.engine = game.GameEngine(self.input, self.output, self.need_ui)
        self.action_space = spaces.Discrete(self.engine.action_n)
        self.observation_space = spaces.Box(low=0, high=255,
                                        shape=(game.MAX_SHAPLE_NUM * game.MAX_FEATURE_NUM,), 
                                        dtype=np.float32)
 
    def step(self, action):
        obs, reward, done, info = self.engine.do_action(action)
        self.reward += reward
        #print(reward)
        #print(done)
        if (reward < 0):
            done = True

        if (done):
            print('episode finish reward = %f' % self.reward)
            obs = self.reset()

        return obs.flatten(), reward, done, info
 
    def reset(self):
        #print('game reset')
        self.reward = 0
        self.engine.reset()
        return self.engine.features.flatten()
 
    def render(self, mode='human', close=False):
        self.engine.draw_game()

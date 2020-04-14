import gym
from gym import error, spaces, utils
from gym.utils import seeding
import numpy as np
import gym_arc.envs.game as game
 
class ARCEnv(gym.Env):
    metadata = {'render.modes': ['human']}   
    def __init__(self, **kwargs):
        #print(kwargs)
        input = kwargs['input']
        output = kwargs['output'] 
        need_ui = kwargs['need_ui']
        self.engine = game.GameEngine(input, output, need_ui)
        self.action_space = spaces.Discrete(self.engine.action_n)
        self.observation_space = spaces.Box(low=0, high=255,
                                        shape=(game.MAX_SHAPLE_NUM, game.MAX_FEATURE_NUM, 1), 
                                        dtype=np.float)
 
    def step(self, action):
        # print('action : ' + str(action))
        obs, reward, done, info = self.engine.do_action(action)
        return obs, reward, done, info
 
    def reset(self):
        self.engine.reset()
        return self.engine.features
 
    def render(self, mode='human', close=False):
        # print('render')
        self.engine.draw_game()
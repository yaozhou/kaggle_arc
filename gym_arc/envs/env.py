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
        self.engine = game.GameEngine(input, output)
        self.action_space = spaces.Discrete(self.engine.action_n)
        self.observation_space = spaces.Box(low=0, high=255,
                                        shape=(self.engine.height, self.engine.width, 1), 
                                        dtype=np.uint8)
 
    def step(self, action):
        # print('action : ' + str(action))
        self.engine.do_action(action)
 
    def reset(self):
        pass
 
    def render(self, mode='human', close=False):
        # print('render')
        self.engine.draw_game()
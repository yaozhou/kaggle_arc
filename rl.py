import gym
import gym_arc
import pdb
import json
import time

INPUT_FILE = '/Users/yao/develop/ARC/data/training/05f2a901.json'
with open(INPUT_FILE,'r') as f:
    puzzle = json.load(f)
puzzle_input = puzzle['train'][0]['input']
puzzle_output = puzzle['train'][0]['output']

env = gym.make('arc-v0', input=puzzle_input, output=puzzle_output)

#pdb.set_trace()

while True:
    env.render()
    env.step(3)
    time.sleep(0.1)
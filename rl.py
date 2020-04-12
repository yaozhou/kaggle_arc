import gym
import gym_arc
import pdb
import json
import time
import keyboard

action = None

def keyboard_hook(x):
    global action
    if (x.event_type != 'down' or x.name < '0' or x.name > '9'): return

    print(x)
    action = int(x.name)

INPUT_FILE = '/Users/yao/develop/ARC/data/training/05f2a901.json'
with open(INPUT_FILE,'r') as f:
    puzzle = json.load(f)
puzzle_input = puzzle['train'][0]['input']
puzzle_output = puzzle['train'][0]['output']

env = gym.make('arc-v0', input=puzzle_input, output=puzzle_output)

keyboard.hook(keyboard_hook)

obs = env.reset()
print(obs)

while True:
    env.render()
    if (action != None):
        print('action : ' + str(action))
        obs = env.step(action)
        print(obs)
        action = None
    time.sleep(0.1)
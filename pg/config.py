#!/bin/python
import os
import torch

LR = 0.001
EPOCHS = 5000
batch_size = 1024 * 12
RENDER = False
MODEL = ''
#MODEL = './result/05f2a901_ 220.model'
INPUT_FILE = './data/08ed6ac7.json'
STEPS_LIMIT = 200
DECAY = 0.98
FILE_ID = os.path.basename(INPUT_FILE).split('.')[0]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
action_model = 'single'

from TetrisBattle.envs.tetris_env import TetrisSingleEnv
import argparse
import os
import shutil
from random import random, randint, sample

import numpy as np
import torch
import torch.nn as nn
from collections import deque
from dqn.deep_q_network import *

from Group37_model import *

env = TetrisSingleEnv(gridchoice="none", obs_type="grid", mode="human")
state = env.reset()
print(state)
done = False

from TetrisBattles.envs.tetris_env import TetrisSingleEnv as game
env1 = game(gridchoice="none", obs_type="grid", mode="human")
env1.reset(state)

model = torch.load("{}/tetris_10000".format(opt.saved_path), map_location=lambda storage, loc: storage)
model.eval()
player=env1.game_interface.tetris_list[env1.game_interface.now_player]
tetris = player['tetris']
tetris.reset(state)
com_event = player["com_event"]
next_steps = env1.game_interface.get_next_states()
next_actions, next_states = zip(*next_steps.items())
next_states = torch.stack(next_states)
predictions = model(next_states)[:, 0]
index = torch.argmax(predictions).item()
action = next_actions[index]

new_x, new_y = get_pos(tetris)
drop = False
for i in action:
    if i == 0:
        continue
    elif i == 1:
        #hold the block
        com_event.set([1])
        for evt in com_event.get():
            tetris.trigger(evt)
    elif i == 2:
        drop = True
        break
    elif i == 3:
        tetris.block.rotate()
    elif i == 4:
        continue
    elif i == 5:
        new_x += +1
    elif i == 6:
        new_x += -1
    else:
        assert False
tetris.px = new_x
state, reward, done, infos = env1.step(0)
if drop:
    state, reward, done, infos = env1.step(2)

time.sleep(500)
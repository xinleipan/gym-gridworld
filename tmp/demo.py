from gridworld import *

env = GridWorld(256, 'plan1.txt', True, True)

_ = env.reset()
for i in range(100):
    _,_,_,_ = env.step(i%5)


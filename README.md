# gym-gridworld

Basic implementation of gridworld game 
for reinforcement learning research. 

## Install gym-gridworld

    cd gym-gridworld
    pip install -e .

## Use gym-gridworld
    
    import gym
    import gym_gridworld
    env = gym.make('gridworld-v0')
    _ = env.reset()
    _ = env.step(env.action_space.sample())

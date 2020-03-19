# gym-gridworld

Basic implementation of gridworld game 
for reinforcement learning research. This environment is used in the following paper:

[How You Act Tells a Lot: Privacy-Leakage Attack on Deep Reinforcement Learning](https://arxiv.org/abs/1904.11082)

Consider cite the paper:

```
@inproceedings{pan2019you,
  author    = {Xinlei Pan and
               Weiyao Wang and
               Xiaoshuai Zhang and
               Bo Li and
               Jinfeng Yi and
               Dawn Song},
  title     = {How You Act Tells a Lot: Privacy-Leaking Attack on Deep Reinforcement
               Learning},
  booktitle = {Proceedings of the 18th International Conference on Autonomous Agents
               and MultiAgent Systems, {AAMAS}},
  pages     = {368--376},
  publisher = {International Foundation for Autonomous Agents and Multiagent Systems},
  year      = {2019},
}
```

## Install gym-gridworld

install virtual environment for gridworld

    cd gym-gridworld
    conda env create -f environment.yml
    conda activate gridworld
    pip install -e .

## Use gym-gridworld
    
    import gym
    import gym_gridworld
    env = gym.make('gridworld-v0')
    _ = env.reset()
    _ = env.step(env.action_space.sample())
    
## Visualize gym-gridworld
In order to visualize the gridworld, you need to set `env.verbose` to `True`

    env.verbose = True
    _ = env.reset()

from gym.envs.registration import register

register(
    id='gridworld-v0',
    entry_point='gym_gridworld.envs:GridworldEnv',
)
register(
    id='gridworld-v1',
    entry_point='gym_gridworld.envs:GridworldEnv1',
)
register(
    id='gridworld-v2',
    entry_point='gym_gridworld.envs:GridworldEnv2',
)

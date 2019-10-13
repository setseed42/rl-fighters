from gym.envs.registration import register

register(
    id='fighters-v0',
    entry_point='gym_fighters.envs:FightersEnv',
)

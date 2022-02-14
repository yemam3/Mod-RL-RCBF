from envs.unicycle_env import UnicycleEnv
from envs.simulated_cars_env import SimulatedCarsEnv

"""
This file includes a function that simply returns one of the supported environments. 
"""


def build_env(env_name, obs_config='default', rand_init=False):
    """Build our custom gym environment."""

    if env_name == 'Unicycle':
        return UnicycleEnv(obs_config, rand_init=rand_init)
    elif env_name == 'SimulatedCars':
        return SimulatedCarsEnv()
    else:
        raise Exception('Env {} not supported!'.format(env_name))

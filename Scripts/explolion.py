import gym
from colorama import Fore, Back

print(
    *list(
        gym.envs.registry.all()
    ), sep="\n"
)
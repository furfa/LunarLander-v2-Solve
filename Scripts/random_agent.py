import argparse
import sys

import gym
from gym import wrappers, logger
from GymRunner import GymRunner
import numpy as np

class RandomAgent(object):
    """The world's simplest agent!"""
    def __init__(self, action_space):
        self.action_space = action_space

    def act(self, observation, reward, done):
        return  np.random.randint(self.action_space)

# New comment
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('env_id', nargs='?', default='LunarLander-v2', help='Select the environment to run')
    args = parser.parse_args()

    # You can set the level to logger.DEBUG or logger.WARN if you
    # want to change the amount of output.
    logger.set_level(logger.INFO)
    
 
 
    GR = GymRunner("LunarLander-v2", '/tmp/random-agent-results') 

    agent = RandomAgent(GR.get_shapes()["action_space"])

    GR.test_agent(agent, 100)

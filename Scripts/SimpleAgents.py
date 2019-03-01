import argparse
import sys

import gym
from gym import wrappers, logger
from GymRunner import GymRunner
import numpy as np
from tensorflow import keras
from tensorflow.keras.layers import Dense, Activation, Flatten
import getkey
class RandomAgent(object):
    """The world's simplest agent!"""
    def __init__(self, action_space):
        self.action_space = action_space

    def act(self, observation, reward, done):
        return  np.random.randint(self.action_space)

class KeyboardAgent(object):
    """Keyboard control"""

    def __init__(self):
        print("Keyb agent created")
    
    def act(self, observation, reward, done):
        key = getkey.getkey()
        if key == getkey.keys.DOWN:
            return 2
        elif key == getkey.keys.LEFT:
            return 3
        elif key == getkey.keys.RIGHT:
            return 1
        return 0 # Ps ничего не делать 


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('env_id', nargs='?', default='LunarLander-v2', help='Select the environment to run')
    args = parser.parse_args()

    # You can set the level to logger.DEBUG or logger.WARN if you
    # want to change the amount of output.
    logger.set_level(logger.INFO)
    
 
 
    GR = GymRunner("LunarLander-v2", '/tmp/random-agent-results') 

    print(GR.get_shapes())
    # agent = RandomAgent(GR.get_shapes()["action_space"])
    agent = KeyboardAgent()


    GR.test_agent(agent, 100)

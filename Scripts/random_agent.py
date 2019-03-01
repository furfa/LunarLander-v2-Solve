import argparse
import sys

import gym
from gym import wrappers, logger
from GymRunner import GymRunner
import numpy as np
from tensorflow import keras
from tensorflow.keras.layers import Dense, Activation, Flatten
class RandomAgent(object):
    """The world's simplest agent!"""
    def __init__(self, action_space):
        self.action_space = action_space

    def act(self, observation, reward, done):
        return  np.random.randint(self.action_space)

class KerasModel(object):
    """The world's simplest agent!"""
    self.input_shape = (0,0)
    self.output_shape = (0)
    def __init__(self, action_space, model, input_shape, output_shape):
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.action_space = action_space
        
        model = keras.models.Sequential([
                Dense(32, input_shape= self.input_shape),
                Flatten(),
                Activation('relu'),
                Dense(self.output_shape),
                Activation('softmax'),
        ])
        print(model.summary())
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        self.model = model

    def fit(self, X, y):
        model.fit(X, y)
        return model

    def act(self, observation, reward, done):
        return np.argmax(self.model.predict([observation])[0])


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

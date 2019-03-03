import copy
import torch
from torch import nn, optim
from torch.nn import functional as F
import numpy as np
import gym
from gym import wrappers, logger

class Memory():
    def __init__(self, size, observation_size, reward_size):
        """
        size - int
        observation_size - int
        reward_size - int
        """
        self.observation_size = observation_size
        self.reward_size = reward_size
        self.index = 0
        self.size = size
        self.data = np.zeros( 
            (size, observation_size+reward_size+observation_size+1)
        )
        
    def append(self, obs, action, reward, next_obs):
        self.data[self.index % self.size] = np.hstack(
            (obs, action, reward, next_obs)
        )

        self.index += 1

    def get_observation(self):
        return self.data[:, 0:self.observation_size]

    def get_action(self):
        return self.data[:, self.observation_size : self.observation_size+1]
    def get_reward(self):
        return self.data[:, self.observation_size+1 : self.observation_size+1+self.reward_size]
    def get_next_observation(self):
        return self.data[:, self.observation_size+1+self.reward_size :] 
    def get_data(self):
        return self.data
    
class Agent(object):
    def __init__(self, MODEL, gamma, epsilon, alpha, 
                 maxMemorySize, epsEnd=0.05, 
                 replace=10000, n_actions=4):
        self.GAMMA = gamma
        self.EPSILON = epsilon
        self.EPS_END = epsEnd
        self.ALPHA = alpha
        self.n_actions = n_actions
        self.memSize = maxMemorySize
        self.steps = 0
        self.learn_step_counter = 0
        self.memory = Memory(self.memSize, observation_size = 8, reward_size = 1)
        self.replace_weight_freq = replace
        self.Q_nn_base = MODEL(alpha)
        self.Q_nn_copy = MODEL(alpha)
        
    def chooseAction(self, observation):
        rand = np.random.random()
        actions = self.Q_nn_base(observation)
        # print(self.EPSILON)
        if rand < 1 - self.EPSILON:
            action = torch.argmax(actions).item()
            print(f"ACTIONS = {action}")
        else:
            action = np.random.choice(range(self.n_actions))            
            print("Random!!")
        self.steps += 1
        return action
    
    def learn(self, batch_size):
        self.Q_nn_copy.optimizer.zero_grad()

        # print(self.learn_step_counter, self.replace_weight_freq)

        if self.learn_step_counter % self.replace_weight_freq == 0:
            # Replace every replace_weight_freq iter
            self.Q_nn_base.load_state_dict(self.Q_nn_copy.state_dict())

        memStart = int(np.random.choice(range(self.memory.size-batch_size-1)))

        miniBatch = copy.deepcopy( self.memory )

        miniBatch.data = miniBatch.data[memStart:memStart+batch_size] # Slice

        # convert to list because memory is an array of numpy objects

        Q_copy_predict = self.Q_nn_copy(
            miniBatch.get_observation()
            )

        Q_base_predict = self.Q_nn_base(
            miniBatch.get_next_observation()
            )
        

        rewards = torch.Tensor( miniBatch.get_reward() )
        y = rewards + self.GAMMA*torch.max( Q_base_predict, dim=1 )[0]
        
        if self.steps > 500:
            if self.EPSILON - 1e-5 > self.EPS_END:
                self.EPSILON -= 1e-5
            else:
                self.EPSILON = self.EPS_END

        loss = self.Q_nn_base.loss(y, torch.max(Q_copy_predict, dim=1)[0] ) 
        loss.backward()
        self.Q_nn_base.optimizer.step()
        self.learn_step_counter += 1

class GymRunner():

    ENV_NAME = ""
    env = None

    def __init__(self, env_name = "LunarLander-v2", outdir = '/tmp/RF-results'):
        self.ENV_NAME = env_name
        self.env = gym.make(self.ENV_NAME)
        self.env.seed(228)
        #self.env = wrappers.Monitor(self.env, directory=outdir, force=True)
        #self.env._max_episode_steps = 1200

    def get_shapes(self):
        return {
            "action_space" : self.env.action_space.n,
            "observation_space" : self.env.observation_space.shape,
        }

    def random_actions(self, agent):

        env = self.env

        while agent.memory.index < agent.memory.size:
            observation = env.reset()
            done = False
            while not done:
                action = env.action_space.sample()
                observation_, reward, done, info = env.step(action)

                agent.memory.append(observation, action, reward, observation_)
                observation = observation_

    def fit(self, agent, n_iters, batch_size=32,visualize=True):

        env = self.env
        scores = []
        # epsHistory = []

        for i in range(n_iters):
            # epsHistory.append(agent.EPSILON)        
            done = False
            observation = env.reset()
            score = 0
            while not done:
                if visualize:
                    env.render()
                action = agent.chooseAction( observation=observation)

                observation_, reward, done, info = env.step(action)
                score += reward

                agent.memory.append(observation, action, reward, observation_)

                observation = observation_            
                agent.learn(batch_size)

            scores.append(score)
            print('score:',score,'iter:',i)

    def test_agent(self, agent, n_iters):
        self.env = gym.make(self.ENV_NAME)
        
        mean_reward = []
        for iter in range(n_iters):
            done = False
            observation = self.env.reset()
            reward = 0
            info = {}

            sum_reward = 0
            while not done:
                self.env.render()

                action = agent.chooseAction(observation)

                observation, reward, done, info = self.env.step(action)

                sum_reward += reward
                print(f"REWARD = {reward}, SUM= {sum_reward}")
                mean_reward.append(reward)
        print('Mean_reward:',np.mean(mean_reward))
        self.env.close()

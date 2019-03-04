import copy
import torch
from torch import nn, optim
from torch.nn import functional as F
import numpy as np
import gym
from gym import wrappers, logger
from tqdm import tqdm
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
            (size, observation_size+reward_size+observation_size+1+1)
        )
        
    def append(self, obs, action, reward, next_obs, done):
        self.data[self.index % self.size] = np.hstack(
            (obs, action, reward, next_obs, done)
        )

        self.index += 1
        self.index %= self.size

    def get_observation(self):
        return self.data[:, 0:self.observation_size]

    def get_action(self):
        return self.data[:, self.observation_size : self.observation_size+1]
    def get_reward(self):
        return self.data[:, self.observation_size+1 : self.observation_size+1+self.reward_size]
    def get_next_observation(self):
        return self.data[:, self.observation_size+1+self.reward_size :self.observation_size+1+self.reward_size+self.observation_size] 
    def get_dones(self):
        return self.data[:, -1]
    def get_data(self):
        return self.data
    def get_size(self):
        return self.data.shape[0] 

class Agent(object):
    """
    Q_nn_base - закрепленная нейронка

    """
    def __init__(self, MODEL, gamma, epsilon, alpha, 
                 maxMemorySize, epsEnd=0.05, eps_minimize_per_iter = 1e-5,
                 n_actions=4, tau=1e-3):
        self.GAMMA = gamma
        self.EPSILON = epsilon
        self.EPS_END = epsEnd
        self.EPS_MINIMIZE = eps_minimize_per_iter
        self.ALPHA = alpha
        self.n_actions = n_actions
        self.memSize = maxMemorySize
        self.TAU = tau
        self.learn_step_counter = 0
        self.memory = Memory(self.memSize, observation_size = 8, reward_size = 1)

        self.Q_nn_base = MODEL(alpha)
        self.Q_nn_copy = MODEL(alpha)
        
    def chooseAction(self, observation):
        rand = np.random.random()

        self.Q_nn_copy.eval()
        with torch.no_grad():
            actions = self.Q_nn_copy(observation)

        if rand < 1 - self.EPSILON:
            action = torch.argmax(actions).item()
            # print(f"ACTIONS = {action}")
        else:
            action = np.random.choice(range(self.n_actions))            
            # print("Random!!")

        self.EPSILON = max(self.EPS_END, self.EPSILON-self.EPS_MINIMIZE)
        return action
    
    def learn(self, batch_size):

        miniBatch = copy.deepcopy( self.memory )
        if True: # Как выделять бач
            selected_rows = np.random.choice( 
                np.arange( self.memory.get_size() ), batch_size, replace=False
            ) # Рандомные строки в количесве batch size
            miniBatch.data = miniBatch.data[selected_rows]
        else:
            memStart = int(np.random.choice(range(self.memory.size-batch_size-1)))
            miniBatch.data = miniBatch.data[memStart:memStart+batch_size] # Slice
        
        dones = torch.Tensor( miniBatch.get_dones() ).unsqueeze(1)
        rewards = torch.Tensor( miniBatch.get_reward() )
        actions = torch.Tensor( miniBatch.get_action() ).long()
        

        self.Q_nn_base.train()
        self.Q_nn_copy.train()

        Q_base_predict = self.Q_nn_base(
            miniBatch.get_next_observation()
            )

        y = rewards + self.GAMMA*torch.max( Q_base_predict, dim=1 )[0].unsqueeze(1) * (1-dones)

        Q_copy_predict = self.Q_nn_copy(
            miniBatch.get_observation()
            )

        self.Q_nn_copy.optimizer.zero_grad()
        loss = self.Q_nn_copy.loss(
            Q_copy_predict.gather(1, actions), 
            y
            ) #Выбирает ревард того действия, которое было сделано в истории

        loss.backward()
        self.Q_nn_copy.optimizer.step()

        
        # if self.learn_step_counter % self.replace_weight_freq == 0:
        #     # Replace every replace_weight_freq iter
        #     self.Q_nn_base.load_state_dict(self.Q_nn_copy.state_dict())       
        self.update_weight(self.Q_nn_copy, self.Q_nn_base, self.TAU)

    def update_weight(self, model_from, model_to, tau):
        for to_p, from_p in zip(model_from.parameters(), model_to.parameters()):
            to_p.data.copy_(tau*from_p.data + (1.0-tau)*to_p.data)


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

        for i in tqdm( range( agent.memory.get_size() ) ):
            observation = env.reset()
            done = False
            while not done:
                action = env.action_space.sample()
                observation_, reward, done, info = env.step(action)

                agent.memory.append(observation, action, reward, observation_, done)
                observation = observation_
    def fit(self, agent, n_iters, batch_size=32, LEARN_FREQ=1, visualize=True):

        env = self.env
        scores = []
        # epsHistory = []

        for iteration in tqdm( range(n_iters) ):
            # epsHistory.append(agent.EPSILON)        
            done = False
            observation = env.reset()
            score = 0
            i = 0
            while not done:
                if visualize:
                    env.render()
                action = agent.chooseAction( observation=observation)

                observation_, reward, done, info = env.step(action)
                score += reward

                agent.memory.append(observation, action, reward, observation_, done)

                observation = observation_            
                if i % LEARN_FREQ == 0:
                    agent.learn(batch_size)

                i += 1

            scores.append(score)
            print('score:',score,'iter:',iteration)

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
            print(f"Game_reward= {sum_reward}")
            mean_reward.append(sum_reward)
        print('Mean_reward:',np.mean(mean_reward))
        self.env.close()


import copy
import torch
from torch import nn, optim
from torch.nn import functional as F
import numpy as np
import gym
from gym import wrappers, logger
from tqdm import tqdm
import random
from collections import namedtuple, deque


class MemoryNumpy(object):
    def __init__(
        self, 
        size, 
        observation_size,
        reward_size,
        action_size,
        seed):
        """
        size - int
        observation_size - int
        reward_size - int
        """
        self.observation_size = observation_size
        self.reward_size = reward_size
        self.index = 0
        self.data = np.zeros( 
            (size, observation_size+reward_size+observation_size+1+1)
        )
        self.size = size 
    def append(self, obs, action, reward, next_obs, done):
        if all( [i is not None for i in (obs, action, reward, next_obs, done)] ):
            self.data[self.index % self.size] = np.hstack(
                (obs, action, reward, next_obs, done)
            )

            self.index += 1
            self.index %= self.size
    
    def sample(self, batch_size):
        miniBatch = copy.deepcopy(self) 
        selected_rows = np.random.choice(np.arange( len(self) ), size=batch_size, replace=False) 
        miniBatch.data = self.data[selected_rows]

        dones = torch.from_numpy( miniBatch.get_dones() ).float().unsqueeze(1) # Fix shape
        rewards = torch.from_numpy( miniBatch.get_reward() ).float()
        actions = torch.from_numpy( miniBatch.get_action() ).long()
        observation = torch.from_numpy( miniBatch.get_observation() ).float()
        next_observation = torch.from_numpy( miniBatch.get_next_observation() ).float()
                    
        return (observation, actions, rewards, next_observation, dones) 

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
    def __len__(self):
        return self.data.shape[0] 
    def get_size(self):
        return len(self)

class ReplayBuffer(object):
    """Fixed-size buffer to store experience tuples."""

    def __init__(
        self, 
        size, 
        observation_size,
        reward_size,
        action_size, 
        seed):
        """Initialize a ReplayBuffer object.

        Params
        ======
            action_size (int): dimension of each action
            size (int): maximum size of buffer
            batch_size (int): size of each training batch
            seed (int): random seed
        """
        self.max_size = size
        self.action_size = action_size
        self.data = deque(maxlen=size)  
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.seed = random.seed(seed)
    
    def append(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        e = self.experience(state, action, reward, next_state, done)
        self.data.append(e)
    
    def sample(self, batch_size):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.data, batch_size)

        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float()
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long()
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float()
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float()
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float()
  
        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.data)
    def get_size(self):
        return self.max_size
class Agent(object):
    """
    Класс реализующий DQN агента
    """ 
    def __init__(self, MODEL, MemClass, 
                 gamma=0.99, 
                 epsilon=1, 
                 alpha=1e-3, 
                 maxMemorySize=1500, 
                 eps_end=0.05, 
                 eps_delta=0.9,
                 eps_how_minimize="multiplic",
                 tau=1e-3,
                 action_space=4, 
                 observation_space=8,
                 ):
        """
        epsilon - коеф рандома
        epsEnd - минимальный epsilon
        eps_delta - коеф уменьшения epsilonа
        action_space - количество доступных действий в среде
        observation_space - размер observation
        tau - коеф обновления весов закрепленной нейронки
        alpha - learning-rate
        ----------------------------------
        Q_nn_base - закрепленная нейронка
        Q_nn_copy - основная
        """
        self.GAMMA = gamma
        self.EPSILON = epsilon
        self.EPS_END = eps_end
        self.EPS_MINIMIZE = eps_delta
        self.eps_how_minimize = eps_how_minimize
        self.ALPHA = alpha
        self.action_space = action_space
        self.observation_space = observation_space
        self.memSize = maxMemorySize
        self.TAU = tau
        self.learn_step_counter = 0
        self.memory = MemClass(self.memSize, observation_space, 1, 1, 228) 


        self.Q_nn_copy = MODEL(alpha, observation_space, action_space)
        self.Q_nn_base = MODEL(alpha, observation_space, action_space)
        self.Q_nn_base.load_state_dict( self.Q_nn_copy.state_dict() )
        self.Q_nn_base.eval()
        
    def chooseAction(self, observation):
        rand = np.random.random()

        self.Q_nn_copy.eval()
        with torch.no_grad():
            actions = self.Q_nn_copy(observation)

        if rand < 1 - self.EPSILON:
            action = torch.argmax(actions).item()
            # print(f"ACTIONS = {action}")
        else:
            action = np.random.choice(range(self.action_space))            
            # print("Random!!")
        if (self.eps_how_minimize == "minus"):
            self.EPSILON = max(self.EPS_END, self.EPSILON-self.EPS_MINIMIZE)
        elif (self.eps_how_minimize == "multiplic"):
            self.EPSILON = max(self.EPS_END, self.EPSILON*self.EPS_MINIMIZE)
        return action
    
    def learn(self, batch_size):

        observation, actions, rewards, next_observation, dones = self.memory.sample(batch_size)
        
        Q_base_predict = self.Q_nn_base(
            next_observation
            ).detach().max(1)[0].unsqueeze(1) # Максималная награда за действие

        y = rewards + ( self.GAMMA * Q_base_predict * (1-dones) )

        Q_copy_predict = self.Q_nn_copy(
            observation
            ).gather(1, actions) #Выбирает ревард того действия, которое было сделано в истории

        self.Q_nn_copy.optimizer.zero_grad()
        loss = self.Q_nn_copy.loss(
            Q_copy_predict,
            y
            ) 

        loss.backward()
        self.Q_nn_copy.optimizer.step()

        self.update_weight(self.Q_nn_copy, self.Q_nn_base, self.TAU)

    def update_weight(self, model_from, model_to, tau):
        for from_p, to_p in zip(model_from.parameters(), model_to.parameters()):
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
    def fit(self, agent, n_iters, max_iters=2000, batch_size=32, LEARN_FREQ=1, visualize=True):

        env = self.env
        scores = list()
        # epsHistory = []

        pbar = tqdm( range(n_iters) )
        for iteration_num in pbar:
            # epsHistory.append(agent.EPSILON)        
            done = False
            observation = env.reset()
            score = 0
            for i in range(max_iters):
                if visualize:
                    env.render()
                action = agent.chooseAction( observation=observation)

                observation_, reward, done, info = env.step(action)
                score += reward

                agent.memory.append(observation, action, reward, observation_, done)

                observation = observation_            
                if i % LEARN_FREQ == 0:
                    agent.learn(batch_size)
                if done:
                    break

            scores.append(score)
            pbar.set_description(f"[SCORE] {score:.2f}")
            if iteration_num % 100 == 0 and iteration_num!=0:
                scores = scores[-100:]
                print(
                    f"\n[MEAN_SCORE] {np.mean(scores):.2f} [STD_SCORE] {np.std(scores):.2f} \n", 
                    end=""
                )

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


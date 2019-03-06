import random
from collections import namedtuple, deque
import numpy as np
import copy
import torch
from torch import nn, optim
from torch.nn import functional as F


class MemoryNumpy(object):
    def __init__(
        self, 
        size, 
        observation_size,
        reward_size,
        action_size,
        seed):
        """
        size(int)
        observation_size(int)
        reward_size(int)
        """
        self.observation_size = observation_size
        self.reward_size = reward_size
        self.index = 0
        self.data = np.zeros( 
            (size, observation_size+reward_size+observation_size+1+1)
        )
        self.size = size 
        self.is_fill = False
    def append(self, obs, action, reward, next_obs, done):
        if all( [i is not None for i in (obs, action, reward, next_obs, done)] ):
            self.data[self.index % self.size] = np.hstack(
                (obs, action, reward, next_obs, done)
            )

            self.index += 1
            if self.index == self.size:
                self.is_fill = True
            self.index %= self.size
    
    def sample(self, batch_size):
        miniBatch = copy.deepcopy(self) 
        
        index_space = len(self) if (self.is_fill) else (self.index)

        if index_space < batch_size:
            raise IndexError("memory length less then batch_size")

        selected_rows = np.random.choice(np.arange( index_space ), size=batch_size, replace=False) 
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

class MemoryDeque(object):
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
import torch
from torch import nn, optim
from torch.nn import functional as F
import numpy as np

"""
    PS : {'action_space': 4, 'observation_space': (8,)}
"""


class SimpleNet(nn.Module):
    def __init__(self, ALPHA):
        super().__init__()

        self.ALPHA = ALPHA

        INPUT_SHAPE = 8
        OUTPUT_SHAPE = 4

        self.model = nn.Sequential(
            nn.Linear(INPUT_SHAPE, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, OUTPUT_SHAPE),

        )

        self.optimizer = optim.Adam(self.parameters(), lr=self.ALPHA) # Tune this

        self.loss = nn.MSELoss()
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')     
        self.to(self.device)
    
    def forward(self, observation):
        observation = torch.Tensor(observation).to(self.device)
        actions = self.model(observation)
        return actions


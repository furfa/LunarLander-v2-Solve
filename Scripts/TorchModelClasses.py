import torch
from torch import nn, optim
from torch.nn import functional as F
import numpy as np

"""
    PS : {'action_space': 4, 'observation_space': (8,)}
"""


class SimpleNet(nn.Module):
    def __init__(self, ALPHA, INPUT_SHAPE, OUTPUT_SHAPE):
        super().__init__()

        self.ALPHA = ALPHA

        self.model = nn.Sequential(
            nn.Linear(INPUT_SHAPE, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, OUTPUT_SHAPE),
        )

        self.optimizer = optim.Adam(self.parameters(), lr=self.ALPHA) # Tune this

        self.loss = nn.MSELoss()
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')     
        self.to(self.device)

    def forward(self, observation):
        observation = torch.Tensor(observation).to(self.device)
        actions = self.model(observation)
        return actions


class HuberNet(nn.Module):
    def __init__(self, ALPHA, INPUT_SHAPE, OUTPUT_SHAPE):
        super().__init__()

        self.ALPHA = ALPHA

        self.model = nn.Sequential(
            nn.Linear(INPUT_SHAPE, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, OUTPUT_SHAPE),
        )

        self.optimizer = optim.RMSprop(self.parameters(), lr=self.ALPHA, momentum=0.0001) # Tune this

        self.loss = nn.SmoothL1Loss()
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')     
        self.to(self.device)

    def forward(self, observation):
        observation = torch.Tensor(observation).to(self.device)
        actions = self.model(observation)
        return actions


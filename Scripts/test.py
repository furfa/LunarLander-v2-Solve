from BaseClasses import *
import TorchModels
import numpy as np
#from agent_torch import *
"""
    Model init 
"""

agent = Agent(TorchModels.SimpleNet, gamma=0.99,     epsilon=1,
                  alpha=0.01, maxMemorySize=10000,
                  replace=None)
gR = GymRunner()



gR.random_actions(agent)

env = gR.env
scores = []
batch_size = 5
# epsHistory = []



gR.fit(agent, 100,visualize=False)


gR.test_agent(agent,n_iters=10)


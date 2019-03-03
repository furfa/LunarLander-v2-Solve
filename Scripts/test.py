from BaseClasses import *
import TorchModels
import numpy as np
#from agent_torch import *
"""
    Model init 
"""

agent = Agent(TorchModels.SimpleNet, gamma=0.99,     epsilon=0.5,
                  alpha=0.01, maxMemorySize=5000,
                  replace=1000)
gR = GymRunner()



gR.random_actions(agent)


gR.fit(agent, 100,visualize=True)


gR.test_agent(agent,n_iters=10)


from BaseClasses import *
import TorchModels
import numpy as np
#from agent_torch import *
"""
    Model init 
"""

agent = Agent(TorchModels.SimpleNet, gamma=0.99, epsilon=0.8,
                epsEnd=0.09, eps_minimize_per_iter = 1e-3,
                alpha=1e-2, maxMemorySize=15000,
                replace=1000)
gR = GymRunner()



gR.random_actions(agent)


gR.fit(
    agent, 
    n_iters = 100,
    batch_size=20,
    visualize=True
)


gR.test_agent(agent,n_iters=10)


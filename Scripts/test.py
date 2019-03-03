from BaseClasses import *
import TorchModels
import numpy as np
#from agent_torch import *
"""
    Model init 
"""

<<<<<<< HEAD
agent = Agent(TorchModels.SimpleNet, gamma=0.95, epsilon=0.5,
                  alpha=0.1, maxMemorySize=1000,
=======
agent = Agent(TorchModels.SimpleNet, gamma=0.99,     epsilon=1,
                  alpha=0.01, maxMemorySize=10000,
>>>>>>> b62f915d006a0d128b528eb13683d9269e905be9
                  replace=None)
gR = GymRunner()



gR.random_actions(agent)

env = gR.env
scores = []
batch_size = 5
# epsHistory = []



gR.fit(agent, 100,visualize=False)


gR.test_agent(agent,n_iters=10)


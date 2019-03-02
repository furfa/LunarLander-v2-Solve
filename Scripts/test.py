from BaseClasses import *
import TorchModels
import numpy as np
#from agent_torch import *
"""
    Model init 
"""

agent = Agent(TorchModels.SimpleNet, gamma=0.95, epsilon=1.0,
                  alpha=0.003, maxMemorySize=5000,
                  replace=None)
gR = GymRunner()

"""
    Случайные действия для начала
    Внутри они выполняются пока не закончится память, выделенная под агента
"""

gR.random_actions(agent)

env = gR.env
scores = []
batch_size = 5
# epsHistory = []



gR.fit(agent, 200)

gR.test_agent(agent,n_iters=15)

# optim - sgd, mean_score = -145.956
# optim - Adam, mean_score = -161.79
# optim - Ada, mean_score =  - 146.63

from BaseClasses import *
import TorchModels
import numpy as np
#from agent_torch import *
"""
    Model init 
"""

agent = Agent(TorchModels.SimpleNet, gamma=0.99, epsilon=0.99,
                epsEnd=0.05, eps_minimize_per_iter = 1e-4,
                alpha=1e-3, maxMemorySize=1500,
                tau=5e-4
                )
gR = GymRunner()

print(11)

gR.random_actions(agent)
print("Заполнение памяти случайными действиями завершено")

gR.fit(
    agent, 
    n_iters = 10000,
    batch_size=32,
    LEARN_FREQ=20,
    visualize=True
)


gR.test_agent(agent,n_iters=10)


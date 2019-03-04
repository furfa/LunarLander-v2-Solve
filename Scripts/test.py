from BaseClasses import *
import TorchModels
import numpy as np
#from agent_torch import *
"""
TODO:
    - Сделать инициализацию памяти на лету (без random actions), т.к заполнение памяти идет в зачет итераций
      при этом модель не обучается
"""

agent = Agent(TorchModels.SimpleNet, gamma=0.99, epsilon=0.99,
                epsEnd=0.05, eps_delta=1e-4,
                alpha=1e-3, maxMemorySize=1500,
                tau=5e-4
                )
gR = GymRunner()

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


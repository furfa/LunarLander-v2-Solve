from MemoryClasses import *
from AgentClasses import *
from GymRunner import GymRunner
import TorchModelClasses as models
import numpy as np
#from agent_torch import *
"""
TODO:
    - Сделать инициализацию памяти на лету (без random actions), т.к заполнение памяти идет в зачет итераций
      при этом модель не обучается
"""

agent = DQN_agent(
        models.SimpleNet, 
        MemoryNumpy,
        gamma=0.99, 
        epsilon=0.99, eps_end=0.01, eps_delta=0.995,
        alpha=5e-3, 
        maxMemorySize=2500,
        tau=1e-3,
        action_space=4,
        observation_space=8,
        )
gR = GymRunner(
    env_name="LunarLander-v2" #"CartPole-v1"
)
print(gR.get_shapes())

gR.random_actions(agent)
print("Заполнение памяти случайными действиями завершено")

gR.fit(
    agent, 
    n_iters = 5000,
    batch_size=64,
    LEARN_FREQ=1,
    visualize=True
)


gR.test_agent(agent,n_iters=100)


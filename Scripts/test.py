from BaseClasses import *
import TorchModels
import numpy as np
#from agent_torch import *
"""
TODO:
    - Сделать инициализацию памяти на лету (без random actions), т.к заполнение памяти идет в зачет итераций
      при этом модель не обучается
"""

agent = Agent(
        TorchModels.SimpleNet, 
        MemoryNumpy,
        gamma=0.99, 
        epsilon=0.99, eps_end=0.01, eps_delta=0.995,
        alpha=5e-3, 
        maxMemorySize=2500,
        tau=1e-3,
        action_space=2,
        observation_space=4,
        )
gR = GymRunner(env_name="CartPole-v1")
print(gR.get_shapes())

gR.random_actions(agent)
print("Заполнение памяти случайными действиями завершено")

gR.fit(
    agent, 
    n_iters = 5000,
    batch_size=64,
    LEARN_FREQ=1,
    visualize=False
)


gR.test_agent(agent,n_iters=100)


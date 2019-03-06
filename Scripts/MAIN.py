from MemoryClasses import *
from AgentClasses import *
from GymRunner import GymRunner
import TorchModelClasses as models
import numpy as np
"""
TODO:
    - Сделать инициализацию памяти на лету (без random actions), т.к заполнение памяти идет в зачет итераций
      при этом модель не обучается
"""

ENV_NAME = "LunarLander-v2"
OBS_SPACE = 8
ACTION_SPACE = 4
 
AGENT = DQN_agent
MEMORY = MemoryNumpy
MODEL = models.SimpleNet

def kostil(reward):
    return (
        reward == 100 or 
        reward == -100 or 
        reward == 10 or 
        reward == 200
        )

def main():
    agent = AGENT(
        MODEL, 
        MEMORY,
        gamma=0.99,

        epsilon=0.99, 
        eps_end=0.01, 
        eps_delta=0.95,

        alpha=1e-3, 
        maxMemorySize=15000,
        tau=1e-3,
        action_space=ACTION_SPACE,
        observation_space=OBS_SPACE,
    )
    gR = GymRunner(
            env_name=ENV_NAME,
            behavior_func=kostil
    )

    gR.random_actions(agent, 64)
    print("Заполнение памяти случайными действиями завершено")

    gR.fit(
        agent, 
        n_iters = 5000,
        batch_size=64,
        LEARN_FREQ=4,
        visualize=False
    )


    gR.test_agent(agent,n_iters=10, render=False)

    print("Готово!")

if __name__ == "__main__":
    main()
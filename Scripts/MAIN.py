from MemoryClasses import *
from AgentClasses import *
from GymRunner import GymRunner
import TorchModelClasses as models
import numpy as np
"""
TODO:
    - Отключать обучение при нескольких последовательных rewardax более 200
    - Явное переобучение на эпизодах-аутлиерах
"""

ENV_NAME = "LunarLander-v2"
OBS_SPACE = 8
ACTION_SPACE = 4
 
AGENT = DQN_agent
MEMORY = MemoryNumpy
MODEL = models.SimpleNet
SEED = 228

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
        eps_delta=0.9,

        alpha=5e-4, 
        maxMemorySize=15000,
        tau=1e-3,
        action_space=ACTION_SPACE,
        observation_space=OBS_SPACE,
        seed=SEED
    )
    gR = GymRunner(
            env_name=ENV_NAME,
            behavior_func=kostil,
            seed=SEED
    )

    gR.random_actions(agent, 128)
    print("Заполнение памяти случайными действиями завершено")

    gR.fit(
        agent, 
        n_iters = 5000,
        batch_size=128,
        LEARN_FREQ=8,
        visualize=False
    )


    gR.test_agent(
        agent,
        n_iters=10,
        render=False,
        save_video=True,
        save_model=True)

    print("Готово!")

if __name__ == "__main__":
    main()

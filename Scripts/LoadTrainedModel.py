from MemoryClasses import *
from AgentClasses import *
from GymRunner import GymRunner
import numpy as np
import pickle
import sys

ENV_NAME = "LunarLander-v2"
OBS_SPACE = 8
ACTION_SPACE = 4
 
SEED = 1337

def kostil(reward):
    return (
        reward == 100 or 
        reward == -100 or 
        reward == 10 or 
        reward == 200
        )

def main(path):
    agent = pickle.load(
        open(path, "rb")
    )
    gR = GymRunner(
            env_name=ENV_NAME,
            behavior_func=kostil,
            seed=SEED
    )

    gR.test_agent(
        agent,
        n_iters=100,
        render=True,
        save_video=False,
        save_model=False)

    print("Готово!")

if __name__ == "__main__":
    try:
        path_to_pickle = sys.argv[1]
        main(path_to_pickle)
    except:
        print("No PATH TO AGENT")
        
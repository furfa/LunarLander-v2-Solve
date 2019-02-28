import gym
from gym import wrappers, logger

class GymRunner():

    ENV_NAME = ""
    env = None

    def __init__(self, env_name = "LunarLander-v2", outdir = '/tmp/RF-results'):
        self.ENV_NAME = env_name
        self.env = gym.make(self.ENV_NAME)
        #self.env = wrappers.Monitor(self.env, directory=outdir, force=True)
        #self.env._max_episode_steps = 1200

    def get_shapes(self):
        return {
            "action_space" : self.env.action_space.n,
            "observation_space" : self.env.observation_space.shape,
        }

    def test_agent(self, agent, n_iters):
        self.env = gym.make(self.ENV_NAME)
        

        for _ in range(n_iters):
            done = False
            observation = self.env.reset()
            reward = 0
            info = {}

            while not done:
                self.env.render()

                action = agent.act(observation, reward, done)

                observation, reward, done, info = self.env.step(action)
                print(f"REWARD = {reward}")
        self.env.close()
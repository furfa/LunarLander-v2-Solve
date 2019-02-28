import gym

class GymRunner():

    env = None

    def __init__(env_name = "LunarLander-v2"):
        self.env = gym.make(env_name)
        #self.env._max_episode_steps = 1200

    def get_shapes():
        return {
            "action_space" : self.env.action_space.n,
            "observation_space" : self.env.observation_space.shape,
        }

    def test_agent(agent, n_iters):
        done = False
        observation = self.env.reset()
        reward = 0
        info = {}

        for _ in range(n_iters):
            while not done:
                self.env.render()

                action = agent.act(observation, reward, done)

                observation, reward, done, info = self.env.step(action)

                print(f"REWARD = {reward}")
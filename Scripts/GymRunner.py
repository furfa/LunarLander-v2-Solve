import numpy as np
import numpy as np
import gym
from gym import wrappers, logger
from tqdm import tqdm

class GymRunner():
    ENV_NAME = ""
    env = None

    def __init__(self, env_name = "LunarLander-v2", outdir = '/tmp/RF-results'):
        self.ENV_NAME = env_name
        self.env = gym.make(self.ENV_NAME)
        self.env.seed(228)
        #self.env = wrappers.Monitor(self.env, directory=outdir, force=True)
        #self.env._max_episode_steps = 1200

    def get_shapes(self):
        return {
            "action_space" : self.env.action_space.n,
            "observation_space" : self.env.observation_space.shape,
        }

    def random_actions(self, agent, n_iters = "full"):
        """
        Случайные действия дли инициализации памяти
        -------------------------------------------------------------------------
        agent (Agent) - объект Агента для заполнения
        n_iters (int, str) - количество заполениния, "full" для заполнения всей памяти
        """
        env = self.env
        
        if n_iters == "full":
            n_iters = agent.memory.get_size() 

        for i in tqdm( range(n_iters) ):
            observation = env.reset()
            done = False
            while not done:
                action = env.action_space.sample()
                next_observation, reward, done, info = env.step(action)

                agent.memory.append(observation, action, reward, next_observation, done)
                observation = next_observation

    def fit(self, agent, n_iters, max_iters=2000, batch_size=32, LEARN_FREQ=1, visualize=True):
        env = self.env
        scores = list()
        pbar = tqdm( range(n_iters) )

        def main_loop(env, scores, pbar, visualize):

            for iteration_num in pbar:
                done = False
                observation = env.reset()
                score = 0
                for i in range(max_iters):
                    if visualize:
                        env.render()
                    action = agent.chooseAction( observation=observation)

                    next_observation, reward, done, info = env.step(action)
                    score += reward

                    agent.memory.append(observation, action, reward, next_observation, done)

                    observation = next_observation            
                    if i % LEARN_FREQ == 0:
                        agent.learn(batch_size)
                    if done:
                        break

                scores.append(score)
                pbar.set_description(f"[SCORE] {score:.2f}")
                if iteration_num % 100 == 0 and iteration_num!=0:
                    scores = scores[-100:]
                    print(
                        f"\n[MEAN_SCORE] {np.mean(scores):.2f} [STD_SCORE] {np.std(scores):.2f} \n", 
                        end=""
                    )
                    
        def try_block(env, scores, pbar, visualize):
            try:
                main_loop(env, scores, pbar, visualize)
            except KeyboardInterrupt:
                inp = input("""
                    o - Остановка обучения,
                    +v - Включить визуализацию
                    -v - Выключить визуализацию
                """)
                if inp == 'o':
                    print("Остановка обучения.")
                elif inp == '+v':
                    try_block(env, scores, pbar, True)
                elif inp == '-v':
                    try_block(env, scores, pbar, False)

        try_block(env, scores, pbar, visualize)

    def test_agent(self, agent, n_iters):
        self.env = gym.make(self.ENV_NAME)
        
        mean_reward = []
        for iter in range(n_iters):
            done = False
            observation = self.env.reset()
            reward = 0
            info = {}

            sum_reward = 0
            while not done:
                self.env.render()

                action = agent.chooseAction(observation)

                observation, reward, done, info = self.env.step(action)

                sum_reward += reward
            print(f"Game_reward= {sum_reward}")
            mean_reward.append(sum_reward)
        print('Mean_reward:',np.mean(mean_reward))
        self.env.close()
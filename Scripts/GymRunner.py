import numpy as np
import gym
from gym import wrappers, logger
from tqdm import tqdm
import os
from datetime import datetime
import copy
import pickle

class GymRunner():
    """
    TODO:
        self.LEARNING_EPISODE_COUNTER не работает
        добавить сохрание модели
    """

    ENV_NAME = ""
    env = None

    def __init__(self, env_name = "LunarLander-v2", behavior_func = lambda reward : False, seed=1):
        self.ENV_NAME = env_name
        self.env = gym.make(self.ENV_NAME)
        np.random.seed(seed)
        self.env.seed(seed)
        self.additional_behavior = behavior_func


        self.LEARNING_EPISODE_COUNTER = 0
        self.env_cop = copy.deepcopy(self.env)

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
                self.LEARNING_EPISODE_COUNTER += 1 # СЧЕТЧИК ЭПИЗОДОВ ОБУЧЕНИЯ

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
                self.LEARNING_EPISODE_COUNTER += 1 # СЧЕТЧИК ЭПИЗОДОВ ОБУЧЕНИЯ
                
                done = False
                observation = env.reset()
                score = 0
                reward = None

                for i in range(max_iters):
                    if visualize:
                        env.render()
                    
                    if self.additional_behavior(reward): # Поведение при касании
                        while not done:
                            next_observation, reward, done, info = env.step(0)
                        continue
                    else:
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
                mean_score = np.mean(scores[-100:])
                pbar.set_description(f"[S] {score:.2f} [MS] {mean_score:.2f}")

                if mean_score >= 200:
                    print("Early stopping")
                    return

                if iteration_num % 100 == 0 and iteration_num!=0:
                    scores = scores[-100:]
                    print("-" * 20)
                    print(
                        f"[MEAN_SCORE] {np.mean(scores):.2f} [STD_SCORE] {np.std(scores):.2f} \n", 
                        end=""
                    )
                    print("-" * 20)

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
                else:
                    print("Продолжаем :)")
                    try_block(env, scores, pbar, visualize)

        try_block(env, scores, pbar, visualize)
        pbar.close()

    def test_agent(self, agent, n_iters=10, render=True, save_video=True, video_path="../Results/", save_model=True):
        base_path = video_path
        tmp_path = os.path.join(base_path, "TMP")

        env = self.env_cop

        if save_video:
            env = wrappers.Monitor(env, tmp_path, force=True)
        
        mean_reward = []
        for iter in range(n_iters):
            done = False
            observation = env.reset()
            reward = 0
            info = {}

            sum_reward = 0

            only_passive_action = False

            while not done:
                if render:
                    env.render()

                action = agent.chooseAction(observation)

                if self.additional_behavior(reward): # Поведение при касании
                    only_passive_action = True    
                
                if only_passive_action:
                    action = 0
                else:
                    action = agent.chooseAction( observation=observation)

                observation, reward, done, info = env.step(action)

                sum_reward += reward

            print(f"Game_reward= {sum_reward}")

            mean_reward.append(sum_reward)
        print('Mean_reward:',np.mean(mean_reward))


        # название директории для сохранения
        date = datetime.now()
        new_dir_name = f"{np.mean(mean_reward):.2f} | {self.LEARNING_EPISODE_COUNTER} | {date.day}.{date.month}.{date.year} | {date.hour}:{date.minute}"
        new_dir_path = os.path.join(base_path, new_dir_name)
        if save_video:
            os.replace(tmp_path, new_dir_path)
        if save_model:
            path_to_model_file = os.path.join( new_dir_path, "model.pkl" ) 
            open(path_to_model_file, 'w').close() # touch analog
            with open(path_to_model_file , "wb" ) as output:
                pickle.dump(agent, output, pickle.HIGHEST_PROTOCOL)
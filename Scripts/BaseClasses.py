import torch
import torch as T # Чтобы копипаста работала
from torch import nn, optim
from torch.nn import functional as F
import numpy as np
import gym
from gym import wrappers, logger
class Agent(object):
    def __init__(self, MODEL, gamma, epsilon, alpha, 
                 maxMemorySize, epsEnd=0.05, 
                 replace=10000, actionSpace=[0,1,2,3]):
        self.GAMMA = gamma
        self.EPSILON = epsilon
        self.EPS_END = epsEnd
        self.ALPHA = alpha
        self.actionSpace = actionSpace
        self.memSize = maxMemorySize
        self.steps = 0
        self.learn_step_counter = 0
        self.memory = []
        self.memCntr = 0
        self.replace_target_cnt = replace
        self.Q_eval = MODEL(alpha)
        self.Q_next = MODEL(alpha)

    def storeTransition(self, state, action, reward, state_):
        if self.memCntr < self.memSize:
            self.memory.append([state, action, reward, state_])
        else:            
            self.memory[self.memCntr%self.memSize] = [state, action, reward, state_]
        self.memCntr += 1
        
    def chooseAction(self, observation):
        rand = np.random.random()
        actions = self.Q_eval.forward(observation)
        if rand < 1 - self.EPSILON:
            action = T.argmax(actions[1]).item()
        else:
            action = np.random.choice(self.actionSpace)            
        self.steps += 1
        return action
    
    def learn(self, batch_size):
        self.Q_eval.optimizer.zero_grad()
        if self.replace_target_cnt is not None and \
           self.learn_step_counter % self.replace_target_cnt == 0:
            self.Q_next.load_state_dict(self.Q_eval.state_dict())

        if self.memCntr+batch_size < self.memSize:            
            memStart = int(np.random.choice(range(self.memCntr)))
        else:
            memStart = int(np.random.choice(range(self.memSize-batch_size-1)))
        miniBatch=self.memory[memStart:memStart+batch_size]
        memory = np.array(miniBatch)

        # convert to list because memory is an array of numpy objects
        Qpred = self.Q_eval.forward(list(memory[:,0][:])).to(self.Q_eval.device)
        Qnext = self.Q_next.forward(list(memory[:,3][:])).to(self.Q_eval.device)       
        
        maxA = T.argmax(Qnext, dim=1).to(self.Q_eval.device) 
        rewards = T.Tensor(list(memory[:,2])).to(self.Q_eval.device)        
        Qtarget = Qpred        
        Qtarget[:,maxA] = rewards + self.GAMMA*T.max(Qnext[1])
        
        if self.steps > 500:
            if self.EPSILON - 1e-4 > self.EPS_END:
                self.EPSILON -= 1e-4
            else:
                self.EPSILON = self.EPS_END

        #Qpred.requires_grad_()        
        loss = self.Q_eval.loss(Qtarget, Qpred).to(self.Q_eval.device)
        loss.backward()
        self.Q_eval.optimizer.step()
        self.learn_step_counter += 1

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

    def random_actions(self, agent):

        env = self.env

        while agent.memCntr < agent.memSize:
            observation = env.reset()
            done = False
            while not done:
                action = env.action_space.sample()
                observation_, reward, done, info = env.step(action)

                agent.storeTransition(observation, action, reward, observation_)
                observation = observation_

    def fit(self, agent, n_iters, batch_size=32):

        env = self.env
        scores = []
        # epsHistory = []

        for i in range(n_iters):
            
            # epsHistory.append(agent.EPSILON)        
            done = False
            observation = env.reset()
            frames = [observation]
            score = 0
            while not done:
                action = agent.chooseAction( observation=observation)

                observation_, reward, done, info = env.step(action)
                score += reward

                agent.storeTransition(observation, action, reward, observation_)
                observation = observation_            
                agent.learn(batch_size)

            scores.append(score)
            print('score:',score)

    def test_agent(self, agent, n_iters):
        self.env = gym.make(self.ENV_NAME)
        

        for _ in range(n_iters):
            done = False
            observation = self.env.reset()
            reward = 0
            info = {}

            while not done:
                self.env.render()

                action = agent.chooseAction(observation, reward, done)

                observation, reward, done, info = self.env.step(action)
                print(f"REWARD = {reward}")
        self.env.close()

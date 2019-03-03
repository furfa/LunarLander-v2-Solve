from keras.layers import Dense,LeakyReLU,ReLU,Activation,Flatten
from keras.optimizers import Adadelta,Adam
from keras.activations import softmax,relu
from keras.models import Sequential
import gym
from rl.agents.dqn import DQNAgent
from rl.policy import BoltzmannQPolicy , EpsGreedyQPolicy,BoltzmannGumbelQPolicy
from rl.memory import SequentialMemory
import numpy as np

env = gym.make("LunarLander-v2")
outup_shape = env.action_space.n
input_shape = env.observation_space.shape

#np.random.seed(123)
#env.seed(123)

model = Sequential()
model.add(Flatten(input_shape=(1,)+input_shape))
model.add(Dense(64,activation='relu'))
model.add(Dense(64,activation='relu'))
model.add(Dense(64,activation='relu'))
model.add(Dense(64,activation='relu'))
model.add(Dense(outup_shape))

memory = SequentialMemory(limit=75000,window_length=1)

policy = BoltzmannQPolicy()

dqn = DQNAgent(model=model, nb_actions=outup_shape, memory=memory, nb_steps_warmup=10,
               target_model_update=0.001, policy=policy)
#optim - sgd
# Дуэлинг Нетворк, как сделал тот чел - параметр DQN агента, по дефолту = False
# Можно изи списать с DQN`A для торча
#target model update = 10000
# test with target_model_ = 10^-3
#gamma .99
#Для торча чел фитить 2к игр,
#
dqn.compile(Adam(lr=0.001), metrics=['mse'])

dqn.fit(env, nb_steps=40000, visualize=True, verbose=1)
dqn.test(env, nb_episodes=10, visualize=True)
from BaseClasses import *
import TorchModels

"""
    Model init 
"""

agent = Agent(TorchModels.SimpleNet, gamma=0.95, epsilon=1.0, 
                  alpha=0.003, maxMemorySize=5000,
                  replace=None) 

gR = GymRunner()

"""
    Случайные действия для начала
    Внутри они выполняются пока не закончится память, выделенная под агента
"""

gR.random_actions(agent)

env = gR.env
scores = []
batch_size = 5
# epsHistory = []

for i in range(100):
    
    # epsHistory.append(agent.EPSILON)        
    done = False
    observation = env.reset()
    frames = [observation]
    score = 0
    while not done:
        env.render()
        action = agent.chooseAction( observation=observation)

        observation_, reward, done, info = env.step(action)
        score += reward

        agent.storeTransition(observation, action, reward, observation_)
        observation = observation_            
        agent.learn(batch_size)

    scores.append(score)
    print('score:',score)





# gR.fit(Agent, 500)
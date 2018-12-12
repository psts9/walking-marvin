import numpy as np
import gym
import pickle
import copy
import envs
import sys, getopt

LEARNING_RATE = 0.03
SIGMA = 0.1
POPULATION_SIZE = 20
MAX_STEPS = 500
DECAY = 1.0
MAX_EPISODES = 1000

L1_SIZE = 24
L2_SIZE = 24

RENDER = False

S_DIM = None
A_DIM = None

def relu(x):
    return x * (x > 0)

class NeuralNetwork():

    def __init__(self, i_dim, o_dim):
        self.learning_rate = LEARNING_RATE
        self.w1 = np.random.randn(i_dim, L1_SIZE)
        self.w2 = np.random.randn(L1_SIZE, L2_SIZE)
        self.w3 = np.random.randn(L2_SIZE, o_dim)

    def feedforward(self, input):
        self.l1 = input.dot(self.w1)
        self.l2 = self.l1.dot(self.w2)
        output  = np.tanh(self.l2.dot(self.w3))
        return output
    
    def get_mutated(self, p):
        mutated = copy.deepcopy(self)
        mutated.w1 += SIGMA * p.w1
        mutated.w2 += SIGMA * p.w2
        mutated.w3 += SIGMA * p.w3
        return mutated
    
    def get_weights(self):
        return self.w1, self.w2, self.w3
    
    def update(self, rewards, population):
        std = rewards.std()
        if std == 0:
            return
        rewards = (rewards - rewards.mean()) / std
        update = self.learning_rate / (POPULATION_SIZE * SIGMA)
        self.w1 += update * np.dot(np.array([p.w1 for p in population]).T, rewards).T
        self.w2 += update * np.dot(np.array([p.w2 for p in population]).T, rewards).T
        self.w3 += update * np.dot(np.array([p.w3 for p in population]).T, rewards).T
        self.learning_rate *= DECAY
    
    def save(self):
        pickle.dump(self.get_weights(), open("save.p", "wb"))
    
    def load(self):
        weights = pickle.load(open("save.p", "rb"))
        self.w1 = weights[0]
        self.w2 = weights[1]
        self.w3 = weights[2]

def create_new_population(nn):
    population = []
    for _ in range(POPULATION_SIZE):
        population.append(NeuralNetwork(S_DIM, A_DIM))
    return population

def get_fitness(population, nn, env):
    rewards = []
    for m in population:
        mutant = nn.get_mutated(m)
        rewards.append(get_reward(mutant, env))
    return np.array(rewards)

def get_reward(nn, env, render=False, print=False):
    total_reward = 0
    observation = env.reset()
    for _ in range(MAX_STEPS):
        if render:
            env.render(mode='Human')
        action = nn.feedforward(observation)
        observation, reward, done, _ = env.step(action)
        total_reward += reward
        if done:
            break
    return total_reward

def walk(nn, env):
    total_reward = 0
    observation = env.reset()
    while True:
        env.render(mode='Human')
        action = nn.feedforward(observation)
        observation, reward, done, _ = env.step(action)
        total_reward += reward
        if done:
            break
    return total_reward


def main(argv):

    np.random.seed(42)

    env = gym.make('Marvin-v0')
    env.seed(1234)

    opts, args = getopt.getopt(argv, "whls", ["walk", "help", "load", "save"])

    WALK = False
    LOAD = False
    SAVE = False

    for opt, arg in opts:
        if opt in ("-h", "--help"):
            print("(-w or --walk) to show just the walking process")
            print("(-l or --load) to load a save")
            print("(-s or --save) to save a model")
        elif opt in ("-w", "--walk"):
            WALK = True
        elif opt in ("-l", "--load"):
            LOAD = True
        elif opt in ("-s", "--save"):
            SAVE = True

    global S_DIM
    global A_DIM

    S_DIM = env.observation_space.shape[0]
    A_DIM = env.action_space.shape[0]

    nn = NeuralNetwork(S_DIM, A_DIM)

    if LOAD:
        nn.load()

    if not WALK:
        for episode in range(MAX_EPISODES):
            population = create_new_population(nn)
            fitness = get_fitness(population, nn, env)
            nn.update(fitness, population)
            reward = get_reward(nn, env, render=RENDER)
            print('episode =', episode + 1, 'reward =', reward, 'min =', fitness.min(), 'max =', fitness.max(), 'avg =', fitness.mean())
            #if fitness.mean() >= 10.0:
             #   break
    
    if SAVE:
        nn.save()

    while True:
        reward = walk(nn, env)
        print('reward =', reward)
        
    


"""
def main():

    env = gym.make('BipedalWalker-v2')
    
    S_DIM = env.observation_space.shape[0]
    A_DIM = env.action_space.shape[0]

    #nn = NeuralNetwork(S_DIM, A_DIM)

    if RESUME:
        w = pickle.load(open("save.p", "rb"))
    else:
        w = np.random.randn(S_DIM, A_DIM)
    
    for episode in range(1000):
        #reward = f(env, nn, render=True)
        reward = f2(w, env, render=RENDER)
        print('episode = ', episode, 'reward =', reward)

        N = np.random.randn(POPULATION_SIZE, A_DIM)
        R = np.zeros(POPULATION_SIZE)
        for i in range(POPULATION_SIZE):
            w_try = w + SIGMA * N[i]
            R[i] = f2(w_try, env, render=False)
        
        A = (R - np.mean(R)) / np.std(R)
        w = w + LEARNING_RATE / (POPULATION_SIZE * SIGMA) * N.T.dot(A)

        if (episode % 100 == 0):
            pickle.dump(w, open("save.p", "wb"))
"""





if __name__ == '__main__':
    main(sys.argv[1:])

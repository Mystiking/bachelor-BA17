import gym
from gym import wrappers
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Model, Sequential
from keras.layers import Dense, Input, Convolution2D, MaxPooling2D, UpSampling2D
from keras import regularizers
from keras import optimizers 
from keras.optimizers import sgd, RMSprop
from keras import backend as K
import tensorflow as tf
import random

GAME = 'CartPole-v0'
L_RATE = 0.005
MINI_BATCH = 32
GAMMA = 0.9
N_STEP = 8
ALPHA = 0.5

class Environment:
    def __init__(self, GAME):
        self.env = gym.make(GAME)
        self.agent = Agent()
        #self.critic = Critic()
    
    def run(self):
        state = self.env.reset()
        state = state.reshape([1, 4])
        total_reward = 0
        while True:
            self.env.render()
            action = self.agent.get_action(state)
            _state, reward, done, info = self.env.step(action)
            _state = _state.reshape([1, 4])
            self.agent.train(state, action, reward, _state, done)
            total_reward += reward
            state = _state
            if done:
                break
        print(total_reward)

class Critic:
    def __init__(self):
        self.replay_memory = [[], [], [], [], []]
        self.session = tf.Session()
        K.set_session(self.session)
        K.manual_variable_initialization(True)

        shared_input = Input(batch_shape=(None, NUM_STATES))
        shared_layer = Dense(16, activation='relu', init='uniform')(shared_input)
        shared_optimizer = RMSprop(L_RATE, decay=0.99)

        actor_output = Dense(NUM_ACTIONS, activation='softmax', init='uniform')(shared_layer)
        self.actor_model = Model(shared_input, actor_output)
        #self.actor_model.compile(loss='mse', optimizer=shared_optimizer)

        critic_output = Dense(1, activation='linear', init='uniform')(shared_layer)
        self.critic_model = Model(shared_input, critic_output)
        #self.critic_model.compile(loss='mse', optimizer=shared_optimizer)
        self.optimizer = tf.train.RMSPropOptimizer(L_RATE, decay=0.99)

        self.graph = self.train_model()
        g = tf.global_variables_initializer()
        self.session.run(g)
        self.default_graph = tf.get_default_graph()

        # self.default_graph.finalize()

    def train_model(self):#, state, action, reward):
        #tf_state = tf.Variable(state)
        #tf_action = tf.Variable(action)
        #tf_reward = tf.Variable(reward)
        # Tensorflow placeholders for a state, an action and a reward
        tf_state = tf.placeholder(tf.float32, shape=(None, NUM_STATES))
        tf_action = tf.placeholder(tf.float32, shape=(None, NUM_ACTIONS))
        tf_reward = tf.placeholder(tf.float32, shape=(None, 1))

        # Policy (action probabilities)
        probs = self.actor_model(tf_state)

        # Value of state
        value = self.critic_model(tf_state)
        
        advantage = tf_reward - value
        
        prob = tf.reduce_sum(probs * tf_action)
        log_prob = tf.log(prob)
        loss_policy = - log_prob * advantage
        loss_value = ALPHA * advantage * advantage

        loss_total = tf.reduce_mean(loss_value + loss_policy)
        minimized = self.optimizer.minimize(loss_total)
        return tf_state, tf_action, tf_reward, minimized

    def optimize(self):
        state, action, reward, _state, done = self.replay_memory
        state = np.vstack(state)
        action = np.vstack(action)
        reward = np.vstack(reward)
        _state = np.vstack(_state)

        value = self.critic_model.predict(_state)
        reward = reward + (GAMMA ** N_STEPS) * value if not done else reward
        tf_state, tf_action, tf_reward, minimize = self.graph
        self.session.run(minimize, feed_dict={tf_state : state, tf_action : action, tf_reward : reward})

    def push_to_replay_memory(self, state, action, reward, _state, done):
        self.replay_memory[0].append(state)
        self.replay_memory[1].append(action)
        self.replay_memory[2].append(reward)
        self.replay_memory[3].append(_state)
        self.replay_memory[4].append(done)


class Agent:
    
    def __init__(self):
        self.memory = []
        self.R = 0

    def get_action(self, state):
        probs = critic.actor_model.predict(state)
        action = np.random.choice(NUM_ACTIONS, p=probs[0])
        return action

    def get_sample(self, memory, n):
        state, action, _, _, _ = memory[0]
        _, _, _, _state, done = memory[n-1]

        return state, action, self.R, _state, done

    def train(self, state, action, reward, _state, done):
        a_one_hot = np.zeros(NUM_ACTIONS)
        a_one_hot[action] = 1
        
        self.memory.append((state, a_one_hot, reward, _state, done))
        
        self.R = (self.R + reward * (GAMMA ** N_STEP)) / GAMMA

        if done:
            while(len(self.memory) > 0):
                n = len(self.memory)
                s, a, r, _s, d = self.get_sample(self.memory, n)
                critic.push_to_replay_memory(s, a, r, _s, d)

                self.R = (self.R - self.memory[0][2]) / GAMMA
                self.memory.pop(0)
            self.R = 0

        if len(self.memory) >= N_STEP:
            s, a, r, _s, d = self.get_sample(self.memory, N_STEP)
            critic.push_to_replay_memory(s, a, r, _s, d)

            self.R = self.R - self.memory[0][2]
            self.memory.pop(0)


episodes = 200
env = Environment(GAME)
NUM_ACTIONS = env.env.action_space.n
NUM_STATES = env.env.observation_space.shape[0]
critic = Critic()

while(episodes > 0):
    env.run()
    critic.optimize()


print("Hello world")

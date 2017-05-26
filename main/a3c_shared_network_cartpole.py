import gym
from gym import wrappers
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Model, Sequential
from keras.layers import Dense, Input, Convolution2D, MaxPooling2D, UpSampling2D, Flatten
from keras import regularizers
from keras import optimizers 
from keras.optimizers import sgd, RMSprop
from keras import backend as K
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
import random
import time
import threading

def resize(img):
    return img[::2,::2]

def rgb2grayscale(img):
    return np.dot(img[:,:,:3], [0.2126, 0.7152, 0.0722])

gtheta = 0
gw = 0
# Constants
ALPHA = 0.9
GAMMA = 0.9
LR = 0.001
EPSILON = 1e-8
lock = threading.Lock()
GAME_NAME = 'CartPole-v0'
# The environtment that we will be using
env = gym.make(GAME_NAME)
NUM_STATE = env.observation_space.shape[0]
NUM_ACTIONS = env.action_space.n

# For graphing
reward_list = []

# Init layers of the model
input_layer      = Input(shape=[4])
# First conv layer
first= Dense(8, activation='relu', init='uniform')(input_layer)
# Output layers
output_actor  = Dense(units=NUM_ACTIONS, activation='softmax', kernel_initializer='uniform')(first)
output_critic = Dense(units=1, activation='linear', kernel_initializer='uniform')(first)
# The "final" networks
global_model = Model(inputs=input_layer, outputs=[output_actor, output_critic])
# Must be done to enable threading
global_model._make_predict_function()

# Initialize tf global variables
sess = tf.Session()
K.set_session(sess)
init = tf.global_variables_initializer()
sess.run(init)
graph = tf.get_default_graph()


class Agent(threading.Thread):
    def __init__(self, state_input_shape, action_output_shape, threadid):
        self.result_file = open("breakout_agent_{}_results.csv".format(threadid), 'w')
        self.start_time = time.time()
        threading.Thread.__init__(self)
        self.step_counter = 0
        gw = 0
        # Init layers of the model
        input_layer      = Input(shape=[4])
        # First conv layer
        first= Dense(8, activation='relu', init='uniform')(input_layer)
        # Output layers
        output_actor  = Dense(units=NUM_ACTIONS, activation='softmax', kernel_initializer='uniform')(first)
        output_critic = Dense(units=1, activation='linear', kernel_initializer='uniform')(first)
        # The "final" networks
        self.local_model = Model(inputs=input_layer, outputs=[output_actor, output_critic])
        # Must be done to enable threading
        self.local_model._make_predict_function()
        print(self.local_model.summary())
        
        self.s_t = tf.placeholder(tf.float32, shape=(None, NUM_STATE))
        self.a_t = tf.placeholder(tf.float32, shape=(None, NUM_ACTIONS))
        self.r_t = tf.placeholder(tf.float32, shape=(None, 1))
        p, v = self.local_model(self.s_t)
        
        weights = self.local_model.trainable_weights[::2] # weight tensors
        # Actor weights
        weights_actor = [weights[0], weights[1]]
        weights_critic = [weights[0], weights[2]]
        
        # "Losses"
        advantage = (self.r_t - v)
        loss_actor = tf.log(tf.reduce_sum(p * self.a_t, axis=1, keep_dims=True) + 1e-10) * (advantage)
        loss_critic = 0.5 * tf.square(advantage)
        loss_entropy = 0.01 * tf.reduce_sum(p * tf.log(p + 1e-10), axis=1, keep_dims=True)
        self.a_grads = tf.gradients(loss_actor + loss_entropy, weights_actor)
        self.c_grads = tf.gradients(loss_critic, weights_critic)

    def run(self):
        self.train(5, 5000)
        print("It worked!")

    def train(self, t_max, episodes):
        global gtheta, gw
        i = 0
        env = gym.make(GAME_NAME)
        while i < episodes: 
            i += 1
            # Initialize weights of local networks
            lock.acquire()
            with graph.as_default():
                start_weights = global_model.get_weights()
                self.local_model.set_weights(start_weights)
            lock.release() 
            dtheta = 0
            dw = 0
            total_reward = 0
            t_start = self.step_counter
            memory = []
            state = env.reset()   
            state = state.reshape([1, NUM_STATE])
            done = False
            while not done:
                probabilities, _ = self.local_model.predict(state)
                action = np.random.choice(2, p=probabilities[0])
                _state, reward, done, info = env.step(action)
                _state = _state.reshape([1, NUM_STATE])
                total_reward += reward
                memory.append([state, action, reward, done])
                state = _state
                self.step_counter += 1
                # Update every 5th step
                if self.step_counter - t_start == t_max or done:
                    R = 0 if done else self.local_model.predict(state)[1]
                    memory = reversed(memory)
                    for s, a, r, d in memory:
                        R = r + GAMMA * R
                        a_list = np.zeros(NUM_ACTIONS)
                        a_list[a] = 1
                        a_list = np.reshape(a_list, (-1, NUM_ACTIONS))
                        actor_grads = sess.run(self.a_grads, {self.s_t : s, self.a_t : a_list, self.r_t : np.reshape(np.array([R]), (-1, 1))})
                        dtheta = dtheta + np.array(actor_grads)
                        critic_grads = sess.run(self.c_grads, {self.s_t : s, self.a_t : a_list, self.r_t : np.reshape(np.array([R]), (-1, 1))})
                        dw = dw + np.array(critic_grads)
                    gtheta = gtheta * ALPHA + (1 - ALPHA) * (dtheta**2)
                    gw = gw * ALPHA + (1 - ALPHA) * (dw**2)
                    lock.acquire()
                    with graph.as_default():
                        j = 0
                        critic_layers = [1, 3]
                        actor_layers = [1, 2]
                        while j < len(dtheta):
                            if j < len(critic_layers):
                                critic_idx = critic_layers[j]
                                actor_idx = actor_layers[j]
                                # Critic
                                update_critic = global_model.layers[critic_idx].get_weights()[0] - LR * (dw[j] / np.sqrt(gw[j] + EPSILON))
                                bias_critic = global_model.layers[critic_idx].get_weights()[1]
                                global_model.layers[critic_idx].set_weights((update_critic, bias_critic))
                                # Actor
                                update_actor = global_model.layers[actor_idx].get_weights()[0] - LR * (dtheta[j] / np.sqrt(gtheta[j] + EPSILON))
                                bias_actor = global_model.layers[actor_idx].get_weights()[1]
                                global_model.layers[actor_idx].set_weights((update_actor, bias_actor))
                                j += 1
                        start_weights = global_model.get_weights()
                        self.local_model.set_weights(start_weights)
                    lock.release()       
                    dtheta = 0
                    dw = 0
                    t_start = self.step_counter
                    memory = []
            print("Episode {} : Reward = {}".format(i, total_reward))
            now = time.time()
            self.result_file.write("{},{},{}\n".format(i, total_reward, now - self.start_time))
        self.result_file.close()

            
        
agent006 = Agent(NUM_STATE, NUM_ACTIONS, 1)
agent007 = Agent(NUM_STATE, NUM_ACTIONS, 2)
#agent008 = Agent(NUM_STATE, NUM_ACTIONS, 3)
#agent009 = Agent(NUM_STATE, NUM_ACTIONS, 4)
#agent010 = Agent(NUM_STATE, NUM_ACTIONS, 5)
#agent011 = Agent(NUM_STATE, NUM_ACTIONS, 6)
#agent012 = Agent(NUM_STATE, NUM_ACTIONS, 7)
#agent013 = Agent(NUM_STATE, NUM_ACTIONS, 8)
#agent014 = Agent(NUM_STATE, NUM_ACTIONS, 9)
#agent015 = Agent(NUM_STATE, NUM_ACTIONS, 10)
#agent016 = Agent(NUM_STATE, NUM_ACTIONS, 11)
#agent017 = Agent(NUM_STATE, NUM_ACTIONS, 12)
#agent018 = Agent(NUM_STATE, NUM_ACTIONS, 13)
#agent019 = Agent(NUM_STATE, NUM_ACTIONS, 14)
#agent020 = Agent(NUM_STATE, NUM_ACTIONS, 15)
#agent021 = Agent(NUM_STATE, NUM_ACTIONS, 16)

agent006.start()
time.sleep(1)
agent007.start() 
time.sleep(1)
#agent008.start() 
time.sleep(1)
#agent009.start() 
time.sleep(1)
#agent010.start() 
time.sleep(1)
#agent011.start() 
time.sleep(1)
#agent012.start() 
time.sleep(1)
#agent013.start() 
time.sleep(1)
#agent014.start() 
time.sleep(1)
#agent015.start() 
time.sleep(1)
#agent016.start() 
time.sleep(1)
#agent017.start() 
time.sleep(1)
#agent018.start() 
time.sleep(1)
#agent019.start() 
time.sleep(1)
#agent020.start() 
time.sleep(1)
#agent021.start() 

agent006.join()
agent007.join() 
#agent008.join() 
#agent009.join() 
#agent010.join() 
#agent011.join() 
#agent012.join() 
#agent013.join() 
#agent014.join() 
#agent015.join() 
#agent016.join() 
#agent017.join() 
#agent018.join() 
#agent019.join() 
#agent020.join() 
#agent021.join() 
'''
c = 0
k = 500
avg_list = []
for i in range(len(reward_list) - 1):
    if c % k == 0:
        avg_list.append(sum(reward_list[i + c : i + c + k]) / k)

print(sum(reward_list) / len(reward_list))
plt.plot(range(len(reward_list)), reward_list, 'r')
plt.plot(range(len(avg_list)), avg_list, 'b')
plt.xlabel('Episodes')
plt.ylabel('Score')
plt.title('Score of the a3c algorithm over time')
plt.show()
'''

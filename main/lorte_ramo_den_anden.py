import gym
from gym import wrappers
import numpy as np
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
    i = img[::2, ::2]
    return i[15:,:]

def rgb2grayscale(img):
    return np.dot(img[:,:,:3], [0.2126, 0.7152, 0.0722])

gtheta = 0
gw = 0
# Constants
ALPHA = 0.99
GAMMA = 0.99
LR = 0.0001
EPSILON = 1e-8
lock = threading.Lock()
GAME_NAME = 'Pong-v0'

# The environtment that we will be using
env = gym.make(GAME_NAME)
NUM_STATE = env.observation_space.shape
NUM_ACTIONS = 3#env.action_space.n
RESIZED_WIDTH = int(NUM_STATE[0] / 2) - 15
RESIZED_HEIGHT = int(NUM_STATE[1] / 2)

# For graphing
reward_list = []

# Init layers of the model
input_layer      = Input(shape=(RESIZED_WIDTH, RESIZED_HEIGHT, 1))
# First conv layer
first= Convolution2D(16, kernel_size=(8, 8), strides=4, padding='same', use_bias=1, activation='relu', kernel_initializer='uniform')(input_layer)
# Second conv layer
second= Convolution2D(32, kernel_size=(4, 4), strides=2, padding='same', use_bias=1, activation='relu', kernel_initializer='uniform')(first)
# Flattening the convs
flat= Flatten()(second)
# Dense layer 1
dense= Dense(units=256, activation='relu', kernel_initializer='uniform')(flat)
# Output layers
output_actor  = Dense(units=NUM_ACTIONS, activation='softmax', kernel_initializer='uniform')(dense)
output_critic = Dense(units=1, activation='linear', kernel_initializer='uniform')(dense)
# The "final" networks
global_model = Model(inputs=input_layer, outputs=[output_actor, output_critic])
# Must be done to enable threading
global_model._make_predict_function()

opt = tf.train.RMSPropOptimizer(LR, epsilon=EPSILON, decay=ALPHA)
# Initialize tf global variables
sess = tf.Session()
K.set_session(sess)
init = tf.global_variables_initializer()
sess.run(init)
graph = tf.get_default_graph()


class Agent(threading.Thread):
    def __init__(self, state_input_shape, action_output_shape, threadid, main = False):
        self.result_file = open("breakout_agent_{}_results.csv".format(threadid), 'w')
        self.main = main
        self.start_time = time.time()
        threading.Thread.__init__(self)
        self.step_counter = 0
        gw = 0
        # Init layers of the model
        input_layer      = Input(shape=(RESIZED_WIDTH, RESIZED_HEIGHT, 1))
        # First conv layer
        first = Convolution2D(16, kernel_size=(8, 8), strides=4, padding='same', use_bias=1, activation='relu', kernel_initializer='uniform')(input_layer)
        # Second conv layer
        second = Convolution2D(32, kernel_size=(4, 4), strides=2, padding='same', use_bias=1, activation='relu', kernel_initializer='uniform')(first)
        # Flattening the convs
        flat = Flatten()(second)
        # Dense layer 1
        dense = Dense(units=256, activation='relu', kernel_initializer='uniform')(flat)
        # Output layers
        output_actor  = Dense(units=NUM_ACTIONS, activation='softmax', kernel_initializer='uniform')(dense)
        output_critic = Dense(units=1, activation='linear', kernel_initializer='uniform')(dense)
        # The "final" networks
        self.local_model = Model(inputs=input_layer, outputs=[output_actor, output_critic])
        # Must be done to enable threading
        self.local_model._make_predict_function()
        
        self.s_t = tf.placeholder(tf.float32, shape=(None, RESIZED_WIDTH, RESIZED_HEIGHT, 1))
        self.a_t = tf.placeholder(tf.float32, shape=(None, NUM_ACTIONS))
        self.r_t = tf.placeholder(tf.float32, shape=(None, 1))
        self.advantage = tf.placeholder(tf.float32, shape=(None, 1))
        p, v = self.local_model(self.s_t)
        
        weights = self.local_model.trainable_weights[::2] # weight tensors
        # Actor weights
        
        # "Losses"
        
        log_p = tf.log(tf.clip_by_value(p, 0.000001, 0.999999))
        loss_actor = -tf.reduce_mean(tf.reduce_sum(tf.multiply(log_p, self.a_t), reduction_indices=1) * self.advantage)
        loss_critic = 0.5 * tf.reduce_mean(tf.square(self.r_t - v))
        loss_entropy = 0.01 * tf.reduce_sum(tf.multiply(p, -log_p))
        with graph.as_default():
            self.a_grads = tf.gradients(loss_critic + loss_actor - loss_entropy, weights) 
            self.a_grads, _ = tf.clip_by_global_norm(self.a_grads, 40.0)
            grads_var = list(zip(self.a_grads, weights))
            self.update = opt.apply_gradients(grads_var)

    def run(self):
        self.train(5, 5000)
        print("It worked!")

    def train(self, t_max, episodes):
        global gtheta, gw
        i = 0
        env = gym.make(GAME_NAME)
        while i < episodes: 
            i += 1
            repeat = 0
            # Initialize weights of local networks
            with graph.as_default():
                start_weights = global_model.get_weights()
                self.local_model.set_weights(start_weights)
            dtheta = 0
            dw = 0
            total_reward = 0
            t_start = self.step_counter
            memory = []
            state = env.reset()   
            state = resize(rgb2grayscale(state) )
            state = np.reshape(state, (-1, state.shape[0], state.shape[1], 1))
            done = False
            while not done:
                probabilities, value = self.local_model.predict(state)
                if repeat % 50 == 0 and self.main:
                    print(probabilities)
                    print(value)
                action = np.random.choice([1, 2, 3], p=probabilities[0])
                _state, reward, done, info = env.step(action)
                _state = resize(rgb2grayscale(_state) )
                _state = np.reshape(_state, (-1, _state.shape[0], _state.shape[1], 1))
                total_reward += reward
                memory.append([state, action, reward, done, value])
                state = _state
                self.step_counter += 1
                repeat += 1
                # Update every 5th step
                if self.step_counter - t_start == t_max or done:
                    R = 0 if done else self.local_model.predict(state)[1]
                    memory = reversed(memory)
                    for s, a, r, d, v in memory:
                        R = r + GAMMA * R
                        # a_list : [0, 0, 0]
                        a_list = np.zeros(NUM_ACTIONS)
                        # a_list : [1, 0, 0]
                        # For breakout
                        if a == 2:
                            a_list[1] = 1.
                        elif a == 3:
                            a_list[2] = 1.
                        else:
                            a_list[0] = 1.

                        a_list = np.reshape(a_list, (-1, NUM_ACTIONS))
                        advantage = np.reshape(np.array([R - v]), (-1, 1))
                        sess.run(self.update, {self.s_t : s, self.a_t : a_list, self.r_t : np.reshape(np.array([R]), (-1, 1)), self.advantage : advantage})
                    with graph.as_default():
                        start_weights = self.local_model.get_weights()
                        global_model.set_weights(start_weights)


                    '''
                    gtheta = gtheta * ALPHA + (1 - ALPHA) * (dtheta**2)
                    gw = gw * ALPHA + (1 - ALPHA) * (dw**2)
                    lock.acquire()
                    with graph.as_default():
                        j = 0
                        critic_layers = [1, 2, 4, 6]
                        actor_layers = [1, 2, 4, 5]
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
                    '''
                    dtheta = 0
                    dw = 0
                    t_start = self.step_counter
                    memory = []
            print("Episode {} : Reward = {}".format(i, total_reward))
            now = time.time()
            self.result_file.write("{},{},{}\n".format(i, total_reward, now - self.start_time))
        self.result_file.close()

            
        
agent006 = Agent(NUM_STATE, NUM_ACTIONS, 1, main = True)
agent007 = Agent(NUM_STATE, NUM_ACTIONS, 2)
agent008 = Agent(NUM_STATE, NUM_ACTIONS, 3)
agent009 = Agent(NUM_STATE, NUM_ACTIONS, 4)
agent010 = Agent(NUM_STATE, NUM_ACTIONS, 5)
agent011 = Agent(NUM_STATE, NUM_ACTIONS, 6)
agent012 = Agent(NUM_STATE, NUM_ACTIONS, 7)
agent013 = Agent(NUM_STATE, NUM_ACTIONS, 8)
agent014 = Agent(NUM_STATE, NUM_ACTIONS, 9)
agent015 = Agent(NUM_STATE, NUM_ACTIONS, 10)
agent016 = Agent(NUM_STATE, NUM_ACTIONS, 11)
agent017 = Agent(NUM_STATE, NUM_ACTIONS, 12)
agent018 = Agent(NUM_STATE, NUM_ACTIONS, 13)
agent019 = Agent(NUM_STATE, NUM_ACTIONS, 14)
agent020 = Agent(NUM_STATE, NUM_ACTIONS, 15)
agent021 = Agent(NUM_STATE, NUM_ACTIONS, 16)

agent006.start()
time.sleep(1)
agent007.start() 
time.sleep(1)
agent008.start() 
time.sleep(1)
agent009.start() 
time.sleep(1)
agent010.start() 
time.sleep(1)
agent011.start() 
time.sleep(1)
agent012.start() 
time.sleep(1)
agent013.start() 
time.sleep(1)
agent014.start() 
time.sleep(1)
agent015.start() 
time.sleep(1)
agent016.start() 
time.sleep(1)
agent017.start() 
time.sleep(1)
agent018.start() 
time.sleep(1)
agent019.start() 
time.sleep(1)
agent020.start() 
time.sleep(1)
agent021.start() 

agent006.join()
agent007.join() 
agent008.join() 
agent009.join() 
agent010.join() 
agent011.join() 
agent012.join() 
agent013.join() 
agent014.join() 
agent015.join() 
agent016.join() 
agent017.join() 
agent018.join() 
agent019.join() 
agent020.join() 
agent021.join() 
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

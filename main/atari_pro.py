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

def rgb2grayscale(img):
    return np.dot(img[:,:,:3], [0.299, 0.587, 0.114])

results = open('results_cp_11.csv', 'w')

# Constants
ALPHA = 0.5
GAMMA = 0.9
LR = 0.001
EPSILON = 1e-8
lock = threading.Lock()
GAME_NAME = 'Pong-v0'
# The environtment that we will be using
env = gym.make(GAME_NAME)
NUM_STATE = env.observation_space.shape
NUM_ACTIONS = env.action_space.n
nb_filters = 1
nb_conv = 3

# For graphing
reward_list = []

# Global networks
print(NUM_ACTIONS)
# Init layers of the model
input_layer   = Input(shape=(NUM_STATE[0], NUM_STATE[1], 1))
# First conv layer
first_actor   = Convolution2D(nb_filters, kernel_size=(nb_conv, nb_conv), padding='same', use_bias=1, activation='relu')(input_layer)
first_critic  = Convolution2D(nb_filters, kernel_size=(nb_conv, nb_conv), padding='same', use_bias=1, activation='relu')(input_layer)
# Second conv layer
second_actor  = Convolution2D(nb_filters, kernel_size=(nb_conv, nb_conv), strides=(5, 5), padding='same', use_bias=1, activation='relu')(first_actor)
second_critic = Convolution2D(nb_filters, kernel_size=(nb_conv, nb_conv), strides=(5, 5), padding='same', use_bias=1, activation='relu')(first_critic)
# Flattening the convs
flat_actor    = Flatten()(second_critic)
flat_critic   = Flatten()(second_actor)
# Dense layer 1
dense_actor   = Dense(output_dim=8, activation='relu', kernel_initializer='uniform')(flat_actor)
dense_critic  = Dense(output_dim=8, activation='relu', kernel_initializer='uniform')(flat_critic)
# Output layers
output_actor  = Dense(output_dim=NUM_ACTIONS, activation='softmax', kernel_initializer='uniform')(dense_actor)
output_critic = Dense(output_dim=1, activation='linear', kernel_initializer='uniform')(dense_critic)
# The "final" networks
global_critic_model = Model(inputs=input_layer, outputs=output_critic)
global_actor_model = Model(inputs=input_layer, outputs=output_actor)
# Must be done to enable threading
global_critic_model._make_predict_function()
global_actor_model._make_predict_function()

# Initialize tf global variables
sess = tf.Session()
K.set_session(sess)
init = tf.global_variables_initializer()
sess.run(init)
graph = tf.get_default_graph()


class Agent(threading.Thread):
    def __init__(self, state_input_shape, action_output_shape):
        threading.Thread.__init__(self)
        self.opt = tf.train.RMSPropOptimizer(0.005)
        self.gtheta = 0
        self.gw = 0
        # Init layers of the model
        input_layer   = Input(shape=(NUM_STATE[0], NUM_STATE[1], 1))
        # First conv layer
        first_actor   = Convolution2D(nb_filters, kernel_size=(nb_conv, nb_conv), padding='same', use_bias=1, activation='relu')(input_layer)
        first_critic  = Convolution2D(nb_filters, kernel_size=(nb_conv, nb_conv), padding='same', use_bias=1, activation='relu')(input_layer)
        # Second conv layer
        second_actor  = Convolution2D(nb_filters, kernel_size=(nb_conv, nb_conv), strides=(5, 5), padding='same', use_bias=1, activation='relu')(first_actor)
        second_critic = Convolution2D(nb_filters, kernel_size=(nb_conv, nb_conv), strides=(5, 5), padding='same', use_bias=1, activation='relu')(first_critic)
        # Flattening the convs
        flat_actor    = Flatten()(second_critic)
        flat_critic   = Flatten()(second_actor)
        # Dense layer 1
        dense_actor   = Dense(output_dim=8, activation='relu', kernel_initializer='uniform')(flat_actor)
        dense_critic  = Dense(output_dim=8, activation='relu', kernel_initializer='uniform')(flat_critic)
        # Output layers
        output_actor  = Dense(output_dim=NUM_ACTIONS, activation='softmax', kernel_initializer='uniform')(dense_actor)
        output_critic = Dense(output_dim=1, activation='linear', kernel_initializer='uniform')(dense_critic)
        # The "final" networks
        self.local_critic_model = Model(inputs=input_layer, outputs=output_critic)
        self.local_actor_model = Model(inputs=input_layer, outputs=output_actor)
        # Must be done to enable threading
        self.local_critic_model._make_predict_function()
        self.local_actor_model._make_predict_function()
        
        self.s_t = tf.placeholder(tf.float32, shape=(None, NUM_STATE[0], NUM_STATE[1], 1))
        self.a_t = tf.placeholder(tf.float32, shape=(None, NUM_ACTIONS))
        self.r_t = tf.placeholder(tf.float32, shape=(None, 1))
        v = self.local_critic_model(self.s_t)
        p = self.local_actor_model(self.s_t)
        # Actor weights
        w_actor = self.local_actor_model.trainable_weights # weight tensors
        w_actor = [weight for weight in w_actor]
        w_actor = w_actor[::2]
        # Critic weights
        w_critic = self.local_critic_model.trainable_weights # weight tensors
        w_critic = [weight for weight in w_critic]
        w_critic = w_critic[::2]
        # "Losses"
        advantage = (self.r_t - v)
        loss_actor = tf.log(tf.reduce_sum(p * self.a_t, axis=1, keep_dims=True) + 1e-10) * tf.stop_gradient(advantage)
        loss_critic = tf.square(advantage)
        self.a_grads = tf.gradients(loss_actor, w_actor)
        self.c_grads = tf.gradients(loss_critic, w_critic)

    def run(self):
        self.train(200, 4000)
        print("It worked!")

    def train(self, t, episodes):
        print("OMG")
        i = 0
        
        env = gym.make(GAME_NAME)
        while i < episodes: 
            i += 1
            # Initialize weights of local networks
            lock.acquire()
            with graph.as_default():
                start_weights_actor = global_actor_model.get_weights()
                self.local_actor_model.set_weights(start_weights_actor)
                start_weights_critic = global_critic_model.get_weights()
                self.local_critic_model.set_weights(start_weights_critic)
            lock.release()
            
            dtheta = 0
            dw = 0
            total_reward = 0
            I = 1.
            memory = []
            state = env.reset()   
            state = rgb2grayscale(state) 
            state = np.reshape(state, (-1, state.shape[0], state.shape[1], 1))
            done = False
            value = self.local_critic_model.predict(state)
            while not done:
                probabilities = self.local_actor_model.predict(state)
                action = np.random.choice(6, p=probabilities[0])
                _state, reward, done, info = env.step(action)
                _state = rgb2grayscale(_state) 
                _state = np.reshape(_state, (-1, _state.shape[0], _state.shape[1], 1))
                _value = self.local_critic_model.predict(_state)
                _value = _value if not done else np.array([[0.]])
                total_reward += reward
                #td_error = reward + GAMMA * _value - value
                memory.append([state, action, reward, done])
                state = _state
                value = _value
            print("Episode {} : Reward = {}".format(i, total_reward))
            results.write(str(i) + ',' + str(total_reward) + '\n')
            reward_list.append(total_reward)
            R = 0
            print("Entering memory", len(memory)) 
            for s, a, r, d in memory:
                R = r + GAMMA * R
                a_list = np.zeros(NUM_ACTIONS)
                a_list[a] = 1.
                a_list = np.reshape(a_list, (-1, NUM_ACTIONS))
                actor_grads = sess.run(self.a_grads, {self.s_t : s, self.a_t : a_list, self.r_t : np.reshape(np.array([R]), (-1, 1))})
                dtheta = dtheta + np.array(actor_grads)
                critic_grads = sess.run(self.c_grads, {self.s_t : s, self.r_t : np.reshape(np.array([R]), (-1, 1))})
                dw = dw + np.array(critic_grads)
            lock.acquire()
            print("Lock acquired")
            with graph.as_default():
                
                self.gtheta = self.gtheta * ALPHA + (1 - ALPHA) * (dtheta**2)
                self.gw = self.gw * ALPHA + (1 - ALPHA) * (dw**2)
                j = 0
                layers_w_weights = [1, 2, 4, 5]
                while j < len(dtheta):
                    if j < len(layers_w_weights):
                        model_idx = layers_w_weights[j]
                        # Critic
                        update_critic = global_critic_model.layers[model_idx].get_weights()[0] - LR * dw[j] / np.sqrt(self.gw[j] + EPSILON)
                        bias_critic = global_critic_model.layers[model_idx].get_weights()[1]
                        global_critic_model.layers[model_idx].set_weights((update_critic, bias_critic))
                        # Actor
                        update_actor = global_actor_model.layers[model_idx].get_weights()[0] - LR * dtheta[j] / np.sqrt(self.gtheta[j] + EPSILON)
                        bias_actor = global_actor_model.layers[model_idx].get_weights()[1]
                        global_actor_model.layers[model_idx].set_weights((update_actor, bias_actor))
                        j += 1
            lock.release()
            
        
agent006 = Agent(NUM_STATE, NUM_ACTIONS)
agent007 = Agent(NUM_STATE, NUM_ACTIONS)
agent008 = Agent(NUM_STATE, NUM_ACTIONS)
agent009 = Agent(NUM_STATE, NUM_ACTIONS)

agent006.start()
agent007.start()
agent008.start()
agent009.start()

agent006.join()
agent007.join()
agent008.join()
agent009.join()
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

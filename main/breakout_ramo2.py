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
    return img[::2, ::2]


def rgb2grayscale(img):
    return np.dot(img[:, :, :3], [0.2126, 0.7152, 0.0722])


# Constants
ALPHA = 0.9
GAMMA = 0.9
LR = 0.00005
LEARNING_RATE = 5e-3
EPSILON = 1e-8
lock = threading.Lock()
GAME_NAME = 'Breakout-v0'
# The environtment that we will be using
env = gym.make(GAME_NAME)
NUM_STATE = env.observation_space.shape
NUM_ACTIONS = 3  # env.action_space.n
RESIZED_WIDTH = int(NUM_STATE[0] / 2)
RESIZED_HEIGHT = int(NUM_STATE[1] / 2)
nb_filters = 1
nb_conv = 3

# For graphing
reward_list = []

# Global networks
print(NUM_ACTIONS)
# Init layers of the model
input_layer = Input(shape=(RESIZED_WIDTH, RESIZED_HEIGHT, 1))
# First conv layer
first_layer = Convolution2D(16, kernel_size=(8, 8), strides=4, padding='same', use_bias=1, activation='relu',
                            kernel_initializer='uniform')(input_layer)
# Second conv layer
second_layer = Convolution2D(32, kernel_size=(4, 4), strides=2, padding='same', use_bias=1, activation='relu',
                             kernel_initializer='uniform')(first_layer)
# Flattening the convs
flat_layer = Flatten()(second_layer)
# Dense layer 1
dense_layer = Dense(units=256, activation='relu', kernel_initializer='uniform')(flat_layer)
# Output layers
output_actor = Dense(units=NUM_ACTIONS, activation='softmax', kernel_initializer='uniform')(dense_layer)
output_critic = Dense(units=1, activation='linear', kernel_initializer='uniform')(dense_layer)
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


class Global_network:
    memory = []
    lock_queue = threading.Lock()
    def __init__(self, state_input_shape, action_output_shape, threadid):

        self.session = tf.Session()
        K.set_session(self.session)
        K.manual_variable_initialization(True)

        self.model = self.build_model()
        self.graph = self.build_graph(self.model)

        self.session.run(tf.global_variables_initializer())
        self.default_graph = tf.get_default_graph()

        self.default_graph.finalize()  # avoid modifications


    def build_model(self):
        # Init layers of the model
        input_layer = Input(shape=(RESIZED_WIDTH, RESIZED_HEIGHT, 1))
        # First conv layer
        first_layer = Convolution2D(16, kernel_size=(8, 8), strides=4, padding='same', use_bias=1,
                                    activation='relu',
                                    kernel_initializer='uniform')(input_layer)
        # Second conv layer
        second_layer = Convolution2D(32, kernel_size=(4, 4), strides=2, padding='same', use_bias=1,
                                     activation='relu',
                                     kernel_initializer='uniform')(first_layer)
        # Flattening the convs
        flat_layer = Flatten()(second_layer)
        # Dense layer 1
        dense_layer = Dense(units=256, activation='relu', kernel_initializer='uniform')(flat_layer)
        # Output layers
        output_actor = Dense(units=NUM_ACTIONS, activation='softmax', kernel_initializer='uniform')(dense_layer)
        output_critic = Dense(units=1, activation='linear', kernel_initializer='uniform')(dense_layer)
        # The "final" networks
        model = Model(inputs=input_layer, outputs=[output_actor, output_critic])
        # Must be done to enable threading
        model._make_predict_function()
        print(model.summary())

        return model

    def build_graph(self, model):
        s_t = tf.placeholder(tf.float32, shape=(None, RESIZED_WIDTH, RESIZED_HEIGHT, 1))
        a_t = tf.placeholder(tf.float32, shape=(None, NUM_ACTIONS))
        r_t = tf.placeholder(tf.float32, shape=(None, 1))
        p, v = self.model(s_t)
        # Actor weights
        w_actor = self.model.trainable_weights  # weight tensors
        w_actor = [weight for weight in w_actor]
        w_actor = w_actor[::2]
        # "Losses"
        advantage = (r_t - v)
        loss_actor = (- tf.log(tf.reduce_sum(p * a_t, axis=1, keep_dims=True) + 1e-10)) * tf.stop_gradient(advantage)
        loss_critic = 0.5 * tf.square(advantage)
        loss_entropy = 0.01 * tf.reduce_sum(p * tf.log(p + 1e-10), axis=1, keep_dims=True)
        total_loss = (loss_actor + loss_critic + loss_entropy)
        #self.a_grads = tf.gradients(loss_actor + loss_entropy, w_actor)
        optimizer = tf.train.RMSPropOptimizer(LEARNING_RATE, decay=.99)
        minimize = optimizer.minimize(total_loss)

        return s_t, a_t, r_t, minimize

    def predict(self, s):
        with self.default_graph.as_default():
            p, v = self.model.predict(s)
            return p, v

    def predict_p(self, s):
        with self.default_graph.as_default():
            p, v = self.model.predict(s)
            return p

    def predict_v(self, s):
        with self.default_graph.as_default():
            p, v = self.model.predict(s)
            return v

    def optimize(self, s, a, r):
        s_t, a_t, r_t, minimize = self.graph
        self.session.run(minimize, feed_dict={s_t: s, a_t: a, r_t: np.reshape(np.array([r]), (-1, 1))})

class Agent(threading.Thread):
    def __init__(self, state_input_shape, action_output_shape, threadid):
        self.result_file = open("breakout_agent_{}_results.csv".format(threadid), 'w')
        self.start_time = time.time()
        threading.Thread.__init__(self)
        self.step_counter = 0
        self.gtheta = 0
        self.gw = 0
        self.memory = []  # used for n_step return
        self.step_counter  = 0

    def run(self):
        self.train(50, 5000)
        print("It worked!")

    def train(self, t_max, episodes):
        i = 0
        env = gym.make(GAME_NAME)
        while i < episodes:
            i += 1
            repeat = 0
            # Initialize weights of local networks
            total_reward = 0
            t_start = self.step_counter
            state = env.reset()
            state = resize(rgb2grayscale(state))
            state = np.reshape(state, (-1, state.shape[0], state.shape[1], 1))
            done = False
            while not done:
                probabilities, value = global_network.predict(state)
                print(value)
                action = np.random.choice([1, 2, 3], p=probabilities[0])
                _state, reward, done, info = env.step(action)
                _state = resize(rgb2grayscale(_state))
                _state = np.reshape(_state, (-1, _state.shape[0], _state.shape[1], 1))
                total_reward += reward
                self.memory.append([state, action, reward, done])
                state = _state
                self.step_counter += 1
                # Update every 5th step
                if self.step_counter - t_start == t_max or done:
                    if done:
                        R = 0
                    else:
                        v = global_network.predict_v(state)
                        R = v
                    self.memory = reversed(self.memory)
                    for s, a, r, d in self.memory:
                        R = r + GAMMA * R
                        # a_list : [0, 0, 0]
                        a_list = np.zeros(NUM_ACTIONS)
                        # a_list : [1, 0, 0]
                        # For breakout
                        if a == 1:
                            a_list[0] = 1.
                        elif a == 2:
                            a_list[1] = 1.
                        elif a == 3:
                            a_list[2] = 1.

                        a_list = np.reshape(a_list, (-1, NUM_ACTIONS))
                        lock.acquire()
                        global_network.optimize(s, a_list, R)
                        lock.release()

                        #actor_grads = sess.run(self.a_grads, {self.s_t: s, self.a_t: a_list,
                        #                                      self.r_t: np.reshape(np.array([R]), (-1, 1))})
                        #dtheta = dtheta + np.array(actor_grads)
                        #critic_grads = sess.run(self.c_grads, {self.s_t: s, self.a_t: a_list,
                        #                                       self.r_t: np.reshape(np.array([R]), (-1, 1))})
                        #dw = dw + np.array(critic_grads)

                    t_start = self.step_counter
                    self.memory = []
            print("Episode {} : Reward = {}".format(i, total_reward))
            print(probabilities)
            now = time.time()
            self.result_file.write("{},{},{}\n".format(i, total_reward, now - self.start_time))
        self.result_file.close()

global_network = Global_network(NUM_STATE, NUM_ACTIONS, 10)

agent006 = Agent(NUM_STATE, NUM_ACTIONS, 1)
# agent007 = Agent(NUM_STATE, NUM_ACTIONS, 2)
# agent008 = Agent(NUM_STATE, NUM_ACTIONS, 3)
# agent009 = Agent(NUM_STATE, NUM_ACTIONS, 4)

agent006.start()
# agent007.start()
# agent008.start()
# agent009.start()

agent006.join()
# agent007.join()
# agent008.join()
# agent009.join()
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

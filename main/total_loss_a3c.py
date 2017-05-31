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

rewards = []
global_step_counter = 0
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
NUM_THREADS = 16
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
        
        self.s_t = tf.placeholder(tf.float32, shape=(None, RESIZED_WIDTH, RESIZED_HEIGHT, 1), name="states")
        self.a_t = tf.placeholder(tf.float32, shape=(None, NUM_ACTIONS), name="actions")
        self.r_t = tf.placeholder(tf.float32, shape=(None, 1), name="reward")
        self.advantage = tf.placeholder(tf.float32, shape=(None, 1), name="advantages")
        p, v = self.local_model(self.s_t)
        
        weights = self.local_model.trainable_weights[::2] # weight tensors
        # Actor weights
        weights_a = [weights[0], weights[1], weights[2], weights[3]]
        weights_c = [weights[0], weights[1], weights[2], weights[4]]
        # "Losses"
        
        log_p = tf.log(tf.clip_by_value(p, 0.000001, 0.999999))
        loss_actor = -tf.reduce_mean(tf.reduce_sum(tf.multiply(log_p, self.a_t), reduction_indices=1) * self.advantage)
        loss_critic = tf.reduce_mean(tf.square(self.r_t - v))
        loss_entropy = 0.01 * tf.reduce_sum(tf.multiply(p, -log_p))
        with graph.as_default():
            self.a_grads = tf.gradients(loss_actor - loss_entropy, weights_a) 
            grads_var = list(zip(self.a_grads, weights_a))
            self.a_update = opt.apply_gradients(grads_var)
            self.c_grads = tf.gradients(loss_critic, weights_c) 
            grads_var = list(zip(self.c_grads, weights_c))
            self.c_update = opt.apply_gradients(grads_var)

    def run(self):
        self.train(5, 5000)
        print("It worked!")

    def train(self, t_max, episodes):
        global gtheta, gw, global_step_counter, rewards
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
                action = np.random.choice([1, 2, 3], p=probabilities[0])
                _state, reward, done, info = env.step(action)
                _state = resize(rgb2grayscale(_state) )
                _state = np.reshape(_state, (-1, _state.shape[0], _state.shape[1], 1))
                total_reward += reward
                memory.append([state, action, reward, done, value])
                state = _state
                self.step_counter += 1
                global_step_counter += 1
                repeat += 1
                # Update every 5th step
                if self.step_counter - t_start == t_max or done:
                    R = np.array([[0]]) if done else self.local_model.predict(state)[1]
                    memory = reversed(memory)
                    action_list = []
                    return_list = []
                    state_list  = []
                    advantages  = []
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
                        action_list.append(a_list)
                        if R != 0:
                            return_list.append(R[0])
                        else:
                            return_list.append(R)
                        s = np.reshape(s,  (s.shape[1], s.shape[2], 1))
                        state_list.append(s)
                        advantage = np.reshape(R - v, (-1, 1))
                        advantages.append(advantage[0])
                    sess.run(self.a_update, {self.s_t : state_list,
                                             self.a_t : action_list,
                                             self.r_t : return_list,
                                             self.advantage : np.array(advantages)})
                    sess.run(self.c_update, {self.s_t : state_list,
                                             self.a_t : action_list,
                                             self.r_t : return_list,
                                             self.advantage : np.array(advantages)})
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
            rewards.append(total_reward)
            print("Episode {} : Reward = {}".format(i, total_reward))
            if self.main:
                avg_reward = sum(rewards) / len(rewards)
                rewards = []
                print("T : ", global_step_counter, "Average reward : ", avg_reward)
            now = time.time()
            self.result_file.write("{},{},{}\n".format(i, total_reward, now - self.start_time))
        self.result_file.close()

            
        
agents = [Agent(NUM_STATE, NUM_ACTIONS, a, True) if a == 0 else Agent(NUM_STATE, NUM_ACTIONS, a) for a in range(NUM_THREADS)]

for a in agents:
    a.start()
    time.sleep(1)


for a in agents:
    a.join()

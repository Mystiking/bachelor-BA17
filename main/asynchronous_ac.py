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
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
import random
import time
import threading

# Constants
ALPHA = 0.3
BETA = 0.2
GAMMA = 0.9
lock = threading.Lock()
GAME_NAME = 'CartPole-v0'
# The environtment that we will be using
env = gym.make(GAME_NAME)
NUM_STATE = env.observation_space.shape[0]
NUM_ACTIONS = env.action_space.n
# For graphing
reward_list = []

# Global networks
input_layer = Input(shape=[NUM_STATE])
first_critic = Dense(8,activation='relu',kernel_initializer='uniform')(input_layer)
first_actor = Dense(8,activation='relu',kernel_initializer='uniform')(input_layer)
output_critic = Dense(1,activation='linear',kernel_initializer='uniform')(first_critic)
output_actor = Dense(NUM_ACTIONS,activation='softmax',kernel_initializer='uniform')(first_actor)
# The "final" networks
global_critic_model = Model(inputs=input_layer, outputs=output_critic)
global_actor_model = Model(inputs=input_layer, outputs=output_actor)
# Here the optimizer is irrelevant since we won't be using the Keras
# function "fit" to optimize the model
global_critic_model.compile(optimizer='sgd', loss='mse')
global_actor_model.compile(optimizer='sgd', loss='mse')
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
        self.m_theta = 0
        self.m_w = 0
        # Init layers of the model
        input_layer = Input(shape=[state_input_shape])
        first_critic = Dense(8,activation='relu',kernel_initializer='uniform')(input_layer)
        first_actor = Dense(8,activation='relu',kernel_initializer='uniform')(input_layer)
        output_critic = Dense(1,activation='linear',kernel_initializer='uniform')(first_critic)
        output_actor = Dense(action_output_shape,activation='softmax',kernel_initializer='uniform')(first_actor)
        # The "final" networks
        self.local_critic_model = Model(inputs=input_layer, outputs=output_critic)
        self.local_actor_model = Model(inputs=input_layer, outputs=output_actor)
        # Here the optimizer is irrelevant since we won't be using the Keras
        # function "fit" to optimize the model
        self.local_critic_model.compile(optimizer='sgd', loss='mse')
        self.local_actor_model.compile(optimizer='sgd', loss='mse')
        # Must be done to enable threading
        self.local_critic_model._make_predict_function()
        self.local_actor_model._make_predict_function()
        
        # Gradient function for calculating gradients of the critic model in this thread
        weights = self.local_critic_model.trainable_weights # weight tensors
        weights = [weight for weight in weights]
        weights = weights[::2]
        critic_gradients = self.local_critic_model.optimizer.get_gradients(self.local_critic_model.total_loss, weights) # gradient tensors
        
        critic_input_tensors = [self.local_critic_model.inputs[0], # input data
                                self.local_critic_model.sample_weights[0], # how much to weight each sample by
                                self.local_critic_model.targets[0], # labels
                                K.learning_phase()] # train or test mode

        self.get_critic_gradients = K.function(inputs=critic_input_tensors, outputs=critic_gradients)
        # Gradient function for calculating gradients of the actor model in this thread
        weights = self.local_actor_model.trainable_weights # weight tensors
        weights = [weight for weight in weights]
        weights = weights[::2]
        actor_gradients = self.local_actor_model.optimizer.get_gradients(self.local_actor_model.total_loss, weights) # gradient tensors
        
        actor_input_tensors = [self.local_actor_model.inputs[0], # input data
                         self.local_actor_model.sample_weights[0], # how much to weight each sample by
                         self.local_actor_model.targets[0], # labels
                         K.learning_phase(), # train or test mode
        ]
        
        self.get_actor_gradients = K.function(inputs=actor_input_tensors, outputs=actor_gradients)
    
    def run(self):
        self.train(200, 5000)
        print("It worked!")

    def train(self, t, episodes):
        i = 0
        env = gym.make(GAME_NAME)
        while i < episodes: 
            i += 1
            # Initialize weights of local networks
            with graph.as_default():
                start_weights_actor = global_actor_model.get_weights()
                self.local_actor_model.set_weights(start_weights_actor)
                start_weights_critic = global_critic_model.get_weights()
                self.local_critic_model.set_weights(start_weights_critic)
            total_reward = 0
            I = 1.
            memory = []
            state = env.reset()            
            state = state.reshape([1, NUM_STATE])
            done = False
            value = self.local_critic_model.predict(state)
            while not done:
                probabilities = self.local_actor_model.predict(state)
                action = np.random.choice(2, p=probabilities[0])
                _state, reward, done, info = env.step(action)
                _state = _state.reshape([1, NUM_STATE])
                _value = self.local_critic_model.predict(_state)
                _value = _value if not done else np.array([[0.]])
                total_reward += reward
                td_error = reward + GAMMA * _value - value
                memory.append([state, value, probabilities, action, td_error])
                # Critic grads for this state and value

                critic_inputs = [state, [1], value, 0]
                critic_grads = self.get_critic_gradients(critic_inputs)
                # Actor grads for this state and log probabilty
                log_prob = np.array([np.array([np.log(probabilities[0][action] + 1e-7)])])
                actor_inputs = [state, [1], log_prob, 0]
                actor_grads = self.get_actor_gradients(actor_inputs)
                # Updating gradients
                #if first:
                #    total_actor_grads = ALPHA * I * td * actor_grads
                #    total_critic_grads = BETA * td * critic_grads
                #    first = False
                #else:
                #    total_actor_grads += ALPHA * I * td * actor_grads
                #    total_critic_grads += BETA * td * critic_grads
                j = 0
                while j < len(actor_grads):
                    # Critic
                    update_critic = self.local_critic_model.layers[j+1].get_weights()[0] + BETA * td_error * critic_grads[j]
                    bias_critic = self.local_critic_model.layers[j+1].get_weights()[1]
                    self.local_critic_model.layers[j+1].set_weights((update_critic, bias_critic))
                    # Actor
                    update_actor = self.local_actor_model.layers[j+1].get_weights()[0] + ALPHA * I * td_error * actor_grads[j]
                    bias_actor = self.local_actor_model.layers[j+1].get_weights()[1]
                    self.local_actor_model.layers[j+1].set_weights((update_actor, bias_actor))
                    j += 1
                I = I * GAMMA
                state = _state
                value = _value
            print("Episode {} : Reward = {}".format(i, total_reward))
            reward_list.append(total_reward)
            lock.acquire()
            with graph.as_default():
                delta_w = np.array(self.local_critic_model.get_weights()) -  np.array(start_weights_critic)
                delta_theta = np.array(self.local_actor_model.get_weights()) - np.array(start_weights_actor)
                global_critic_model.set_weights(global_critic_model.get_weights() + delta_w)
                global_actor_model.set_weights(global_actor_model.get_weights() + delta_theta)
            lock.release()
            # Computing gradients
            # "hack" to get correct shape of grads
            '''
            for s, v, p, a, td in memory:
                # Critic grads for this state and value
                critic_inputs = [s, [1], v, 0]
                critic_grads = self.get_critic_gradients(critic_inputs)
                # Actor grads for this state and log probabilty
                log_prob = np.array([np.array([np.log(p[0][a] + 1e-7)])])
                actor_inputs = [s, [1], log_prob, 0]
                actor_grads = self.get_actor_gradients(actor_inputs)
                # Updating gradients
                j = 0
                while j < len(actor_grads):
                    # Critic
                    update_critic = self.local_critic_model.layers[j+1].get_weights()[0] + BETA * td_error * critic_grads[0][j]
                    bias_critic = self.local_critic_model.layers[j+1].get_weights()[1]
                    self.local_critic_model.layers[j+1].set_weights((update_critic, bias_critic))
                    # Actor
                    update_actor = self.local_actor_model.layers[j+1].get_weights()[0] + ALPHA * I * td * actor_grads[0][j]
                    bias_actor = self.local_actor_model.layers[j+1].get_weights()[1]
                    self.local_actor_model.layers[j+1].set_weights((update_actor, bias_actor))
                I = I * GAMMA
                
            lock.acquire()
            # Updating the global parameters
            with graph.as_default():
                # Updating critic model
                j = 0
                while j < len(total_critic_grads):
                    self.m_w = alpha * self.m_w + (1. - alpha) * total_critic_grads[0][j]
                    delta_w = global_critic_model.layers[j+1].get_weights()[0] - LR * self.m_w
                    bias_w = global_critic_model.layers[j+1].get_weights()[1]
                    global_critic_model.layers[j+1].set_weights((delta_w, bias_w))
                    j += 1
                # Updating actor model
                j = 0
                while j < len(total_actor_grads):
                    self.m_theta = alpha * self.m_theta + (1. - alpha) * total_actor_grads[0][j]
                    delta_theta = global_actor_model.layers[j+1].get_weights()[0] - LR * self.m_theta
                    bias_theta = global_actor_model.layers[j+1].get_weights()[1]
                    global_actor_model.layers[j+1].set_weights((delta_theta, bias_theta))
                    j += 1
            lock.release()
'''

agent006 = Agent(NUM_STATE, NUM_ACTIONS)
#agent007 = Agent(NUM_STATE, NUM_ACTIONS)
#agent008 = Agent(NUM_STATE, NUM_ACTIONS)
#agent009 = Agent(NUM_STATE, NUM_ACTIONS)

agent006.start()
#agent007.start()
#agent008.start()
#agent009.start()

agent006.join()
#agent007.join()
#agent008.join()
#agent009.join()

c = 0
k = 100
avg_list = []
for i in range(len(reward_list) - 1):
    if c % k == 0:
        avg_list.append(sum(reward_list[i + c : i + c + k]) / k)

print(sum(reward_list) / len(reward_list))
plt.plot(range(len(reward_list)), reward_list, 'r')
plt.plot(range(len(avg_list)), avg_list, 'b')
plt.show()


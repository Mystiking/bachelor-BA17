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
ALPHA = 0.5
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

# Global networks
input_layer = Input(shape=[NUM_STATE])
first_critic = Dense(8,activation='relu',kernel_initializer='uniform')(input_layer)
first_actor = Dense(8,activation='relu',kernel_initializer='uniform')(input_layer)
output_critic = Dense(1,activation='linear',kernel_initializer='uniform')(first_critic)
output_actor = Dense(NUM_ACTIONS,activation='softmax',kernel_initializer='uniform')(first_actor)
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

start_time = time.time()

class Agent(threading.Thread):
    def __init__(self, state_input_shape, action_output_shape, agent_id):
        threading.Thread.__init__(self)
        self.results = open('a3c_run_1_agent_{}.csv'.format(agent_id), 'w')
        self.step_counter = 1
        self.gtheta = 0
        self.gw = 0
        # Init layers of the model
        input_layer = Input(shape=[state_input_shape])
        first_critic = Dense(8,activation='relu',kernel_initializer='uniform')(input_layer)
        first_actor = Dense(8,activation='relu',kernel_initializer='uniform')(input_layer)
        output_critic = Dense(1,activation='linear',kernel_initializer='uniform')(first_critic)
        output_actor = Dense(action_output_shape,activation='softmax',kernel_initializer='uniform')(first_actor)
        # The "final" networks
        self.local_critic_model = Model(inputs=input_layer, outputs=output_critic)
        self.local_actor_model = Model(inputs=input_layer, outputs=output_actor)
        # Must be done to enable threading
        self.local_critic_model._make_predict_function()
        self.local_actor_model._make_predict_function()
        
        self.s_t = tf.placeholder(tf.float32, shape=(None, NUM_STATE))
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
        self.train(5, 2000)
        print("It worked!")

    def train(self, t_max, episodes):
        i = 0
        t_start = self.step_counter
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
            state = state.reshape([1, NUM_STATE])
            done = False
            value = self.local_critic_model.predict(state)
            while not done:
                probabilities = self.local_actor_model.predict(state)
                #print(probabilities)
                action = np.random.choice(2, p=probabilities[0])
                _state, reward, done, info = env.step(action)
                _state = _state.reshape([1, NUM_STATE])
                _value = self.local_critic_model.predict(_state)
                _value = _value if not done else np.array([[0.]])
                total_reward += reward
                #td_error = reward + GAMMA * _value - value
                memory.append([state, action, reward, done])
                state = _state
                value = _value
                self.step_counter += 1
                if done or self.step_counter - t_start == t_max:
                    R = 0 if done else self.local_critic_model.predict(state)
                    memory = reversed(memory)
                    for s, a, r, d in memory:
                        R = r + GAMMA * R
                        a_list = np.zeros(NUM_ACTIONS)
                        a_list[a] = 1.
                        a_list = np.reshape(a_list, (-1, 2))
                        actor_grads = sess.run(self.a_grads, {self.s_t : s, self.a_t : a_list, self.r_t : np.reshape(np.array([R]), (-1, 1))})
                        dtheta = dtheta + np.array(actor_grads)

                        critic_grads = sess.run(self.c_grads, {self.s_t : s, self.r_t : np.reshape(np.array([R]), (-1, 1))})
                        dw = dw + np.array(critic_grads)
                    lock.acquire()
                    with graph.as_default():
                        self.gtheta = self.gtheta * ALPHA + (1 - ALPHA) * (dtheta**2)
                        self.gw = self.gw * ALPHA + (1 - ALPHA) * (dw**2)
                        j = 0
                        while j < len(dtheta):
                            # Critic
                            update_critic = global_critic_model.layers[j+1].get_weights()[0] - LR * dw[j] / np.sqrt(self.gw[j] + EPSILON)
                            bias_critic = global_critic_model.layers[j+1].get_weights()[1]
                            global_critic_model.layers[j+1].set_weights((update_critic, bias_critic))
                            # Actor
                            update_actor = global_actor_model.layers[j+1].get_weights()[0] - LR * dtheta[j] / np.sqrt(self.gtheta[j] + EPSILON)
                            bias_actor = global_actor_model.layers[j+1].get_weights()[1]
                            global_actor_model.layers[j+1].set_weights((update_actor, bias_actor))
                            j += 1
                    lock.release()
                    dw = 0
                    dtheta = 0
                    t_start = self.step_counter
                    memory = []

            now = time.time()
            self.results.write(str(i) + ',' + str(total_reward) + ',' + str(now - start_time) + '\n')
            reward_list.append(total_reward)
            
        
agent006 = Agent(NUM_STATE, NUM_ACTIONS, 1)
agent007 = Agent(NUM_STATE, NUM_ACTIONS, 2)
agent008 = Agent(NUM_STATE, NUM_ACTIONS, 3)
agent009 = Agent(NUM_STATE, NUM_ACTIONS, 4)

agent006.start()
agent007.start()
agent008.start()
agent009.start()

agent006.join()
agent007.join()
agent008.join()
agent009.join()

#c = 0
#k = 500
#avg_list = []
#for i in range(len(reward_list) - 1):
#    if c % k == 0:
#        avg_list.append(sum(reward_list[i + c : i + c + k]) / k)

#print(sum(reward_list) / len(reward_list))
#plt.plot(range(len(reward_list)), reward_list, 'r')
#plt.plot(range(len(avg_list)), avg_list, 'b')
#plt.xlabel('Episodes')
#plt.ylabel('Score')
#plt.title('Score of the a3c algorithm over time')
#plt.show()


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

flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_integer('num_threads', 1, 'Number of threads')

NUM_THREADS = FLAGS.num_threads 
for run in range(10):
    # Constants
    ALPHA = 0.9
    GAMMA = 0.90
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
    first= Dense(8, activation='relu', kernel_initializer='uniform')(input_layer)
    #second = Dense(8, activation='relu', kernel_initializer='uniform')(first)
    # Output layers
    output_actor  = Dense(units=NUM_ACTIONS, activation='softmax', kernel_initializer='uniform')(first)
    output_critic = Dense(units=1, activation=None, kernel_initializer='uniform')(first)
    # The "final" networks
    global_model = Model(inputs=input_layer, outputs=[output_actor, output_critic])
    # Must be done to enable threading
    global_model._make_predict_function()
    
    opt = tf.train.RMSPropOptimizer(LR, decay=ALPHA, epsilon=EPSILON)
    
    # Initialize tf global variables
    sess = tf.Session()
    K.set_session(sess)
    init = tf.global_variables_initializer()
    sess.run(init)
    graph = tf.get_default_graph()
    
    
    class Agent(threading.Thread):
        def __init__(self, state_input_shape, action_output_shape, threadid, main = False):
            self.result_file = open("cartpole_{}_threads/cartpole_agent_{}_results_{}.csv".format(NUM_THREADS, threadid, run), 'w')
            self.start_time = time.time()
            threading.Thread.__init__(self)
            self.step_counter = 0
            # Init layers of the model
            input_layer      = Input(shape=[4])
            # First conv layer
            first= Dense(8, activation='relu', kernel_initializer='uniform')(input_layer)
            #second= Dense(8, activation='relu', kernel_initializer='uniform')(first)
            # Output layers
            output_actor  = Dense(units=NUM_ACTIONS, activation='softmax', kernel_initializer='uniform')(first)
            output_critic = Dense(units=1, activation=None, kernel_initializer='uniform')(first)
            # The "final" networks
            self.local_model = Model(inputs=input_layer, outputs=[output_actor, output_critic])
            # Must be done to enable threading
            self.local_model._make_predict_function()
            
            self.s_t = tf.placeholder(tf.float32, shape=(None, NUM_STATE))
            self.a_t = tf.placeholder(tf.float32, shape=(None, NUM_ACTIONS), name="actions")
            self.r_t = tf.placeholder(tf.float32, shape=(None, 1), name="reward")
            self.advantage = tf.placeholder(tf.float32, shape=(None, 1), name="advantages")

            p, v = self.local_model(self.s_t)
            #self.advantage = (self.r_t - v)
            
            weights = self.local_model.trainable_weights[::2] # weight tensors
            # Actor weights
            weights_a = [weights[0], weights[1]]#, weights[2]]
            weights_c = [weights[0], weights[2]]#, weights[3]]
            # "Losses"
 
            log_p = tf.log(tf.clip_by_value(p, 0.000001, 0.999999))
            loss_actor = -tf.reduce_mean(tf.reduce_sum(tf.multiply(log_p, self.a_t), reduction_indices=1) * self.advantage)
            loss_critic = 0.5 * tf.reduce_mean(tf.square(self.r_t - v))
            loss_entropy = 0.01 * tf.reduce_sum(tf.multiply(p, -log_p))
            total_loss = (loss_actor - loss_entropy + loss_critic)

            with graph.as_default():
                global_weights = global_model.trainable_weights[::2]
                global_actor_weights = [global_weights[0], global_weights[1]]#, global_weights[2]]
                global_critic_weights =[global_weights[0], global_weights[2]]#, global_weights[3]]

                #self.grads = tf.gradients(total_loss, weights)
                #grads_var = list(zip(self.grads, global_weights))
                #self.update = opt.apply_gradients(grads_var)
                self.a_grads = tf.gradients(loss_actor - loss_entropy, weights_a) 
                grads_var = list(zip(self.a_grads, global_actor_weights))
                self.a_update = opt.apply_gradients(grads_var)
                self.c_grads = tf.gradients(loss_critic, weights_c) 
                grads_var = list(zip(self.c_grads, global_critic_weights))
                self.c_update = opt.apply_gradients(grads_var)
    
    
        def run(self):
            self.train(5, 8000)
    
        def train(self, t_max, episodes):
            global gtheta, gw, LR
            i = 0
            env = gym.make(GAME_NAME)
            while i < episodes: 
    
                i += 1
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
                state = state.reshape([1, NUM_STATE])
                done = False
                while not done:
                    probabilities, value = self.local_model.predict(state)
                    action = np.random.choice(2, p=probabilities[0])
                    _state, reward, done, info = env.step(action)
                    _state = _state.reshape([1, NUM_STATE])
                    total_reward += reward
                    memory.append([state, action, reward, done, value])
                    state = _state
                    self.step_counter += 1
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
                            a_list = np.zeros(NUM_ACTIONS)
                            a_list[a] = 1
                            action_list.append(a_list) 
                            if R != 0:
                                return_list.append(R[0])
                            else:
                                return_list.append(R)
                            s = s[0]
                            state_list.append(s)
                            advantage = np.reshape(R - v, (-1, 1))
                            advantages.append(advantage[0])
                        state_list  = state_list[::-1]
                        action_list = action_list[::-1]
                        return_list = return_list[::-1]
                        advantages  = advantages[::-1]
                        #sess.run(self.update, {self.s_t : state_list,
                        #                       self.a_t : action_list,
                        #                       self.r_t : return_list,
                        #                       self.advantage : advantages})
                        sess.run(self.a_update, {self.s_t : state_list,
                                                 self.a_t : action_list,
                                                 self.r_t : return_list,
                                                 self.advantage : np.array(advantages)})
                        sess.run(self.c_update, {self.s_t : state_list,
                                                 self.a_t : action_list,
                                                 self.r_t : return_list,
                                                 self.advantage : np.array(advantages)})


                        with graph.as_default():
                            start_weights = global_model.get_weights()
                            self.local_model.set_weights(start_weights)
                        t_start = self.step_counter
                        memory = []
                print("Episode : {}, Reward : {}".format(i, total_reward))
                now = time.time()
                self.result_file.write("{},{},{}\n".format(i, total_reward, now - self.start_time))
            self.result_file.close()
    
                
    agents = [Agent(NUM_STATE, NUM_ACTIONS, a) for a in range(NUM_THREADS)]
  
    for a in agents:
        a.start()
        time.sleep(1)

    for a in agents:
        a.join()


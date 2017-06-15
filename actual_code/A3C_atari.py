import gym
from scipy.misc import imresize
import numpy as np
from keras.models import Model
from keras.layers import Dense, Input, Convolution2D, Flatten
from keras import backend as K
import tensorflow as tf
import time
import threading
# Avoid annoying error messages about CPU instructions
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# Makes it easier to assign number of threads/episodes to a run
flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_integer('num_threads', 1, 'Number of threads')
flags.DEFINE_integer('max_steps', 1, 'Number of max steps')

'''
@param img: An input image
@return: A resized image (with values between 0 and 1)
'''
def resize(img):
    return imresize(img, (84, 84)).astype('float32') * (1./255.)

'''
@param img: An input image
@return: The (naive) grayscale of the image
'''
def rgb2grayscale(img):
    return img.astype('float32').mean(2)

# Used to keep track of "mean" rewards/values for informative printing
rewards = []
values = []
# Global timesteps
global_step_counter = 0
# Constants
ALPHA = 0.99
GAMMA = 0.99
LR = 0.0001
EPSILON = 1e-8
GAME_NAME = 'SpaceInvaders-v0'
# The environtment that we will be using.
# It is initalized such that "NUM_STATE" can be assigned.
env = gym.make(GAME_NAME)
NUM_STATE = env.observation_space.shape
# NUM_ACTIONS is 3 for all games we will be playing
NUM_ACTIONS = 3
NUM_THREADS = FLAGS.num_threads 
NUM_FRAMES = 4
RESIZED_WIDTH = 84
RESIZED_HEIGHT = 84

# Tensorflow session : needed to actually "run" the computational graphs
sess = tf.Session()
# Make it the default Keras session
K.set_session(sess)

######################
# The Global Network #
######################
# Input layer - "output" shape : 84 x 84 x 4
input_layer = Input(shape=(RESIZED_WIDTH, RESIZED_HEIGHT, NUM_FRAMES))
# Convolutional layer - output shape : 16 x 21 x 21
first = Convolution2D(16, kernel_size=(8, 8), strides=[4, 4], padding='same', use_bias=0, activation='relu', kernel_initializer='uniform')(input_layer)
# Convolutional layer - output shape : 32 x 11 x 11
second = Convolution2D(32, kernel_size=(4, 4), strides=[2, 2], padding='same', use_bias=0, activation='relu', kernel_initializer='uniform')(first)
# "Flatten layer" - output shape : 3872 x 1
flat = Flatten()(second)
# Dense layer - output shape : 256 x 1
dense = Dense(units=256, activation='relu', kernel_initializer='uniform')(flat)
# Output layers
# Policy output shape : NUM_ACTIONS x 1
output_actor  = Dense(units=NUM_ACTIONS, activation='softmax', kernel_initializer='uniform')(dense)
# Value estimate output shape : 1 x 1
output_critic = Dense(units=1, activation=None, kernel_initializer='uniform')(dense)
# Initializing the global model
global_model = Model(inputs=input_layer, outputs=[output_actor, output_critic])
# Must be done to enable threading
global_model._make_predict_function()
# Define the global RMSPropOptimizer
opt = tf.train.RMSPropOptimizer(LR, decay=ALPHA, epsilon=EPSILON)
 
# Initialize global variables
init = tf.global_variables_initializer()
sess.run(init)
# "graph" is needed to use the global model
graph = tf.get_default_graph()
# Timing the runs
start_time = time.time()

'''
@class Actor: An actor in the Actor-Critic model,
              but also containing the environment
'''
class Actor(threading.Thread):
    def __init__(self, state_input_shape, action_output_shape, threadid, main = False):
        threading.Thread.__init__(self)

        self.num_frames = NUM_FRAMES
        # Where to save the results of the run
        self.result_file = open("{}_agent_{}_results_{}.csv".format(GAME_NAME, threadid, FLAGS.num_threads), 'w')
        # Only one actor should be main - used to print
        self.main = main
        # Used to create the 4-block of states
        self.state = None
        # Determines when the program should perform an asynchronous update
        self.step_counter = 0
        # Input layer - "output" shape : 84 x 84 x 4
        input_layer = Input(shape=(RESIZED_WIDTH, RESIZED_HEIGHT, NUM_FRAMES))
        # Convolutional layer - output shape : 16 x 21 x 21
        first = Convolution2D(16, kernel_size=(8, 8), strides=[4, 4], padding='same', use_bias=0, activation='relu', kernel_initializer='uniform')(input_layer)
        # Convolutional layer - output shape : 32 x 11 x 11
        second = Convolution2D(32, kernel_size=(4, 4), strides=[2, 2], padding='same', use_bias=0, activation='relu', kernel_initializer='uniform')(first)
        # "Flatten layer" - output shape : 3872 x 1
        flat = Flatten()(second)
        # Dense layer - output shape : 256 x 1
        dense = Dense(units=256, activation='relu', kernel_initializer='uniform')(flat)
        # Output layers
        # Policy output shape : NUM_ACTIONS x 1
        output_actor  = Dense(units=NUM_ACTIONS, activation='softmax', kernel_initializer='uniform')(dense)
        # Value estimate output shape : 1 x 1
        output_critic = Dense(units=1, activation=None, kernel_initializer='uniform')(dense)
        # The "final" networks
        self.local_model = Model(inputs=input_layer, outputs=[output_actor, output_critic])
        # Must be done to enable threading
        self.local_model._make_predict_function()
        
        # TF placeholder for a state
        self.s_t = tf.placeholder(tf.float32, shape=(None, RESIZED_WIDTH, RESIZED_HEIGHT, NUM_FRAMES), name="states")
        # TF placeholder for the action taken (as a 1-hot array) 
        self.a_t = tf.placeholder(tf.float32, shape=(None, NUM_ACTIONS), name="actions")
        # TF placeholder for a reward
        self.r_t = tf.placeholder(tf.float32, shape=(None, 1), name="reward")
        # TF placeholder for an advantage
        self.advantage = tf.placeholder(tf.float32, shape=(None, 1), name="advantages")
        # "Predictions" of the probabilities and value
        p, v = self.local_model(self.s_t)

        # The weights of the local model
        weights = self.local_model.trainable_weights
        # Actor weights
        weights_a = [weights[0], weights[1], weights[2], weights[3]]
        # Critic weights
        weights_c = [weights[0], weights[1], weights[2], weights[4]]
 
        # Updating the weights of the global model
        # The logarithm of the probabilities
        log_p = tf.log(tf.clip_by_value(p, 0.000001, 0.999999))
        # The policy times the experienced advantage, -loss_actor to make the loss positive
        loss_actor = -tf.reduce_mean(tf.reduce_sum(tf.multiply(log_p, self.a_t), reduction_indices=1) * self.advantage)
        # The squared advantage (weighted) 
        loss_critic = 0.5 * tf.reduce_mean(tf.square(self.r_t - v))
        # The entropy (weighted), -log_p to make the entropy positive
        loss_entropy = 0.01 * tf.reduce_sum(tf.multiply(p, -log_p))
        
        # Computing the gradients and performing the asynchronous update
        with graph.as_default():
            # Weights of the global model
            global_weights = global_model.trainable_weights
            global_actor_weights = [global_weights[0], global_weights[1], global_weights[2], global_weights[3]]
            global_critic_weights = [global_weights[0], global_weights[1], global_weights[2], global_weights[4]]
            # Computing and applying the actor gradients
            actor_gradients = tf.gradients(loss_actor - loss_entropy, weights_a) 
            actor_gradients, _ = tf.clip_by_global_norm(actor_gradients, 40.0)
            grads_var = list(zip(actor_gradients, global_actor_weights))
            self.policy_update = opt.apply_gradients(grads_var)
            # Computing and applying the critic gradients
            critic_gradients = tf.gradients(loss_critic, weights_c) 
            critic_gradients, _ = tf.clip_by_global_norm(critic_gradients, 40.0)
            grads_var = list(zip(critic_gradients, global_critic_weights))
            self.critic_update = opt.apply_gradients(grads_var)

    '''
    @void: Starts the training process for "FLAGS.max_steps" episodes
    '''
    def run(self):
        self.train(5, FLAGS.max_steps)

    '''
    @param img: A frame from an Atari game
    @param start: Whether or not this is the first state
    @return: A 4-stacked representation of the last 4 states
    '''
    def preprocess(self, img, start = None):
        gray = rgb2grayscale(img) 
        s = resize(gray)
        s = s.reshape(1, s.shape[0], s.shape[1], 1)
        if start or self.state is None:
            self.state = np.repeat(s, self.num_frames, axis=3)
        else:
            self.state = np.append(s, self.state[:,:,:,:self.num_frames-1], axis=3)
        return self.state

    '''
    @param t_max: Amount of steps to perform before updating
    @param steps: How many steps to play  
    '''
    def train(self, t_max, episodes):
        global global_step_counter, rewards, values, start_time
        # Create an environment
        env = gym.make(GAME_NAME)
        now = time.time()
        # Run for at most "episodes" iterations or 
        while global_step_counter < episodes and now - start_time < 57600.0: 
            with graph.as_default():
                # Initialize weights of local networks
                start_weights = global_model.get_weights()
                self.local_model.set_weights(start_weights)
            # Used for printing
            total_reward = 0
            aliens_killed = 0
            t_start = self.step_counter
            memory = []
            state = env.reset()   
            state = self.preprocess(state, True)
            done = False
            while not done:
                probabilities, value = self.local_model.predict(state)
                values.append(value)
                action = np.random.choice([1, 2, 3], p=probabilities[0])
                reward = 0
                for j in range(NUM_FRAMES):
                        _state, one_reward, done, info = env.step(action)
                        reward += one_reward
                        if done:
                            break
                        prev_state = _state
                total_reward += reward
                reward = np.clip(reward, -1, 1)
                aliens_killed += reward
                if GAME_NAME == 'SpaceInvaders-v0':
                    _state = np.maximum.reduce([prev_state, _state])
                _state = self.preprocess(_state)
                memory.append([state, action, reward, done, value])
                state = _state
                self.step_counter += 1
                global_step_counter += 1
                if global_step_counter >= episodes:
                    break
                # Update every t_max step
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
                            return_list.append(R[0])
                        s = np.reshape(s,  (s.shape[1], s.shape[2], 4))
                        state_list.append(s)
                        adv = np.reshape(R - v, (-1, 1))
                        advantages.append(adv[0])
                    # Re-reverse the lists
                    state_list  = state_list[::-1]
                    action_list = action_list[::-1]
                    return_list = return_list[::-1]
                    advantages  = advantages[::-1]
                    # Update gradients
                    sess.run(self.policy_update, {self.s_t : state_list,
                                      self.a_t : action_list,
                                      self.r_t : return_list,
                                      self.advantage : advantages})
                    sess.run(self.critic_update, {self.s_t : state_list,
                                      self.a_t : action_list,
                                      self.r_t : return_list,
                                      self.advantage : advantages})
                    t_start = self.step_counter
                    memory = []
                    with graph.as_default():
                        # Reset weights to global weights
                        start_weights = global_model.get_weights()
                        self.local_model.set_weights(start_weights)
            rewards.append(total_reward)
            # Print summary of performance 
            if self.main:
                avg_reward = sum(rewards) / len(rewards)
                avg_vals = sum(values) / len(values)
                rewards = []
                values = []
                print("Timesteps:", global_step_counter, "Average reward:", avg_reward, "Probs:", probabilities, "Avg val:", avg_vals, "Aliens killed:", aliens_killed)
            now = time.time()
            # Write to result file
            self.result_file.write("{},{},{},{}\n".format(global_step_counter, total_reward, now - start_time, aliens_killed))
        self.result_file.close()

            
# Make one actor for each NUM_THREAD 
actors = [Actor(NUM_STATE, NUM_ACTIONS, a, True) if a == 0 else Actor(NUM_STATE, NUM_ACTIONS, a) for a in range(NUM_THREADS)]

for a in actors:
    a.start()
    time.sleep(1)

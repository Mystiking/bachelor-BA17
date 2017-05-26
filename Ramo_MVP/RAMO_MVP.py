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




# The environtment that we will be using
GAME_NAME = 'Breakout-v0'
env = gym.make(GAME_NAME)
NUM_STATE = env.observation_space.shape
NUM_ACTIONS = 3#env.action_space.n
RESIZED_WIDTH = int(NUM_STATE[0] / 2)
RESIZED_HEIGHT = int(NUM_STATE[1] / 2)


#########################################
#       METHODS USED FOR PREPROCESING   #
#########################################
def resize(img):
    return img[::2,::2]

def rgb2grayscale(img):
    return np.dot(img[:,:,:3], [0.2126, 0.7152, 0.0722])




class Agent(threading.Thread):
    def __init__(self, state_input_shape, action_output_shape, threadid):
        optimizer = tf.train.AdamOptimizer(1e-4)

        self.result_file = open("breakout_agent_{}_results.csv".format(threadid), 'w')
        self.start_time = time.time()
        threading.Thread.__init__(self)
        self.step_counter = 0
        self.optimizer = optimizer


        with tf.variable_scope('network'):
            self.action = tf.placeholder('int32', [None], name='action')
            self.target_value = tf.placeholder('float32', [None], name='target_value')
            self.state, self.policy, self.value = self.build_model(84, 84, 4)
            self.weights = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='network')
            self.advantages = tf.placeholder('float32', [None], name='advantages')


        with tf.variable_scope('optimizer'):
            action_one_hot = tf.one_hot(self.action, self.action_size, 1.0, 0.0)
            self.log_policy = tf.log(tf.clip_by_value(self.policy, 0.000001, 0.999999))
            self.log_pi_for_action = tf.reduce_sum(tf.multiply(self.log_policy, action_one_hot), reduction_indices=1)


            self.policy_loss = -tf.reduce_mean(self.log_pi_for_action * self.advantages)
            self.value_loss = tf.reduce_mean(tf.square(self.target_value - self.value))
            self.entropy = tf.reduce_sum(tf.multiply(self.policy, -self.log_policy))
            self.loss = 0.5 * self.value_loss + self.policy_loss - self.entropy * 0.01

            grads = tf.gradients(self.loss, self.weights)
            grads, _ = tf.clip_by_global_norm(grads, 40.0)
            grads_vars = list(zip(grads, self.weights))

            self.train_op = optimizer.apply_gradients(grads_vars)










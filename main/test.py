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

def rgb2grayscale(img):
    return np.dot(img[:,:,:3], [0.2126, 0.7152, 0.0722])

def resize(img):
    return img[::2, ::2]


episodes = 1
GAME_NAME = 'Breakout-v0'
i = 0
env = gym.make(GAME_NAME)
NUM_STATE = env.observation_space.shape
NUM_ACTIONS = env.action_space.n
print(env.observation_space.shape, NUM_ACTIONS)
state = env.reset()   
#plt.imshow(state)
#plt.show()

while i < episodes: 
    i += 1
    total_reward = 0
    I = 1.
    state = env.reset()   
    rgbState = rgb2grayscale(state)
    done = False
    c = 0
    while not done:
        env.render()
        action = 4#env.action_space.sample() 
        _state, reward, done, info = env.step(action)
        total_reward += reward
        state = _state
        c += 1
        if c == 300:
            break
'''            
    print("Episode {} : Reward = {}".format(i, total_reward))
plt.imshow(state)
plt.show()
state = rgb2grayscale(state)
plt.imshow(state, cmap='Greys')
plt.show()
state = resize(state)
plt.imshow(state, cmap='Greys')
plt.show()
'''

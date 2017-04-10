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
import tensorflow as tf
import random
import time

def train_critic(td_error):
    weights = critic_model.layers[1].weights
    gradients = critic_model.optimizer.get_gradients(critic_model.total_loss, weights) 
    w = weights
    w = w + beta * td_error * gradients
    critic_model.layers[1].W = w

def train_actor(td_error):
    weights = actor_model.layers[1].weights
    gradients = actor_model.optimizer.get_gradients(actor_model.total_loss, weights)
    theta = weights
    theta = theta + alpha * I * td_error * gradients
    actor_model.layers[1].W = theta
    


input = Input(shape=[4])
value = Dense(1, activation='linear')(input)
probs = Dense(2, activation='softmax')(input)
critic_model = Model(inputs=input, outputs=value)
critic_model.compile(optimizer='sgd', loss='mse')

actor_model = Model(inputs=input, outputs=probs)
actor_model.compile(optimizer='sgd', loss='mse')

# Contants
episodes = 200
gamma = 0.9
alpha = 0.1
beta = 0.2
I = 1.

env = gym.make('CartPole-v0')

for i in range(episodes):
    total_reward = 0
    state = env.reset()
    state = state.reshape([1, 4])
    done = False
    while (not done):
        before = (time.time())
        probabilities = actor_model.predict(state)
        after = (time.time())

        print("Time to pred {}".format(after - before))
        action = np.random.choice(2, p=probabilities[0])
        _state, reward, done, info = env.step(action)
        _state = _state.reshape([1, 4])
        total_reward += reward
        value = critic_model.predict(state)
        _value = critic_model.predict(_state) if not done else 0
        td_error = reward + gamma * _value - value
        train_critic(td_error)
        train_actor(td_error)
        I = I * gamma
        state = _state


    print("Episode", i, " - reward :",total_reward)


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


def train_critic_old(td_error):
    weights = critic_model.optimizer.weights
    gradients = critic_model.optimizer.get_gradients(critic_model.total_loss, weights) 
    w = weights
    w = w + beta * td_error * gradients
#    critic_model.layers[1].W = w
    critic_model.optimizer.set_weights(w)

def train_actor_old(td_error):
    #weights = actor_model.layers[1].weights
    #gradients = actor_model.optimizer.compute_gradients(gradients_actor)
    weights = actor_model.optimizer.weights
    gradients = actor_model.optimizer.get_gradients(actor_model.total_loss, weights)
    print(type(gradients))
#    gradients = K.gradients(actor_model.total_loss, actor_model.optimizer.weights)
    theta = weights + alpha * I * td_error * gradients

def train_critic():
    # Tensorflow placeholders for I, alpha and td_error
    tf_beta = tf.placeholder(tf.float32, shape=(None, 1))
    tf_td_error = tf.placeholder(tf.float32, shape=(None, 1))
    
    weights = critic_model.layers[1].weights[0]
    gradients = tf.gradients(critic_model.total_loss, weights) 

    w = weights + tf_beta * tf_td_error * gradients
    critic_model.layers[1].weights[0] = w
    return tf_beta, tf_td_error, critic_model.layers[1].weights[0]

def train_actor():
    # Tensorflow placeholders for I, alpha and td_error
    tf_I = tf.placeholder(tf.float32, shape=(None, 1))
    tf_alpha = tf.placeholder(tf.float32, shape=(None, 1))
    tf_td_error = tf.placeholder(tf.float32, shape=(None, 1))

    weights = actor_model.layers[1].weights[0]
    gradients = tf.gradients(actor_model.total_loss, weights)
    
    theta = weights + tf_alpha * tf_I * tf_td_error * gradients
    actor_model.layers[1].weights[0] = theta
    return tf_I, tf_alpha, tf_td_error, actor_model.layers[1].weights[0]
    
sess = tf.Session()
K.set_session(sess)

input = Input(shape=[4])
value = Dense(8, activation='relu')(input)
value_2 = Dense(1, activation='linear')(value)

probs = Dense(8, activation='relu')(input)
probs_2 = Dense(2, activation='softmax')(probs)

critic_model = Model(inputs=input, outputs=value_2)
critic_model.compile(optimizer='sgd', loss='mse')

actor_model = Model(inputs=input, outputs=probs_2)
actor_model.compile(optimizer='sgd', loss='mse')

# Contants
episodes = 2000
gamma = 0.9
alpha = 0.1
beta = 0.2
I = 1.
# TF symbolic graphs
actor_graph = train_actor()
critic_graph = train_critic()

env = gym.make('CartPole-v0')
reward_list = []

for i in range(episodes):
    total_reward = 0
    state = env.reset()
    state = state.reshape([1, 4])
    done = False
    value = critic_model.predict(state)
    while (not done):
        probabilities = actor_model.predict(state)
        action = np.random.choice(2, p=probabilities[0])
        _state, reward, done, info = env.step(action)
        _state = _state.reshape([1, 4])
        total_reward += reward
        _value = critic_model.predict(_state) if not done else 0
        td_error = reward + gamma * _value - value
        # Training the critic model
        tf_beta, tf_td_error, w = critic_graph
        sess.run(w, feed_dict={tf_beta : np.reshape(np.array(beta), (-1, 1)), tf_td_error : np.reshape(np.array(td_error), (-1, 1))})
        # Training the actor model
        tf_I, tf_alpha, tf_td_error, theta = actor_graph
        sess.run(theta, feed_dict={tf_I : np.reshape(np.array(I), (-1, 1)), tf_alpha: np.reshape(np.array(alpha), (-1, 1)) , tf_td_error : np.reshape(np.array(td_error), (-1, 1))})
        
        I = I * gamma
        state = _state
        value = _value
    reward_list.append(total_reward)
    print("Episode", i, " - reward :",total_reward)

k = 50
c = 0
#for r in range(len(reward_list)):
#    c += r
print(sum(reward_list) / len(reward_list))
plt.plot(range(len(reward_list)), reward_list, 'r')
plt.show()

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

def train_critic():
    # Tensorflow placeholders for I, alpha and td_error
    tf_beta = tf.placeholder(tf.float32, shape=(None, 1))
    tf_td_error = tf.placeholder(tf.float32, shape=(None, 1))
 
    for l in critic_model.layers:
        if l.weights:
            weights = l.weights[0]
            gradients = tf.gradients(critic_model.total_loss, weights)
            w = weights + tf_beta * tf_td_error * gradients
            l.weights[0] = w

    return tf_beta, tf_td_error, critic_model.layers[1].weights[0]


def train_actor():
    # Tensorflow placeholders for I, alpha and td_error
    tf_I = tf.placeholder(tf.float32, shape=(None, 1))
    tf_alpha = tf.placeholder(tf.float32, shape=(None, 1))
    tf_td_error = tf.placeholder(tf.float32, shape=(None, 1))

    for l in actor_model.layers:
        if l.weights:
            weights = l.weights[0]
            gradients = tf.gradients(actor_model.total_loss, weights)
            theta = weights + tf_alpha * tf_I * tf_td_error * gradients
            l.weights[0] = theta
            
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

# Critic gradients

weights = critic_model.trainable_weights # weight tensors
weights = [weight for weight in weights]
gradients_c = critic_model.optimizer.get_gradients(critic_model.total_loss, weights) # gradient tensors

input_tensors_c = [critic_model.inputs[0], # input data
                   critic_model.sample_weights[0], # how much to weight each sample by
                   critic_model.targets[0], # labels
                   K.learning_phase(), # train or test mode
]

get_gradients_critic = K.function(inputs=input_tensors_c, outputs=gradients_c)
print(actor_model.targets[0])

# Actor gradients
weights = actor_model.trainable_weights # weight tensors
weights = [weight for weight in weights]
gradients = actor_model.optimizer.get_gradients(actor_model.total_loss, weights) # gradient tensors

input_tensors = [actor_model.inputs[0], # input data
                 actor_model.sample_weights[0], # how much to weight each sample by
                 actor_model.targets[0], # labels
                 K.learning_phase(), # train or test mode
]

get_gradients_actor = K.function(inputs=input_tensors, outputs=gradients)



# Contants
episodes = 10000
gamma = 0.9
alpha = 0.1
beta = 0.2
I = 1.
# TF symbolic graphs
#actor_graph = train_actor()
#critic_graph = train_critic()

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
        _value = critic_model.predict(_state)
        td_error = reward + gamma * _value - value
        # Training the critic model
        inputs = [_state,
                 [1],
                 _value,
                 0]
        grads = get_gradients_critic(inputs)
        c = 1
        j = 0
        while j < len(grads):
            critic_model.layers[c].set_weights((critic_model.layers[c].get_weights()[0] + beta * td_error * grads[j], critic_model.layers[c].get_weights()[1]))
            c += 1
            j += 2

        # Training the actor model
        one_hot_probs = np.array([np.zeros(len(probabilities[0]))])
        one_hot_probs[0][action] = 1
        inputs = [state,
                 [1],
                 one_hot_probs,
                 0]
        grads = get_gradients_actor(inputs)
        c = 1
        j = 0
        while j < len(grads):
            actor_model.layers[c].set_weights((actor_model.layers[c].get_weights()[0] + alpha * I * td_error * grads[j], actor_model.layers[c].get_weights()[1]))
            c += 1
            j += 2
        I = I * gamma
        state = _state
        value = _value
    reward_list.append(total_reward)
    print("Episode", i, " - reward :",total_reward)

c = 0
k = 50
avg_list = []
for i in range(len(reward_list)):
    if c % k == 0:
        avg_list.append(sum(reward_list[i + c : i + c + k]) / k)

print(sum(reward_list) / len(reward_list))
plt.plot(range(len(reward_list)), reward_list, 'r')
plt.plot(range(len(avg_list)), avg_list, 'b')
plt.show()


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
   
sess = tf.Session()
K.set_session(sess)
# Maybe not needed
init = tf.global_variables_initializer()
sess.run(init)

input = Input(shape=[4])
value = Dense(8, activation='relu', init='uniform')(input)
value_2 = Dense(1, activation='linear', init='uniform')(value)

probs = Dense(8, activation='relu', init='uniform')(input)
probs_2 = Dense(2, activation='softmax', init='uniform')(probs)

critic_model = Model(inputs=input, outputs=value_2)
critic_model.compile(optimizer='sgd', loss='mse')

actor_model = Model(inputs=input, outputs=probs_2)
actor_model.compile(optimizer='sgd', loss='mse')

# Critic gradients

weights = critic_model.trainable_weights # weight tensors
weights = [weight for weight in weights]
weights = weights[::2]
gradients_c = critic_model.optimizer.get_gradients(critic_model.total_loss, weights) # gradient tensors

input_tensors_c = [critic_model.inputs[0], # input data
                   critic_model.sample_weights[0], # how much to weight each sample by
                   critic_model.targets[0], # labels
                   K.learning_phase(), # train or test mode
]

get_gradients_critic = K.function(inputs=input_tensors_c, outputs=gradients_c)

# Actor gradients
weights = actor_model.trainable_weights # weight tensors
weights = [weight for weight in weights]
weights = weights[::2]
gradients = actor_model.optimizer.get_gradients(actor_model.total_loss, weights) # gradient tensors

input_tensors = [actor_model.inputs[0], # input data
                 actor_model.sample_weights[0], # how much to weight each sample by
                 actor_model.targets[0], # labels
                 K.learning_phase(), # train or test mode
]

get_gradients_actor = K.function(inputs=input_tensors, outputs=gradients)

# Contants
episodes = 6000
training = 1500
gamma = 0.9
alpha = 0.3
beta = 0.2
# TF symbolic graphs
#actor_graph = train_actor()
#critic_graph = train_critic()

env = gym.make('CartPole-v0')
reward_list = []

for i in range(episodes):
    I = 1.
    total_reward = 0
    state = env.reset()
    state = state.reshape([1, 4])
    done = False
    value = critic_model.predict(state)
    while (not done):
        #if i > training:
        #    env.render()
        probabilities = actor_model.predict(state)
        action = np.random.choice(2, p=probabilities[0])
        _state, reward, done, info = env.step(action)
        _state = _state.reshape([1, 4])
        _value = critic_model.predict(_state)
        _value = _value if not done else np.array([[0.]])
        total_reward += reward
        td_error = reward + gamma * _value - value
        # Training the critic model
        inputs = [state,
                 [1],
                 value,
                 0]
        grads = get_gradients_critic(inputs)
        c = 1
        j = 0
        while j < len(grads):
            critic_model.layers[c].set_weights((critic_model.layers[c].get_weights()[0] + beta * td_error * grads[j], critic_model.layers[c].get_weights()[1]))
            c += 1
            j += 1

        # Training the actor model
        log_prob = np.array([np.array([np.log(probabilities[0][action] + 1e-7)])])
        inputs = [state,
                 [1],
                 log_prob,
                 0]
        grads = get_gradients_actor(inputs)
        c = 1
        j = 0
        while j < len(grads):
            actor_model.layers[c].set_weights((actor_model.layers[c].get_weights()[0] + alpha * I * td_error * grads[j], actor_model.layers[c].get_weights()[1]))
            c += 1
            j += 1
        I = I * gamma
        state = _state
        value = _value
    reward_list.append(total_reward)
    print("Episode", i, " - reward :",total_reward)

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


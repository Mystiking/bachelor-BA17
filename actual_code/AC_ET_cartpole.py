import gym
import numpy as np
from keras.models import Model
from keras.layers import Dense, Input
from keras import backend as K
import tensorflow as tf
import time
# Avoid annoying error messages about CPU instructions
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

'''
@param result_files: File that the result should be written to
@void: Runs the Actor-Critic with eligibility traces experiment
for the CartPole problem a single time.
'''
def train(result_file):
    result = open(result_file, 'w')
    # Contants
    episodes = 4000 
    gamma = 0.9
    alpha = 0.001
    beta = 0.0025
    lambda_w = 0.9
    lambda_theta = 0.9
    global_steps = 0

    start = time.time()
    # Tensorflow session : needed to actually "run" the computational graphs
    sess = tf.Session()
    # Make it the default Keras session
    K.set_session(sess)

    # Initialize global variables
    init = tf.global_variables_initializer()
    sess.run(init)

    # The networks used to estimate the policy and the value of a state
    # Input - output shape : 4 x 1. This layer is the same for both networks.
    input_layer = Input(shape=[4])

    # value - output shape : 8 x 1
    value = Dense(8, activation='relu', kernel_initializer='uniform', use_bias=0)(input_layer)
    # Output layer - output shape : 1 x 1
    value_2 = Dense(1, activation='linear', kernel_initializer='uniform', use_bias=0)(value)
    
    # probs - output shape : 8 x 1
    probs = Dense(8, activation='relu', kernel_initializer='uniform', use_bias=0)(input_layer)
    # Output layer - output shape : 2 x 1
    probs_2 = Dense(2, activation='softmax', kernel_initializer='uniform', use_bias=0)(probs)
    
    # Defining the models
    critic_model = Model(inputs=input_layer, outputs=value_2)
    actor_model = Model(inputs=input_layer, outputs=probs_2)
    
    # Getting the gradients
    # TF placeholder for a state
    s_t = tf.placeholder(tf.float32, shape=(None, 4))
    # TF placeholder for an action (as a 1-hot array)
    a_t = tf.placeholder(tf.float32, shape=(None, 2))
    # The value as predicted by the critic model
    v = critic_model(s_t)
    # The policy as predicted by the actor model
    p = actor_model(s_t)
    # Actor weights
    w_actor = actor_model.trainable_weights # weight tensors
    w_actor = [weight for weight in w_actor]
    # Critic weights
    w_critic = critic_model.trainable_weights # weight tensors
    w_critic = [weight for weight in w_critic]
    # "Losses"
    loss_actor = tf.log(tf.reduce_sum(p * a_t, axis=1, keep_dims=True) + 1e-10)
    loss_critic = v 
    # Computing the gradients
    a_grads = tf.gradients(loss_actor, w_actor)
    c_grads = tf.gradients(loss_critic, w_critic)
    # The environment
    env = gym.make('CartPole-v0')
    i = 0
    start_time = time.time()
    while global_steps < 200000: 
        i += 1
        I = 1.
        e_w = 0
        e_theta = 0
        total_reward = 0
        state = env.reset()
        state = state.reshape([1, 4])
        done = False
        value = critic_model.predict(state)
        while (not done):
            global_steps += 1
            probabilities = actor_model.predict(state)
            action = np.random.choice(2, p=probabilities[0])
            _state, reward, done, info = env.step(action)
            _state = _state.reshape([1, 4])
            _value = critic_model.predict(_state)
            _value = _value if not done else np.array([[0.]])
            total_reward += reward
            td_error = reward + gamma * _value - value
            # Getting the gradients 
            critic_grads = sess.run(c_grads, {s_t : state})
            # Creating the 1-hot array
            a_list = np.zeros(2)
            a_list[action] = 1.
            a_list = np.reshape(a_list, (-1, 2))
            actor_grads = sess.run(a_grads, {s_t : state, a_t : a_list})
            # Updating elegibility traces
            e_w = lambda_w * e_w + I * np.array(critic_grads)
            e_theta = lambda_theta * e_theta + I * np.array(actor_grads)
            # Updating the weights of the layers
            for j in range(1, len(e_w) + 1):
                # Critic update
                critic_model.layers[j].set_weights(critic_model.layers[j].get_weights() + beta * td_error * e_w[j-1])
                # Actor update
                actor_model.layers[j].set_weights(actor_model.layers[j].get_weights() + alpha * td_error * e_theta[j-1])
            I = I * gamma
            state = _state
            value = _value
        print("Episode {} - Reward : {}".format(i, total_reward, global_steps))
        result.write("{},{},{},{}\n".format(i, total_reward, time.time() - start_time, global_steps))
    end = time.time()
    result.write("{}".format(end - start))
    result.close()

# How many times the experiment should be run
runs = 5 
for i in range(runs):
    train("one_step_ac_result{}.csv".format(i))

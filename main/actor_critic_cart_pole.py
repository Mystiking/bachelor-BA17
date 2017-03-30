import gym
from gym import wrappers
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Model, Sequential
from keras.layers import Dense, Input, Convolution2D, MaxPooling2D, UpSampling2D
from keras import regularizers
from keras import optimizers 
from keras.optimizers import sgd, RMSprop
import random

env = gym.make('CartPole-v0')
#env = wrappers.Monitor(env, '/home/max/bachelor-BA17/polebalancing', force=True)

ACTIONS = [0, 1]

shared_optimizer = sgd(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)

actor_model = Sequential()
actor_model.add(Dense(16, input_dim=4, activation='relu'))
#actor_model.add(Dense(20, activation='relu'))
actor_model.add(Dense(2, activation='softmax'))
actor_model.compile(loss='mse', optimizer=shared_optimizer)

#actor_model = Sequential()
#actor_model.add(Dense(20, input_dim=4, activation='tanh'))
#actor_model.add(Dense(20, activation='tanh', init='uniform'))
#actor_model.add(Dense(1, activation='relu'))
#actor_model.compile(loss='mse', optimizer=shared_optimizer)
#actor_model.summary()

critic_model = Sequential()
critic_model.add(Dense(16, input_dim=4, activation='relu'))
#critic_model.add(Dense(20, activation='relu'))
critic_model.add(Dense(1, activation='linear'))
critic_model.compile(loss='mse', optimizer=shared_optimizer)

#critic_model.add(Dense(20, input_dim=4, activation='tanh'))
#critic_model.add(Dense(20, activation='tanh', init='uniform'))
#critic_model.add(Dense(1, activation='relu'))
#critic_model.compile(loss='mse', optimizer=shared_optimizer)

#critic_model.summary()

def softmax(x):
    return np.exp(x - np.max(x)) / np.sum(np.exp(x - np.max(x)))

def training(episodes = 500, batch_size = 40, gamma = 0.9
             , buffer = 200, alpha = 0.1, epsilon = 0.99):
   
    reward_list = []
    # Seperate replay memory for the two networks
    actor_replay = []
    critic_replay = []
 

    for i in range(episodes):
        observation = env.reset()
        observation = observation.reshape([1, 4])
        done = False
        reward = 0
        info = None
        total_reward = reward
        epsilon = epsilon * gamma if epsilon > 0.05 else epsilon
        while (not done):
            #env.render()
            orig_state = observation
            orig_reward = reward
            orig_val = critic_model.predict(np.array(orig_state))
           
            # Picking an action
            if random.random() < epsilon:
                action = random.randint(0, 1)
            else:
                action_vals = (actor_model.predict(np.array(orig_state)))
                action = np.random.choice(ACTIONS, p=action_vals[0])

            # Take the action and observe the result
            new_observation, new_reward, done, info = env.step(action)
            new_observation = new_observation.reshape([1, 4])

            new_state = new_observation

            # Punishing a loss
            #new_reward = new_reward if not done else -100
            # Value of the new state
            new_val = critic_model.predict(np.array(new_state))
            
            # Updating the target value
            td_error = new_reward + gamma * new_val - orig_val
            
            target = orig_val + alpha * (td_error) if not done else [[-10]]
            # Add the state and target to replay memory
            #critic_replay.append([orig_state, target])
            critic_replay.append([orig_state, new_state, action, target])

            # Add last state if it is reached
            if done:
                critic_replay.append([orig_state, new_state, action, target])

            # Add state, action and difference in value to action replay memory
            action_delta = new_val - orig_val
            #actor_replay.append([orig_state, action, action_delta])
            actor_replay.append([orig_state, new_state, action, td_error, action_delta])

            # Trim the critic memory to match "buffer" size
            if len(critic_replay) > buffer:
                critic_replay = critic_replay[-buffer:]
            # Replay the experience to the critic (Training)
            if len(critic_replay) >= buffer:
                minibatch = random.sample(critic_replay, batch_size)
                X_train = []
                Y_train = []
                for m in minibatch:
                    _state, _new_state, _action, _target = m
                    #prediction = critic_model.predict(_new_state)
                    prediction = target
                    y = np.array(prediction)
                    X_train.append(_state[0])
                    Y_train.append(y[0])
                X_train = np.array(X_train)
                Y_train = np.array(Y_train)
                critic_model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=1, verbose=False)
            
            # Trim the actor memory to match "buffer" size
            if len(actor_replay) > buffer:
                actor_replay = actor_replay[-buffer:]
            # Replay the experience to the actor (Training)
            if len(actor_replay) >= buffer:
                minibatch = random.sample(actor_replay, batch_size)
                X_train = []
                Y_train = []
                for m in minibatch:
                    _state, _new_state, _action, _td_error, _action_delta = m
                    old_action_vals = actor_model.predict(_state)[0]
                    old_action_vals[_action] += _td_error
                    X_train.append(_state[0])
                    Y_train.append(old_action_vals)
                X_train = np.array(X_train)
                Y_train = np.array(Y_train)
                actor_model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=1, verbose=False)
 
            observation = new_observation
            total_reward += reward 
            reward = new_reward

            if done:
                print("Episode = {} - reward = {}".format(i, total_reward))
                reward_list.append(total_reward)
    return reward_list

reward_list = training()
for i in range(10):
    i = i * 100
    avg = sum(reward_list[i:100 + i]) / 100
    print(avg)

'''
k = 10000
memory_limit = 100
epsilon = 1
min_epsilon = 0
epsilon_decay = 0.99 
e_min = 0.05
episodes = 0
gamma = 0.9
rewards = []
replay_memory = []
batch_size = 32
while episodes < k:
    total_reward = 0
    observation = env.reset()
    observation = np.reshape(observation, [1, 4])
    done = False
    while not done:
        env.render()
        action = model.predict(observation)
        action = np.argmax(action[0])
        if np.random.uniform(0,1) < epsilon:
            action = np.random.randint(2)
        observation_old = observation
        observation, reward, done, info = env.step(action)
        reward = reward if not done else -10
        total_reward += reward
        observation = np.reshape(observation, [1, 4])
        replay_memory.append([observation_old, action, reward, observation, done])
    rewards.append(total_reward)
    print("Reward for episode {} : {}".format(episodes, total_reward))
#    indices = np.random.choice(len(replay_memory), min(500, len(replay_memory)))

    batch_size = min(batch_size, len(replay_memory))
    mini_batch = random.sample(replay_memory, batch_size)
    X = np.zeros((batch_size, 4))
    Y = np.zeros((batch_size, 2))
    for i in range(batch_size):
        state, action, reward, next_state, done = mini_batch[i]
        target = model.predict(state)[0]
        if done:
            target[action] = reward
        else:
            target[action] = reward + gamma * np.amax(model.predict(next_state)[0])
        X[i], Y[i] = state, target
    model.fit(X, Y, batch_size=batch_size, nb_epoch=1, verbose=0)
    if epsilon > e_min:
        epsilon *= epsilon_decay
    episodes += 1

weights = model.get_weights()
f = open("cart_pole.keras", "w")
f.write(str(weights))
f.close()
averages = []
k = 10
for r in range(len(rewards[:-k])):
    avg = sum(rewards[r:r+k]) / k
    averages.append(avg)
plt.plot(range(len(averages[:-k])), averages[:-k])
plt.show()
'''

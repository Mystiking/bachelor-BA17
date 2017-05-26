import gym
from gym import wrappers
from scipy.misc import imresize
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
import queue
from time import time, sleep, gmtime, strftime




LR = 1e-4
GAMMA = 0.99
T_MAX = 100000000
NUM_THREADS = 8

# The environtment that we will be using
GAME_NAME = 'Breakout-v0'
env = gym.make(GAME_NAME)
NUM_STATE = env.observation_space.shape
NUM_ACTIONS = 3#env.action_space.n
RESIZED_WIDTH = int(NUM_STATE[0] / 2)
RESIZED_HEIGHT = int(NUM_STATE[1] / 2)

training_finished = False

#########################################
#       METHODS USED FOR PREPROCESING   #
#########################################
def resize(img):
    return img[::2,::2]

def rgb2grayscale(img):
    return np.dot(img[:,:,:3], [0.2126, 0.7152, 0.0722])





#########################################
#            CLASS AGENT                #
#########################################

class Agent():
    def __init__(self, session, action_size, model='mnih', optimizer=tf.train.RMSPropOptimizer(LR)):

        self.action_size = action_size
        self.optimizer = optimizer
        self.sess = session



        #########################################
        #            NETWORK                    #
        #########################################

        with tf.variable_scope('network'):
            self.action = tf.placeholder('int32', [None], name='action')
            self.target_value = tf.placeholder('float32', [None], name='target_value')
            self.state, self.policy, self.value = self.build_model(84, 84, 4)
            self.weights = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='network')
            self.advantages = tf.placeholder('float32', [None], name='advantages')




        #########################################
        #           OPTIMIZER                   #
        #########################################

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




    #########################################
    #           PREDICTERS                  #
    #########################################

    def get_policy(self, state):
        return self.sess.run(self.policy, {self.state: state}).flatten()

    def get_value(self, state):
        return self.sess.run(self.value, {self.state: state}).flatten()

    def get_policy_and_value(self, state):
        policy, value = self.sess.run([self.policy, self.value], {self.state:
        state})
        return policy.flatten(), value.flatten()




    #########################################
    #           TRAIN                       #
    #########################################

    # Train the network on the given states and rewards
    def train(self, states, actions, target_values, advantages):
        # Training
        self.sess.run(self.train_op, feed_dict={
            self.state: states,
            self.action: actions,
            self.target_value: target_values,
            self.advantages: advantages
        })





    #########################################
    #           NEURAL NETWORK              #
    #########################################

    # Builds the DQN model as in Mnih, but we get a softmax output for the
    # policy from fc1 and a linear output for the value from fc1.
    def build_model(self, h, w, channels):
        state = tf.placeholder('float32', shape=(None, h, w, channels), name='state')

        #input_layer = Input(shape=(RESIZED_WIDTH, RESIZED_HEIGHT, 1))
        #self.layers['state'] = input_layer
        # First conv layer
        first = Convolution2D(16, kernel_size=(8, 8), strides=4, padding='same', use_bias=1, activation='relu',
                              kernel_initializer='uniform')(state)
        # Second conv layer
        second = Convolution2D(32, kernel_size=(4, 4), strides=2, padding='same', use_bias=1, activation='relu',
                               kernel_initializer='uniform')(first)
        # Flattening the convs
        flat = Flatten()(second)
        # Dense layer 1
        dense = Dense(units=256, activation='relu', kernel_initializer='uniform')(flat)
        # Output layers
        output_actor = Dense(units=NUM_ACTIONS, activation='softmax', kernel_initializer='uniform')(dense)
        output_critic = Dense(units=1, activation='linear', kernel_initializer='uniform')(dense)

        return state, output_actor, output_critic







class CustomGym:
    def __init__(self, game_name, skip_actions=4, num_frames=4, w=84, h=84):
        self.env = gym.make(game_name)
        self.num_frames = num_frames
        self.skip_actions = skip_actions
        self.w = w
        self.h = h
        if game_name == 'SpaceInvaders-v0':
            self.action_space = [1,2,3] # For space invaders
        elif game_name == 'Pong-v0':
            self.action_space = [1,2,3]
        elif game_name == 'Breakout-v0':
            self.action_space = [1,2,3]
        else:
            # Use the actions specified by Open AI. Sometimes this has more
            # actions than we want, and some actions do the same thing.
            self.action_space = range(env.action_space.n)

        self.action_size = len(self.action_space)
        self.observation_shape = self.env.observation_space.shape

        self.state = None
        self.game_name = game_name

    def preprocess(self, obs, is_start=False):
        grayscale = obs.astype('float32').mean(2)
        len(grayscale)
        s = imresize(grayscale, (self.w, self.h)).astype('float32') * (1.0/255.0)
        s = s.reshape(1, s.shape[0], s.shape[1], 1)
        if is_start or self.state is None:
            self.state = np.repeat(s, self.num_frames, axis=3)
        else:
            self.state = np.append(s, self.state[:,:,:,:self.num_frames-1], axis=3)
        return self.state

    def render(self):
        self.env.render()

    def reset(self):
        return self.preprocess(self.env.reset(), is_start=True)

    def step(self, action_idx):
        action = self.action_space[action_idx]
        accum_reward = 0
        prev_s = None
        for _ in range(self.skip_actions):
            s, r, term, info = self.env.step(action)
            accum_reward += r
            if term:
                break
            prev_s = s
        # Takes maximum value for each pixel value over the current and previous
        # frame. Used to get round Atari sprites flickering (Mnih et al. (2015))
        if self.game_name == 'SpaceInvaders-v0' and prev_s is not None:
            s = np.maximum.reduce([s, prev_s])
        return self.preprocess(s), accum_reward, term, info




def async_trainer(agent, env, sess, thread_idx, T_queue):
    print("Training thread", thread_idx)
    T = T_queue.get()
    T_queue.put(T+1)
    t = 0

    last_verbose = T
    last_time = time()
    last_target_update = T

    terminal = True
    total_reward = 0
    while T < T_MAX:
        t_start = t
        batch_states = []
        batch_rewards = []
        batch_actions = []
        baseline_values = []

        if terminal:
            #print(total_reward)
            total_reward = 0
            terminal = False
            state = env.reset()

        while not terminal and len(batch_states) < 5:
            # Save the current state
            batch_states.append(state)

            # Choose an action randomly according to the policy
            # probabilities. We do this anyway to prevent us having to compute
            # the baseline value separately.
            policy, value = agent.get_policy_and_value(state)
            action_idx = np.random.choice(agent.action_size, p=policy)

            # Take the action and get the next state, reward and terminal.
            state, reward, terminal, _ = env.step(action_idx)
            total_reward += reward

            # Update counters
            t += 1
            T = T_queue.get()
            T_queue.put(T+1)

            # Clip the reward to be between -1 and 1
            reward = np.clip(reward, -1, 1)

            # Save the rewards and actions
            batch_rewards.append(reward)
            batch_actions.append(action_idx)
            baseline_values.append(value[0])

        target_value = 0
        # If the last state was terminal, just put R = 0. Else we want the
        # estimated value of the last state.
        if not terminal:
            target_value = agent.get_value(state)[0]
        last_R = target_value

        # Compute the sampled n-step discounted reward
        batch_target_values = []
        for reward in reversed(batch_rewards):
            target_value = reward + GAMMA * target_value
            batch_target_values.append(target_value)
        # Reverse the batch target values, so they are in the correct order
        # again.
        batch_target_values.reverse()

        # Compute the estimated value of each state
        batch_advantages = np.array(batch_target_values) - np.array(baseline_values)

        # Apply asynchronous gradient update
        agent.train(np.vstack(batch_states), batch_actions, batch_target_values,
        batch_advantages)

    global training_finished
    training_finished = True


def estimate_reward(agent, env, episodes=10, max_steps=10000):
    episode_rewards = []
    episode_vals = []
    t = 0
    for i in range(episodes):
        episode_reward = 0
        state = env.reset()
        terminal = False
        while not terminal:
            policy, value = agent.get_policy_and_value(state)
            action_idx = np.random.choice(agent.action_size, p=policy)
            state, reward, terminal, _ = env.step(action_idx)
            t += 1
            episode_vals.append(value)
            episode_reward += reward
            if t > max_steps:
                episode_rewards.append(episode_reward)
                return episode_rewards, episode_vals
        episode_rewards.append(episode_reward)
    return episode_rewards, episode_vals




def a3c(game_name, num_threads=16):
    processes = []
    envs = []
    for _ in range(num_threads+1):
        env = CustomGym(game_name)
        envs.append(env)

    # Separate out the evaluation environment
    evaluation_env = envs[0]
    envs = envs[1:]

    with tf.Session() as sess:
        agent = Agent(session=sess,
        action_size=envs[0].action_size, model='mnih',
        optimizer=tf.train.RMSPropOptimizer(LR))

        # Create a saver, and only keep 2 checkpoints.
        saver = tf.train.Saver(max_to_keep=2)

        T_queue = queue.Queue()

        # Either restore the parameters or don't.
        sess.run(tf.global_variables_initializer())
        T_queue.put(0)

        # Create a process for each worker
        for i in range(num_threads):
            processes.append(threading.Thread(target=async_trainer, args=(agent,
            envs[i], sess, i, T_queue)))


        # Start all the processes
        for p in processes:
            p.daemon = True
            p.start()

        # Until training is finished
        while not training_finished:
            sleep(0.01)

        # Join the processes, so we get this thread back.
        for p in processes:
            p.join()


def main():
    num_threads = NUM_THREADS
    game_name = 'Breakout-v0'
    a3c(game_name, num_threads=NUM_THREADS)


if __name__ == "__main__":
    main()






# OpenGym CartPole-v0 with A3C on GPU
# -----------------------------------
#
# A3C implementation with GPU optimizer threads.
#
# Made as part of blog series Let's make an A3C, available at
# https://jaromiru.com/2017/02/16/lets-make-an-a3c-theory/
#
# author: Jaromir Janisch, 2017

import numpy as np
import tensorflow as tf

import gym, time, random, threading

from keras.models import *
from keras.layers import *
from keras import backend as K

# -- constants
ENV = 'Breakout-v0'

RUN_TIME = 20000
THREADS = 8
OPTIMIZERS = 2
THREAD_DELAY = 0.001

GAMMA = 0.99

N_STEP_RETURN = 8
GAMMA_N = GAMMA ** N_STEP_RETURN

EPS_START = 0.4
EPS_STOP = .15
EPS_STEPS = 75000

MIN_BATCH = 32
LEARNING_RATE = 5e-3

LOSS_V = .5  # v loss coefficient
LOSS_ENTROPY = .01  # entropy coefficient

def resize(img):
    return img[::2,::2]

def rgb2grayscale(img):
    return np.dot(img[:,:,:3], [0.2126, 0.7152, 0.0722])

# ---------
class Brain:
    state_queue = []
    action_queue = []
    reward_queue = []
    _state_queue = []
    _state_terminal_queue = []
    #train_queue = [[], [], [], [], []]  # s, a, r, s', s' terminal mask
    lock_queue = threading.Lock()

    def __init__(self):
        self.session = tf.Session()
        K.set_session(self.session)
        K.manual_variable_initialization(True)

        self.model = self._build_model()
        self.graph = self._build_graph(self.model)

        self.session.run(tf.global_variables_initializer())
        self.default_graph = tf.get_default_graph()

        self.default_graph.finalize()  # avoid modifications

    def _build_model(self):

        input_layer = Input(shape=(RESIZED_WIDTH, RESIZED_HEIGHT, 1))
        # First conv layer
        first_layer = Convolution2D(16, kernel_size=(8, 8), strides=4, padding='same', use_bias=1, activation='relu',
                                    kernel_initializer='uniform')(input_layer)
        # Second conv layer
        second_layer = Convolution2D(32, kernel_size=(4, 4), strides=2, padding='same', use_bias=1, activation='relu',
                                     kernel_initializer='uniform')(first_layer)
        # Flattening the convs
        flat_layer = Flatten()(second_layer)
        # Dense layer 1
        dense_layer = Dense(units=256, activation='relu', kernel_initializer='uniform')(flat_layer)
        # Output layers
        output_actor = Dense(units=NUM_ACTIONS, activation='softmax', kernel_initializer='uniform')(dense_layer)
        output_critic = Dense(units=1, activation='linear', kernel_initializer='uniform')(dense_layer)
        # The "final" networks
        model = Model(inputs=input_layer, outputs=[output_actor, output_critic])
        # Must be done to enable threading
        model._make_predict_function()

        return model

    def _build_graph(self, model):
        s_t = tf.placeholder(tf.float32, shape=(None, RESIZED_WIDTH, RESIZED_HEIGHT, 1))
        a_t = tf.placeholder(tf.float32, shape=(None, NUM_ACTIONS))
        r_t = tf.placeholder(tf.float32, shape=(None, 1))  # not immediate, but discounted n step reward

        p, v = model(s_t)

        log_prob = tf.log(tf.reduce_sum(p * a_t, axis=1, keep_dims=True) + 1e-10)
        advantage = r_t - v

        loss_policy = - log_prob * tf.stop_gradient(advantage)  # maximize policy
        loss_value = LOSS_V * tf.square(advantage)  # minimize value error
        entropy = LOSS_ENTROPY * tf.reduce_sum(p * tf.log(p + 1e-10), axis=1,
                                               keep_dims=True)  # maximize entropy (regularization)

        loss_total = tf.reduce_mean(loss_policy + loss_value + entropy)

        optimizer = tf.train.RMSPropOptimizer(LEARNING_RATE, decay=.99)
        minimize = optimizer.minimize(loss_total)

        return s_t, a_t, r_t, minimize

    def optimize(self):
        if len(self.state_queue) < MIN_BATCH:
            time.sleep(0)  # yield
            return

        with self.lock_queue:
            s = self.state_queue
            a = self.action_queue
            r = self.reward_queue
            s_ = self._state_queue
            s_mask = self._state_terminal_queue
            #s, a, r, s_, s_mask = self.train_queue
            self.train_queue = [[], [], [], [], []]
            self.state_queue = []
            self.action_queue = []
            self.reward_queue = []
            self._state_queue = []
            self._state_terminal_queue = []

        print("s before vstack: {}".format(len(s)))
        s = np.vstack(s)
        print("s shape vstack: {}".format(s.shape))
        a = np.vstack(a)
        r = np.vstack(r)
        print("s_ before vstack: {}".format(len(s_)))
        s_ = np.vstack(s_)
        print("s_ after vstack: {}".format(s_.shape))
        s_mask = np.vstack(s_mask)

        if len(s) > 5 * MIN_BATCH: print("Optimizer alert! Minimizing batch of %d" % len(s))

        v = self.predict_v(s_)
        r = r + GAMMA_N * v * s_mask  # set v to 0 where s_ is terminal state

        s_t, a_t, r_t, minimize = self.graph
        self.session.run(minimize, feed_dict={s_t: s, a_t: a, r_t: r})

    def train_push(self, s, a, r, s_):
        self.state_queue.append(s)
        self.action_queue.append(a)
        self.reward_queue.append(r)

        if s_ is None:
            self._state_queue.append(NONE_STATE)
            self._state_terminal_queue.append(0.)
        else:
            self._state_queue.append(s_)
            self._state_terminal_queue.append(1.)

        #with self.lock_queue:
        #    self.train_queue[0].append(s)
        #    self.train_queue[1].append(a)
        #    self.train_queue[2].append(r)

        #    if s_ is None:
        #        self.train_queue[3].append(NONE_STATE)
        #        self.train_queue[4].append(0.)
        #    else:
        #        self.train_queue[3].append(s_)
        #        self.train_queue[4].append(1.)

    def predict(self, s):
        with self.default_graph.as_default():
            p, v = self.model.predict(s)
            return p, v

    def predict_p(self, s):
        with self.default_graph.as_default():
            p, v = self.model.predict(s)
            return p

    def predict_v(self, s):
        with self.default_graph.as_default():
            p, v = self.model.predict(s)
            return v


# ---------
frames = 0


class Agent:
    def __init__(self, eps_start, eps_end, eps_steps):
        self.eps_start = eps_start
        self.eps_end = eps_end
        self.eps_steps = eps_steps

        self.memory = []  # used for n_step return
        self.R = 0.

    def getEpsilon(self):
        if (frames >= self.eps_steps):
            return self.eps_end
        else:
            return self.eps_start + frames * (self.eps_end - self.eps_start) / self.eps_steps  # linearly interpolate

    def act(self, s):
        eps = self.getEpsilon()
        global frames;
        frames = frames + 1

        if random.random() < eps:
            return random.randint(0, NUM_ACTIONS - 1)

        else:
            p = brain.predict_p(s)[0]

            # a = np.argmax(p)
            a = np.random.choice(NUM_ACTIONS, p=p)

            return a

    def train(self, s, a, r, s_):
        def get_sample(memory, n):
            s, a, _, _ = memory[0]
            _, _, _, s_ = memory[n - 1]

            return s, a, self.R, s_

        a_cats = np.zeros(NUM_ACTIONS)  # turn action into one-hot representation
        a_cats[a] = 1

        self.memory.append((s, a_cats, r, s_))

        self.R = (self.R + r * GAMMA_N) / GAMMA

        if s_ is None:
            while len(self.memory) > 0:
                n = len(self.memory)
                s, a, r, s_ = get_sample(self.memory, n)
                brain.train_push(s, a, r, s_)

                self.R = (self.R - self.memory[0][2]) / GAMMA
                self.memory.pop(0)

            self.R = 0

        if len(self.memory) >= N_STEP_RETURN:
            s, a, r, s_ = get_sample(self.memory, N_STEP_RETURN)
            brain.train_push(s, a, r, s_)

            self.R = self.R - self.memory[0][2]
            self.memory.pop(0)

            # possible edge case - if an episode ends in <N steps, the computation is incorrect


# ---------
class Environment(threading.Thread):
    stop_signal = False

    def __init__(self, render=False, eps_start=EPS_START, eps_end=EPS_STOP, eps_steps=EPS_STEPS):
        threading.Thread.__init__(self)

        self.render = render
        self.env = gym.make(ENV)
        self.agent = Agent(eps_start, eps_end, eps_steps)

    def runEpisode(self):
        s = self.env.reset()
        s = resize(rgb2grayscale(s))
        s = np.reshape(s, (-1, s.shape[0], s.shape[1], 1))

        R = 0
        while True:
            time.sleep(THREAD_DELAY)  # yield

            if self.render: self.env.render()

            a = self.agent.act(s)
            s_, r, done, info = self.env.step(a)
            s_ = resize(rgb2grayscale(s_))
            s_ = np.reshape(s_, (-1, s_.shape[0], s_.shape[1], 1))

            if done:  # terminal state
                s_ = None

            self.agent.train(s, a, r, s_)

            s = s_
            R += r

            if done or self.stop_signal:
                break

        print("Total R:", R)

    def run(self):
        while not self.stop_signal:
            self.runEpisode()

    def stop(self):
        self.stop_signal = True


# ---------
class Optimizer(threading.Thread):
    stop_signal = False

    def __init__(self):
        threading.Thread.__init__(self)

    def run(self):
        while not self.stop_signal:
            brain.optimize()

    def stop(self):
        self.stop_signal = True


# -- main
env_test = Environment(render=True, eps_start=0., eps_end=0.)
NUM_STATE = env_test.env.observation_space.shape
RESIZED_WIDTH = int(NUM_STATE[0] / 2)
RESIZED_HEIGHT = int(NUM_STATE[1] / 2)
NUM_ACTIONS = env_test.env.action_space.n
NONE_STATE = np.zeros(NUM_STATE)

brain = Brain()  # brain is global in A3C

envs = [Environment() for i in range(THREADS)]
opts = [Optimizer() for i in range(OPTIMIZERS)]

for o in opts:
    o.start()

for e in envs:
    e.start()

time.sleep(RUN_TIME)

for e in envs:
    e.stop()
for e in envs:
    e.join()

for o in opts:
    o.stop()
for o in opts:
    o.join()

print("Training finished")
env_test.run()
import gym
import numpy as np
from keras.models import Model
from keras.layers import Dense, Input
from keras import backend as K
import tensorflow as tf
import time
import threading
# Avoid annoying error messages about CPU instructions
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# Makes it easier to assign number of threads/episodes/runs
flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_integer('num_threads', 1, 'Number of threads')
flags.DEFINE_integer('max_steps', 10000, 'Number of maximum timesteps')
flags.DEFINE_integer('num_runs', 1, 'Number of runs')
NUM_THREADS = FLAGS.num_threads 

# Repeat the experiment num_runs time
for run in range(FLAGS.num_runs):
    # Number of global timesteps
    global_steps = 0
    # Constants
    ALPHA = 0.9
    GAMMA = 0.90
    LR = 0.001
    EPSILON = 1e-8
    GAME_NAME = 'CartPole-v0'
    # The environtment that we will be using
    env = gym.make(GAME_NAME)
    NUM_STATE = env.observation_space.shape[0]
    NUM_ACTIONS = env.action_space.n
    ######################
    # The Global Network #
    ######################
    # Input layer - "output" shape : 4 x 1
    input_layer      = Input(shape=[4])
    # Dense layer - output shape : 8 x 1 
    first= Dense(8, activation='relu', kernel_initializer='uniform', use_bias=0)(input_layer)
    # Output layers
    # Policy output shape : NUM_ACTIONS x 1
    output_actor  = Dense(units=NUM_ACTIONS, activation='softmax', kernel_initializer='uniform', use_bias=0)(first)
    # Value estimate output shape : 1 x 1
    output_critic = Dense(units=1, activation=None, kernel_initializer='uniform', use_bias=0)(first)
    # Finalizing the networks
    global_model = Model(inputs=input_layer, outputs=[output_actor, output_critic])
    # Must be done to enable threading
    global_model._make_predict_function()
    # Global optimizer 
    opt = tf.train.RMSPropOptimizer(LR, decay=ALPHA, epsilon=EPSILON)
    # Tensorflow session : needed to actually "run" the computational graphs
    sess = tf.Session()
    # Make it the default Keras session
    K.set_session(sess)
    # Initialize tf global variables
    init = tf.global_variables_initializer()
    sess.run(init)
    # "graph" is needed to use the global model
    graph = tf.get_default_graph()
    # Timing the runs
    start_time = time.time()
    
    class Agent(threading.Thread):
        def __init__(self, state_input_shape, action_output_shape, threadid, main = False):
            self.result_file = open("cartpole_agent_{}_results_{}.csv".format(NUM_THREADS, threadid, run), 'w')
            start_time = time.time()
            threading.Thread.__init__(self)
            self.step_counter = 0
            # Init layers of the model
            input_layer      = Input(shape=[4])
            # First conv layer
            first= Dense(8, activation='relu', kernel_initializer='uniform', use_bias=0)(input_layer)
            #second= Dense(8, activation='relu', kernel_initializer='uniform')(first)
            # Output layers
            output_actor  = Dense(units=NUM_ACTIONS, activation='softmax', kernel_initializer='uniform', use_bias=0)(first)
            output_critic = Dense(units=1, activation=None, kernel_initializer='uniform', use_bias=0)(first)
            # The "final" networks
            self.local_model = Model(inputs=input_layer, outputs=[output_actor, output_critic])
            # Must be done to enable threading
            self.local_model._make_predict_function()
            
            # TF placeholder for a state
            self.s_t = tf.placeholder(tf.float32, shape=(None, NUM_STATE))
            # TF placeholder for the action taken (as a 1-hot array) 
            self.a_t = tf.placeholder(tf.float32, shape=(None, NUM_ACTIONS), name="actions")
            # TF placeholder for a reward
            self.r_t = tf.placeholder(tf.float32, shape=(None, 1), name="reward")
            # TF placeholder for an advantage
            self.advantage = tf.placeholder(tf.float32, shape=(None, 1), name="advantages")
            # "Predictions" of the probabilities and value
            p, v = self.local_model(self.s_t)
            
            # The weights of the local model
            weights = self.local_model.trainable_weights # weight tensors
            # Actor weights
            actor_weights = [weights[0], weights[1]]
            # Critic weights
            critic_weights = [weights[0], weights[2]]
            # Updating the weights of the global model
            # The logarithm of the probabilities
            log_p = tf.log(tf.clip_by_value(p, 0.000001, 0.999999))
            # The policy times the experienced advantage
            loss_actor = tf.reduce_mean(tf.reduce_sum(tf.multiply(log_p, self.a_t), reduction_indices=1) * self.advantage)
            # The squared advantage (weighted) 
            loss_critic = 0.5 * tf.reduce_mean(tf.square(self.r_t - v))
            # The entropy (weighted)
            loss_entropy = 0.01 * tf.reduce_sum(tf.multiply(p, log_p))

            # Computing the gradients and performing the asynchronous update
            with graph.as_default():
                # Weights of the global model
                global_weights = global_model.trainable_weights
                global_actor_weights = [global_weights[0], global_weights[1]]
                global_critic_weights =[global_weights[0], global_weights[2]]
                # Computing and applying the actor gradients
                actor_gradients = tf.gradients(loss_actor + loss_entropy, actor_weights) 
                grads_var = list(zip(actor_gradients, global_actor_weights))
                self.policy_update = opt.apply_gradients(grads_var)
                # Computing and applying the critic gradients
                critic_gradients = tf.gradients(loss_critic, critic_weights) 
                grads_var = list(zip(critic_gradients, global_critic_weights))
                self.critic_update = opt.apply_gradients(grads_var)
    
        '''
        @void: Starts the training process
        '''
        def run(self):
            self.train(1)
        '''
        @param t_max: Amount of steps to perform before updating
        '''
        def train(self, t_max):
            global LR, global_steps
            i = 0
            env = gym.make(GAME_NAME)
            while global_steps < FLAGS.max_steps:
                # Initialize weights of local networks
                with graph.as_default():
                    start_weights = global_model.get_weights()
                    self.local_model.set_weights(start_weights)
                # Used for printing
                i += 1
                total_reward = 0
                t_start = self.step_counter
                memory = []
                state = env.reset()   
                state = state.reshape([1, NUM_STATE])
                done = False
                while not done:
                    global_steps += 1
                    probabilities, value = self.local_model.predict(state)
                    action = np.random.choice(2, p=probabilities[0])
                    _state, reward, done, info = env.step(action)
                    _state = _state.reshape([1, NUM_STATE])
                    total_reward += reward
                    memory.append([state, action, reward, done, value])
                    state = _state
                    self.step_counter += 1
                    # Update every t_max step
                    if self.step_counter - t_start == t_max or done:
                        R = np.array([[0]]) if done else self.local_model.predict(state)[1]
                        memory = reversed(memory)
                        action_list = []
                        return_list = []
                        state_list  = []
                        advantages  = []
                        for s, a, r, d, v in memory:
                            R = r + GAMMA * R
                            a_list = np.zeros(NUM_ACTIONS)
                            a_list[a] = 1
                            action_list.append(a_list) 
                            if R != 0:
                                return_list.append(R[0])
                            else:
                                return_list.append(R)
                            s = s[0]
                            state_list.append(s)
                            advantage = np.reshape(R - v, (-1, 1))
                            advantages.append(advantage[0])
                        # Re-reverse the lists
                        state_list  = state_list[::-1]
                        action_list = action_list[::-1]
                        return_list = return_list[::-1]
                        advantages  = advantages[::-1]
                        # Update gradients
                        sess.run(self.policy_update, {self.s_t : state_list,
                                                 self.a_t : action_list,
                                                 self.r_t : return_list,
                                                 self.advantage : np.array(advantages)})
                        sess.run(self.critic_update, {self.s_t : state_list,
                                                 self.a_t : action_list,
                                                 self.r_t : return_list,
                                                 self.advantage : np.array(advantages)})


                        with graph.as_default():
                            # Reset weights to global weights
                            start_weights = global_model.get_weights()
                            self.local_model.set_weights(start_weights)
                        t_start = self.step_counter
                        memory = []
                # Print summary of performance 
                print("Episode : {}, Reward : {}".format(i, total_reward))
                now = time.time()
                # Write to result file
                self.result_file.write("{},{},{},{}\n".format(i, total_reward, now - start_time, global_steps))
            self.result_file.close()
    
    # Make one actor for each NUM_THREAD 
    agents = [Agent(NUM_STATE, NUM_ACTIONS, a) for a in range(NUM_THREADS)]
  
    for a in agents:
        a.start()
        time.sleep(1)

    # Join since we are looping!
    for a in agents:
        a.join()


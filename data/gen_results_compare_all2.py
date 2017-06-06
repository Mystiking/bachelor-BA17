import matplotlib.pyplot as plt
import pandas as pd  
import os 
import re
from data_handler2 import * 

#test = get_data_points_from_file('cartpole_agent_0_results_0.csv', 'cartpole/cartpole_1_threads')
#test2 = separate_data_atari(test)

#test = get_score_and_counter_one_agent_atari('cartpole/cartpole_1_threads', 'cartpole_agent_0_results_0.csv')

test = get_score_and_counter_all_agents_atari('spaceInvaders/space_8_threads')
plot_variance_one_run_graph_atari('spaceInvaders/space_1_threads', 1000, 2, '1 threads', 'Pong')
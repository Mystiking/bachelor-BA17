import matplotlib.pyplot as plt
import pandas as pd  
import os 
import re
from data_handler import * 

avg_scores = []
folders = ['a3c_2_agents', 'a3c_4_agents']




a3c_1_agents_scores = get_average_score_for_every_agent_in_folder_k('spaceInvaders/space_1_threads', 1, 1)
a3c_2_agents_scores = get_average_score_for_every_agent_in_folder_k('spaceInvaders/space_2_threads', 2, 1)
a3c_4_agents_scores = get_average_score_for_every_agent_in_folder_k('spaceInvaders/space_4_threads', 4, 1)
a3c_8_agents_scores = get_average_score_for_every_agent_in_folder_k('spaceInvaders/space_8_threads', 8, 1)
a3c_16_agents_scores = get_average_score_for_every_agent_in_folder_k('spaceInvaders/space_16_threads', 16, 1)




#a3c_4_agents_scores = get_average_score_for_every_agent_in_folder('a3c_4_agents', 4)
#a3c_8_agents_scores = get_average_score_for_every_agent_in_folder('a3c_8_agents', 8)

test = []

test.append(a3c_1_agents_scores)
test.append(a3c_2_agents_scores)
test.append(a3c_4_agents_scores)
test.append(a3c_8_agents_scores)
test.append(a3c_16_agents_scores)


plot_graph(test, 2000, 1)



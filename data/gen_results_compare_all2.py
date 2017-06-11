import matplotlib.pyplot as plt
import pandas as pd  
import os 
import re
from data_handler2 import * 

#test = get_data_points_from_file('cartpole_agent_0_results_0.csv', 'cartpole/cartpole_1_threads')
#test2 = separate_data_atari(test)

#test = get_score_and_counter_one_agent_atari('cartpole/cartpole_1_threads', 'cartpole_agent_0_results_0.csv')

#test = get_score_and_counter_all_agents_atari('spaceInvaders/space_8_threads')
#plot_variance_one_run_graph_atari('spaceInvaders/space_1_threads', 1000, 2, '1 threads', 'Pong')

'''
plot_score_run_graph_cart_pole('cartpole_1_threads', 201, 200000, 2, 'cartpole_true', 6, 'cartpole_1_thread_variance', '1 threads')
plot_score_run_graph_cart_pole('cartpole_2_threads', 201, 200000, 4, 'cartpole_true', 6, 'cartpole_2_thread_variance', '2 threads')
plot_score_run_graph_cart_pole('cartpole_4_threads', 201, 200000, 8, 'cartpole_true', 6, 'cartpole_4_thread_variance', '4 threads')
plot_score_run_graph_cart_pole('cartpole_8_threads', 201, 200000, 10, 'cartpole_true', 6, 'cartpole_8_thread_variance', '8 threads')
plot_score_run_graph_cart_pole('cartpole_16_threads', 201, 200000, 12, 'cartpole_true', 6, 'cartpole_16_thread_variance', '16 threads')
'''

#plot_score_all_run_graph_cart_pole(201, 200000, 2, 'cartpole_true', 6, 'cartpole_compare_k5')
#plot_score_all_run_graph_cart_pole(201, 200000, 2, 'cartpole_true', 10, 'cartpole_compare_k10')
#plot_score_all_run_graph_cart_pole(201, 200000, 2, 'cartpole_true', 50, 'cartpole_compare_k502')
#plot_score_all_run_graph_cart_pole(201, 200000, 2, 'cartpole_true', 100, 'cartpole_compare_k100')

#plot_score_all_run_graph_cart_pole_time(201, 650, 2, 'cartpole_true', 50, 'cartpole_compare_k50')


########################################
#      ELIGIBILITY CARTPOLE SCORE      #
########################################


#plot_score_all_run_graph_cart_pole_eligibilty_score(201, 200000, 2, 6, 'eligibility_k_5')
'''plot_score_all_run_graph_cart_pole_eligibilty_score(201, 200000, 2, 50, 'eligibility_k_50')
plot_score_all_run_graph_cart_pole_eligibilty_score(201, 200000, 2, 100, 'eligibility_k_100')
plot_score_all_run_graph_cart_pole_eligibilty_score(201, 200000, 2, 500, 'eligibility_k_500')
'''



########################################
#      ELIGIBILITY CARTPOLE TIME      #
########################################


#plot_score_all_run_graph_cart_pole_eligibilty_time(201, 1515, 2, 6, 'eligibility_time_k_5')
'''plot_score_all_run_graph_cart_pole_eligibilty_time(201, 1515, 2, 50, 'eligibility_time_k_50')
plot_score_all_run_graph_cart_pole_eligibilty_time(201, 1515, 2, 100, 'eligibility_time_k_100')
plot_score_all_run_graph_cart_pole_eligibilty_time(201, 1515, 2, 500, 'eligibility_time_k_500')
'''



#plot_score_eligibilty_vs_A3C(201, 200000, 2, 'cartpole_true', 50, 'cartpole_compare')

#plot_score_eligibilty_vs_A3C_time(201, 2000, 2, 'cartpole_true', 6, 'cartpole_compare_time')




#find_max_score_at_step()



#plot_score_all_run_graph_spcaeinvaders_counter_score_aliens(32, 9504462, 2, 'space', 50, 'breakout_compare_k50')

#plot_score_all_run_graph_atari_counter(17, 57600, 2, 'breakout_11_06', 500, 'breakout_compare_k50')

#plot_score_all_run_graph_spcaeinvaders_counter_score(1000, 9504462, 2, 'space', 50, 'cartpole_compare_k50')
#plot_score_all_run_graph_spcaeinvaders_counter_score(1000, 9504462, 2, 'space', 50, 'cartpole_compare_k50')
#plot_score_all_run_graph_spcaeinvaders_counter_score(1000, 9504462, 2, 'space', 50, 'cartpole_compare_k50')
#plot_score_all_run_graph_spcaeinvaders_counter_score(1000, 9504462, 2, 'space', 50, 'cartpole_compare_k50')
#plot_score_all_run_graph_spcaeinvaders_counter_score(1000, 9504462, 2, 'space', 50, 'cartpole_compare_k50')




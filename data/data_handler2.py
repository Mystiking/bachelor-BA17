import matplotlib.pyplot as plt
import numpy as np
import os
import re


########################################
#        GET DATA FROM FILE            #
########################################

def get_data_points_from_file(fname, folder):
    data = np.genfromtxt(folder + '/' + fname, delimiter=',')
    return data


def get_files_containing_regex(filelist, regex):
    pattern = re.compile(regex)
    matches = []
    for f in filelist:
        match = re.findall(pattern, f)
        if match:
            matches.append(match[0])
    return matches




########################################
#           SPLIT THE DATA             #
########################################

def separate_data_atari(data):
    return data[:,0], data[:,1], data[:,2]

def separate_data_spaceinvaders(data):
    return data[:,0], data[:,1], data[:,2], data[:,3]

def separate_data_cartpole(data):
    return data[:,0], data[:,1], data[:,2], data[:,3]







########################################
#   GET INFORMATION FROM ONE AGENT     #
########################################


def get_score_and_counter_one_agent_atari(folder, file):
    filenames = os.listdir(folder)
    data = get_data_points_from_file(file, folder)
    global_counter, score, time = separate_data_atari(data)
    list_tuples_counter_score = list(zip(global_counter, score))
    return list_tuples_counter_score



def get_score_and_counter_one_agent_spaceinvaders_score(folder, file):
    filenames = os.listdir(folder)
    data = get_data_points_from_file(file, folder)
    global_counter, score, time, alienskilled = separate_data_spaceinvaders(data)
    list_tuples_counter_score = list(zip(global_counter, score))
    return list_tuples_counter_score


def get_score_and_counter_one_agent_spaceinvaders_aliens_killed(folder, file):
    filenames = os.listdir(folder)
    data = get_data_points_from_file(file, folder)
    global_counter, score, time, alienskilled = separate_data_spaceinvaders(data)
    list_tuples_counter_score = list(zip(global_counter, alienskilled))
    return list_tuples_counter_score


def get_score_and_counter_one_agent_cartpole(folder, file):
    filenames = os.listdir(folder)
    data = get_data_points_from_file(file, folder)
    episodes, score, time, global_counter = separate_data_cartpole(data)
    list_tuples_counter_score = list(zip(global_counter, score))
    return list_tuples_counter_score







########################################
#   GET INFORMATION FROM ALL AGENTS    #
########################################


def get_score_and_counter_all_agents_atari(folder):
    filenames = os.listdir(folder)
    agent_folders = get_files_containing_regex(filenames, '.+agent' + '.+')
    print(agent_folders)
    all_data = []
    for s in agent_folders:
    	data = get_score_and_counter_one_agent_atari(folder, s)
    	all_data = sum([all_data, data], [])
    all_data.sort(key=lambda tup:tup[0])
    return all_data



def get_score_and_counter_all_agents_spaceinvaders_score(folder):
    filenames = os.listdir(folder)
    agent_folders = get_files_containing_regex(filenames, '.+agent' + '.+')
    print(agent_folders)
    all_data = []
    for s in agent_folders:
    	data = get_score_and_counter_one_agent_spaceinvaders_score(folder, s)
    	all_data = sum([all_data, data], [])
    all_data.sort(key=lambda tup:tup[0])
    return all_data



def get_score_and_counter_all_agents_spaceinvaders_score(folder):
    filenames = os.listdir(folder)
    agent_folders = get_files_containing_regex(filenames, '.+agent' + '.+')
    print(agent_folders)
    all_data = []
    for s in agent_folders:
    	data = get_score_and_counter_one_agent_spaceinvaders_aliens_killed(folder, s)
    	all_data = sum([all_data, data], [])
    all_data.sort(key=lambda tup:tup[0])
    return all_data



def get_score_and_counter_all_agents_spaceinvaders_cartpole(folder):
    filenames = os.listdir(folder)
    agent_folders = get_files_containing_regex(filenames, '.+agent' + '.+')
    print(agent_folders)
    all_data = []
    for s in agent_folders:
    	data = get_score_and_counter_one_agent_cartpole(folder, s)
    	all_data = sum([all_data, data], [])
    all_data.sort(key=lambda tup:tup[0])
    return all_data







'''
    total_score = 0
    counter = 0
    for s in agent_folders:
        counter += 1
        data = get_data_points_from_file(s, folder)
        episodes, scores, times = separate_data(data)
        total_score += np.array(scores)
    total_score = total_score / counter
    return total_score
'''


def plot_variance_one_run_graph_atari(folder, y_max ,colour_num, string_number_of_threads, atari_game):
	data = get_score_and_counter_all_agents_atari(folder)
	split_data = list(map(list, zip(*data)))
	x_points = split_data[0]
	y_points = split_data[1]
	
	tableau20 = [(31, 119, 180), (174, 199, 232), (255, 127, 14), (255, 187, 120),
    			 (44, 160, 44), (152, 223, 138), (214, 39, 40), (255, 152, 150),
                 (148, 103, 189), (197, 176, 213), (140, 86, 75), (196, 156, 148),
                 (227, 119, 194), (247, 182, 210), (127, 127, 127), (199, 199, 199),
                 (188, 189, 34), (219, 219, 141), (23, 190, 207), (158, 218, 229)]
	
	for i in range(len(tableau20)):
		r, g, b = tableau20[i]
		tableau20[i] = (r / 255., g / 255., b / 255.)    
 
	plt.figure(figsize=(28, 10))
          
	ax = plt.subplot(111)
	ax.spines["top"].set_visible(False)
	ax.spines["bottom"].set_visible(False)
	ax.spines["right"].set_visible(False)
	ax.spines["left"].set_visible(False)
        
	ax.get_xaxis().tick_bottom()
	ax.get_yaxis().tick_left()

	plt.ylim(0, y_max)
	plt.xlim(0, x_points[-1] + 5)

	print(y_max)
	plt.yticks(range(0, y_max, int(50)), [str(x) for x in range(0, y_max, 50)], fontsize=14)
	plt.xticks(fontsize=14)
	  
	# Provide tick lines across the plot to help your viewers trace along    
	# the axis ticks. Make sure that the lines are light and small so they    
	# don't obscure the primary data lines.  
	for y in range(0, y_max + 1, int(50)):
		plt.plot(range(0, int(x_points[-1])), [y] * len(range(0, int(x_points[-1]))), "-", lw=0.5, color="black", alpha=0.3)
	    #plt.plot(range(0, x_points[-1]), [y] * len(range(0, x_points[-1])), "-", lw=0.5, color="black", alpha=0.3)
	  
	# Remove the tick marks; they are unnecessary with the tick lines we just plotted.    
	plt.tick_params(axis="both", which="both", bottom="off", top="off",
	                labelbottom="on", left="off", right="off", labelleft="on")

	#plt.plot(range(1000), a3c_2_agents_scores, color=tableau20[0])
	#plt.plot(range(1000), a3c_2_agents_scores[0], color=tableau20[0])
	plt.plot(x_points, y_points, color=tableau20[colour_num])
	plt.text(int(x_points[-1]) + 0.5, int(y_points[-1]), string_number_of_threads, fontsize=14, color=tableau20[1])

	plt.title(atari_game, y=1.02)
	plt.xlabel('Time in seconds')
	plt.ylabel('Score')
	plt.savefig("avg_score_pr_episode.png", bbox_inches="tight")







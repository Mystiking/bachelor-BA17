import matplotlib.pyplot as plt
import numpy as np
import os
import re
from operator import itemgetter


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



def remove_last_elements(data, k):
    length = len(data)
    elements_to_remove = length % k
    if elements_to_remove == 0:
        data = data
    else:
        data = data[:-elements_to_remove]
    return data



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


def get_score_and_time_one_agent_spaceinvaders_score(folder, file):
    filenames = os.listdir(folder)
    data = get_data_points_from_file(file, folder)
    global_counter, score, time, alienskilled = separate_data_spaceinvaders(data)
    list_tuples_counter_score = list(zip(time, score))
    return list_tuples_counter_score



def get_score_and_time_one_agent_spaceinvaders_aliens_killed(folder, file):
    filenames = os.listdir(folder)
    data = get_data_points_from_file(file, folder)
    global_counter, score, time, alienskilled = separate_data_spaceinvaders(data)
    list_tuples_counter_score = list(zip(time, alienskilled))
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
#    GET TIME/SCORE FROM ALL AGENTS    #
########################################

def get_score_and_time_one_agent_cartpole(folder, file):
    filenames = os.listdir(folder)
    data = get_data_points_from_file(file, folder)
    episodes, score, time, global_counter = separate_data_cartpole(data)
    list_tuples_counter_score = list(zip(time, score))
    return list_tuples_counter_score



########################################
#   GET INFORMATION FROM ALL AGENTS    #
########################################


def get_score_and_counter_all_agents_atari(folder):
    filenames = os.listdir(folder)
    agent_folders = get_files_containing_regex(filenames, '.+agent' + '.+')
    all_data = []
    for s in agent_folders:
        data = get_score_and_counter_one_agent_atari(folder, s)
        all_data = sum([all_data, data], [])
    all_data.sort(key=lambda tup:tup[0])
    return all_data



def get_score_and_counter_all_agents_spaceinvaders_score(folder):
    filenames = os.listdir(folder)
    agent_folders = get_files_containing_regex(filenames, '.+agent' + '.+')
    all_data = []
    for s in agent_folders:
    	data = get_score_and_counter_one_agent_spaceinvaders_score(folder, s)
    	all_data = sum([all_data, data], [])
    all_data.sort(key=lambda tup:tup[0])
    return all_data





def get_aliens_and_counter_all_agents_spaceinvaders_aliens(folder):
    filenames = os.listdir(folder)
    agent_folders = get_files_containing_regex(filenames, '.+agent' + '.+')
    all_data = []
    for s in agent_folders:
        data = get_score_and_counter_one_agent_spaceinvaders_aliens_killed(folder, s)
        all_data = sum([all_data, data], [])
    all_data.sort(key=lambda tup:tup[0])
    return all_data


def get_aliens_and_time_all_agents_spaceinvaders_aliens(folder):
    filenames = os.listdir(folder)
    agent_folders = get_files_containing_regex(filenames, '.+agent' + '.+')
    all_data = []
    for s in agent_folders:
        data = get_score_and_time_one_agent_spaceinvaders_aliens_killed(folder, s)
        all_data = sum([all_data, data], [])
    all_data.sort(key=lambda tup:tup[0])
    return all_data



def get_score_and_counter_all_agents_spaceinvaders_aliens(folder):
    filenames = os.listdir(folder)
    agent_folders = get_files_containing_regex(filenames, '.+agent' + '.+')
    all_data = []
    for s in agent_folders:
        data = get_score_and_counter_one_agent_spaceinvaders_score(folder, s)
        all_data = sum([all_data, data], [])
    all_data.sort(key=lambda tup:tup[0])
    return all_data



def get_score_and_time_all_agents_spaceinvaders_aliens(folder):
    filenames = os.listdir(folder)
    agent_folders = get_files_containing_regex(filenames, '.+agent' + '.+')
    all_data = []
    for s in agent_folders:
        data = get_score_and_time_one_agent_spaceinvaders_score(folder, s)
        all_data = sum([all_data, data], [])
    all_data.sort(key=lambda tup:tup[0])
    return all_data








def get_score_and_counter_all_agents_cartpole(folder):
    filenames = os.listdir(folder)
    agent_folders = get_files_containing_regex(filenames, '.+agent' + '.+')
    all_data = []
    for s in agent_folders:
    	data = get_score_and_counter_one_agent_cartpole(folder, s)
    	all_data = sum([all_data, data], [])
    all_data.sort(key=lambda tup:tup[0])
    return all_data





########################################
#    GET TIME/SCORE FROM ALL AGENTS    #
########################################


def get_score_and_time_all_agents_cartpole(folder):
    filenames = os.listdir(folder)
    agent_folders = get_files_containing_regex(filenames, '.+agent' + '.+')
    all_data = []
    for s in agent_folders:
        data = get_score_and_time_one_agent_cartpole(folder, s)
        all_data = sum([all_data, data], [])
    all_data.sort(key=lambda tup:tup[0])
    return all_data



def get_average_score(input_scores, k):
    input_scores = np.array(input_scores)
    mean_chunk_list = np.mean(input_scores.reshape(-1, k), axis=1)
    return mean_chunk_list



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



def plot_score_run_graph_atari(folder, y_max, x_max ,colour_num, string_number_of_threads, atari_game, k):
    tableau20 = [(31, 119, 180), (174, 199, 232), (255, 127, 14), (255, 187, 120),
                 (44, 160, 44), (152, 223, 138), (214, 39, 40), (255, 152, 150),
                 (148, 103, 189), (197, 176, 213), (140, 86, 75), (196, 156, 148),
                 (227, 119, 194), (247, 182, 210), (127, 127, 127), (199, 199, 199),
                 (188, 189, 34), (219, 219, 141), (23, 190, 207), (158, 218, 229)]
    
    for i in range(len(tableau20)):
        r, g, b = tableau20[i]
        tableau20[i] = (r / 255., g / 255., b / 255.)    
 
    plt.figure(figsize=(16, 10))
          
    ax = plt.subplot(111)
    ax.spines["top"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
        
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()

    plt.ylim(0, y_max)
    plt.xlim(0, x_max + 5)

    print(y_max)
    plt.yticks(range(0, y_max, int(50)), [str(x) for x in range(0, y_max, 50)], fontsize=14)
    plt.xticks(fontsize=14)
      
    # Provide tick lines across the plot to help your viewers trace along    
    # the axis ticks. Make sure that the lines are light and small so they    
    # don't obscure the primary data lines.

    for y in range(0, y_max + 1, int(50)):
        plt.plot(range(0, int(x_max)), [y] * len(range(0, int(x_max))), "-", lw=0.5, color="black", alpha=0.3)
        #plt.plot(range(0, x_points[-1]), [y] * len(range(0, x_points[-1])), "-", lw=0.5, color="black", alpha=0.3)
      
    # Remove the tick marks; they are unnecessary with the tick lines we just plotted.    
    plt.tick_params(axis="both", which="both", bottom="off", top="off",
                    labelbottom="on", left="off", right="off", labelleft="on")

    #plt.plot(range(1000), a3c_2_agents_scores, color=tableau20[0])
    #plt.plot(range(1000), a3c_2_agents_scores[0], color=tableau20[0])
    amount_of_agents = ['1 threads', '2 threads', '4 threads', '8 threads', '16 threads']
    folders = ['space_1_threads', 'space_2_threads', 'space_4_threads', 'space_8_threads', 'space_16_threads']

    for j in range(len(amount_of_agents)):
        data = get_score_and_counter_all_agents_atari(atari_game + folders[j])
        split_data = list(map(list, zip(*data)))
        x_points = split_data[0]
        x_points = get_average_score(x_points, k)
        y_points = split_data[1]
        y_points = y_points[::k]
        plt.plot(x_points, y_points, color=tableau20[j])
        plt.text(int(x_points[-1]) + 0.5, int(y_points[-1]), amount_of_agents[j], fontsize=14, color=tableau20[j])

    plt.title(atari_game, y=1.02)
    plt.xlabel('Time in seconds')
    plt.ylabel('Score')
    plt.savefig("avg_score_pr_episode.png", bbox_inches="tight")






def plot_score_run_graph_spaceinvaders_score(folder, y_max, x_max ,colour_num, string_number_of_threads, atari_game, k):
    tableau20 = [(31, 119, 180), (174, 199, 232), (255, 127, 14), (255, 187, 120),
                 (44, 160, 44), (152, 223, 138), (214, 39, 40), (255, 152, 150),
                 (148, 103, 189), (197, 176, 213), (140, 86, 75), (196, 156, 148),
                 (227, 119, 194), (247, 182, 210), (127, 127, 127), (199, 199, 199),
                 (188, 189, 34), (219, 219, 141), (23, 190, 207), (158, 218, 229)]
    
    for i in range(len(tableau20)):
        r, g, b = tableau20[i]
        tableau20[i] = (r / 255., g / 255., b / 255.)    
 
    plt.figure(figsize=(16, 10))
          
    ax = plt.subplot(111)
    ax.spines["top"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
        
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()

    plt.ylim(0, y_max)
    plt.xlim(0, x_max + 5)

    print(y_max)
    plt.yticks(range(0, y_max, int(50)), [str(x) for x in range(0, y_max, 50)], fontsize=14)
    plt.xticks(fontsize=14)
      
    # Provide tick lines across the plot to help your viewers trace along    
    # the axis ticks. Make sure that the lines are light and small so they    
    # don't obscure the primary data lines.

    for y in range(0, y_max + 1, int(50)):
        plt.plot(range(0, int(x_max)), [y] * len(range(0, int(x_max))), "-", lw=0.5, color="black", alpha=0.3)
        #plt.plot(range(0, x_points[-1]), [y] * len(range(0, x_points[-1])), "-", lw=0.5, color="black", alpha=0.3)
      
    # Remove the tick marks; they are unnecessary with the tick lines we just plotted.    
    plt.tick_params(axis="both", which="both", bottom="off", top="off",
                    labelbottom="on", left="off", right="off", labelleft="on")

    #plt.plot(range(1000), a3c_2_agents_scores, color=tableau20[0])
    #plt.plot(range(1000), a3c_2_agents_scores[0], color=tableau20[0])
    amount_of_agents = ['1 threads', '2 threads', '4 threads', '8 threads', '16 threads']
    folders = ['space_1_threads', 'space_2_threads', 'space_4_threads', 'space_8_threads', 'space_16_threads']

    for j in range(len(amount_of_agents)):
        data = get_score_and_counter_all_agents_spaceinvaders_score(atari_game + folders[j])
        split_data = list(map(list, zip(*data)))
        x_points = split_data[0]
        x_points = get_average_score(x_points, k)
        y_points = split_data[1]
        y_points = y_points[::k]
        plt.plot(x_points, y_points, color=tableau20[j])
        plt.text(int(x_points[-1]) + 0.5, int(y_points[-1]), amount_of_agents[j], fontsize=14, color=tableau20[j])

    plt.title(atari_game, y=1.02)
    plt.xlabel('Time in seconds')
    plt.ylabel('Score')
    plt.savefig("avg_score_pr_episode.png", bbox_inches="tight")




def plot_score_run_graph_spaceinvaders_aliens(folder, y_max, x_max ,colour_num, string_number_of_threads, atari_game, k):
    tableau20 = [(31, 119, 180), (174, 199, 232), (255, 127, 14), (255, 187, 120),
                 (44, 160, 44), (152, 223, 138), (214, 39, 40), (255, 152, 150),
                 (148, 103, 189), (197, 176, 213), (140, 86, 75), (196, 156, 148),
                 (227, 119, 194), (247, 182, 210), (127, 127, 127), (199, 199, 199),
                 (188, 189, 34), (219, 219, 141), (23, 190, 207), (158, 218, 229)]
    
    for i in range(len(tableau20)):
        r, g, b = tableau20[i]
        tableau20[i] = (r / 255., g / 255., b / 255.)    
 
    plt.figure(figsize=(16, 10))
          
    ax = plt.subplot(111)
    ax.spines["top"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
        
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()

    plt.ylim(0, y_max)
    plt.xlim(0, x_max + 5)

    print(y_max)
    plt.yticks(range(0, y_max, int(50)), [str(x) for x in range(0, y_max, 50)], fontsize=14)
    plt.xticks(fontsize=14)
      
    # Provide tick lines across the plot to help your viewers trace along    
    # the axis ticks. Make sure that the lines are light and small so they    
    # don't obscure the primary data lines.

    for y in range(0, y_max + 1, int(50)):
        plt.plot(range(0, int(x_max)), [y] * len(range(0, int(x_max))), "-", lw=0.5, color="black", alpha=0.3)
        #plt.plot(range(0, x_points[-1]), [y] * len(range(0, x_points[-1])), "-", lw=0.5, color="black", alpha=0.3)
      
    # Remove the tick marks; they are unnecessary with the tick lines we just plotted.    
    plt.tick_params(axis="both", which="both", bottom="off", top="off",
                    labelbottom="on", left="off", right="off", labelleft="on")

    #plt.plot(range(1000), a3c_2_agents_scores, color=tableau20[0])
    #plt.plot(range(1000), a3c_2_agents_scores[0], color=tableau20[0])
    amount_of_agents = ['1 threads', '2 threads', '4 threads', '8 threads', '16 threads']
    folders = ['space_1_threads', 'space_2_threads', 'space_4_threads', 'space_8_threads', 'space_16_threads']

    for j in range(len(amount_of_agents)):
        data = get_score_and_counter_all_agents_spaceinvaders_aliens(atari_game + '/' + folders[j])
        split_data = list(map(list, zip(*data)))
        x_points = split_data[0]
        x_points = get_average_score(x_points, k)
        y_points = split_data[1]
        y_points = y_points[::k]
        plt.plot(x_points, y_points, color=tableau20[j])
        plt.text(int(x_points[-1]) + 0.5, int(y_points[-1]), amount_of_agents[j], fontsize=14, color=tableau20[j])

    plt.title('CartPole', y=1.02)
    plt.xlabel('Time in seconds')
    plt.ylabel('Score')
    plt.savefig("avg_score_pr_episode.png", bbox_inches="tight")



def plot_score_run_graph_cart_pole(folder, y_max, x_max ,colour_num, atari_game, k, name_of_file, number_of_threads):
    tableau20 = [(31, 119, 180), (174, 199, 232), (255, 127, 14), (255, 187, 120),
                 (44, 160, 44), (152, 223, 138), (214, 39, 40), (255, 152, 150),
                 (148, 103, 189), (197, 176, 213), (140, 86, 75), (196, 156, 148),
                 (227, 119, 194), (247, 182, 210), (127, 127, 127), (199, 199, 199),
                 (188, 189, 34), (219, 219, 141), (23, 190, 207), (158, 218, 229)]
    
    for i in range(len(tableau20)):
        r, g, b = tableau20[i]
        tableau20[i] = (r / 255., g / 255., b / 255.)    
 
    plt.figure(figsize=(16, 10))
          
    ax = plt.subplot(111)
    ax.spines["top"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
        
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()

    plt.ylim(0, y_max)
    plt.xlim(0, x_max + 5)

    print(y_max)
    plt.yticks(range(0, y_max, int(50)), [str(x) for x in range(0, y_max, 50)], fontsize=14)
    plt.xticks(fontsize=14)
      
    # Provide tick lines across the plot to help your viewers trace along    
    # the axis ticks. Make sure that the lines are light and small so they    
    # don't obscure the primary data lines.

    for y in range(0, y_max + 1, int(50)):
        plt.plot(range(0, int(x_max)), [y] * len(range(0, int(x_max))), "-", lw=0.5, color="black", alpha=0.3)
        #plt.plot(range(0, x_points[-1]), [y] * len(range(0, x_points[-1])), "-", lw=0.5, color="black", alpha=0.3)
      
    # Remove the tick marks; they are unnecessary with the tick lines we just plotted.    
    plt.tick_params(axis="both", which="both", bottom="off", top="off",
                    labelbottom="on", left="off", right="off", labelleft="on")

    #plt.plot(range(1000), a3c_2_agents_scores, color=tableau20[0])
    #plt.plot(range(1000), a3c_2_agents_scores[0], color=tableau20[0])
    amount_of_agents = ['1 threads', '2 threads', '4 threads', '8 threads', '16 threads']
    folders = ['cartpole_1_threads']#, 'cartpole_2_threads', 'cartpole_4_threads', 'cartpole_8_threads']

    for j in range(len(folders)):
        data = get_score_and_counter_all_agents_cartpole(atari_game + '/' + folder)
        print(len(data))
        data = remove_last_elements(data, k)
        split_data = list(map(list, zip(*data)))
        x_points = split_data[0]
        if (k != 1):
            x_points = x_points[::int(k / 2)]
            x_points = x_points[1::2]
        #x_points = x_points[::k]
        y_points = split_data[1]
        if (k != 1):
            y_points = get_average_score(y_points, k)
        np.insert(x_points, 0, 0)
        np.insert(y_points, 0, 0)
        plt.plot(x_points, y_points, color=tableau20[colour_num], label=amount_of_agents[j])
        plt.text(int(x_points[-1]) + 0.5, int(y_points[-1]), number_of_threads, fontsize=14, color=tableau20[colour_num])
    plt.title('CartPole', y=1.02)
    plt.xlabel('Global time steps')
    plt.ylabel('Score')
    plt.savefig(name_of_file, bbox_inches="tight")
    #plt.show()



def plot_score_all_run_graph_cart_pole(y_max, x_max ,colour_num, atari_game, k, name_of_file):
    tableau20 = [(31, 119, 180), (174, 199, 232), (255, 127, 14), (255, 187, 120),
                 (44, 160, 44), (152, 223, 138), (214, 39, 40), (255, 152, 150),
                 (148, 103, 189), (197, 176, 213), (140, 86, 75), (196, 156, 148),
                 (227, 119, 194), (247, 182, 210), (127, 127, 127), (199, 199, 199),
                 (188, 189, 34), (219, 219, 141), (23, 190, 207), (158, 218, 229)]
    
    for i in range(len(tableau20)):
        r, g, b = tableau20[i]
        tableau20[i] = (r / 255., g / 255., b / 255.)    
 
    plt.figure(figsize=(16, 10))
          
    ax = plt.subplot(111)
    ax.spines["top"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
        
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()

    plt.ylim(0, y_max)
    plt.xlim(0, x_max + 5)

    print(y_max)
    plt.yticks(range(0, y_max, int(50)), [str(x) for x in range(0, y_max, 50)], fontsize=20)
    plt.xticks(fontsize=20)
      
    # Provide tick lines across the plot to help your viewers trace along    
    # the axis ticks. Make sure that the lines are light and small so they    
    # don't obscure the primary data lines.

    for y in range(0, y_max + 1, int(50)):
        plt.plot(range(0, int(x_max)), [y] * len(range(0, int(x_max))), "-", lw=0.5, color="black", alpha=0.3)
        #plt.plot(range(0, x_points[-1]), [y] * len(range(0, x_points[-1])), "-", lw=0.5, color="black", alpha=0.3)
      
    # Remove the tick marks; they are unnecessary with the tick lines we just plotted.    
    plt.tick_params(axis="both", which="both", bottom="off", top="off",
                    labelbottom="on", left="off", right="off", labelleft="on")

    #plt.plot(range(1000), a3c_2_agents_scores, color=tableau20[0])
    #plt.plot(range(1000), a3c_2_agents_scores[0], color=tableau20[0])
    amount_of_agents = ['1 threads', '2 threads', '4 threads', '8 threads', '16 threads']
    folders = ['cartpole_1_threads', 'cartpole_2_threads', 'cartpole_4_threads', 'cartpole_8_threads', 'cartpole_16_threads']

    for j in range(len(folders)):
        data = get_score_and_counter_all_agents_cartpole(atari_game + '/' + folders[j])
        data = remove_last_elements(data, k)
        split_data = list(map(list, zip(*data)))
        x_points = split_data[0]
        x_points = x_points[::int(k / 2)]
        x_points = x_points[1::2]
        y_points = split_data[1]
        y_points = get_average_score(y_points, k)
        np.insert(x_points, 0, 0)
        np.insert(y_points, 0, 0)

        plt.plot(x_points, y_points, color=tableau20[colour_num * j])
        if j == 2:
            plt.text(200000+ 0.5, int(y_points[-1] - 7), amount_of_agents[j], fontsize=20, color=tableau20[colour_num * j])
        else:
            plt.text(200000+ 0.5, int(y_points[-1]), amount_of_agents[j], fontsize=20, color=tableau20[colour_num * j])

    plt.title('CartPole', y=1.02, fontsize=20)
    plt.xlabel('Global time steps', fontsize=20)
    plt.ylabel('Score', fontsize=20)
    plt.savefig(name_of_file, bbox_inches="tight")
    #plt.show()



def plot_score_all_run_graph_cart_pole_eligibilty_score(y_max, x_max ,colour_num, k, name_of_file):
    tableau20 = [(31, 119, 180), (174, 199, 232), (255, 127, 14), (255, 187, 120),
                 (44, 160, 44), (152, 223, 138), (214, 39, 40), (255, 152, 150),
                 (148, 103, 189), (197, 176, 213), (140, 86, 75), (196, 156, 148),
                 (227, 119, 194), (247, 182, 210), (127, 127, 127), (199, 199, 199),
                 (188, 189, 34), (219, 219, 141), (23, 190, 207), (158, 218, 229)]
    
    for i in range(len(tableau20)):
        r, g, b = tableau20[i]
        tableau20[i] = (r / 255., g / 255., b / 255.)    
 
    plt.figure(figsize=(16, 10))
          
    ax = plt.subplot(111)
    ax.spines["top"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
        
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()

    plt.ylim(0, y_max)
    plt.xlim(0, x_max + 5)
    plt.yticks(range(0, y_max, int(50)), [str(x) for x in range(0, y_max, 50)], fontsize=20)
    plt.xticks(fontsize=20)

    for y in range(0, y_max + 1, int(50)):
        plt.plot(range(0, int(x_max)), [y] * len(range(0, int(x_max))), "-", lw=0.5, color="black", alpha=0.3)
      
  
    plt.tick_params(axis="both", which="both", bottom="off", top="off",
                    labelbottom="on", left="off", right="off", labelleft="on")

    data = get_score_and_counter_all_agents_cartpole('eligibility' + '/')
    print(len(data))
    data = remove_last_elements(data, k)
    split_data = list(map(list, zip(*data)))
    x_points = split_data[0]
    x_points = x_points[::int(k / 2)]
    x_points = x_points[1::2]
    y_points = split_data[1]
    y_points = get_average_score(y_points, k)
    np.insert(x_points, 0, 0)
    np.insert(y_points, 0, 0)

    plt.plot(x_points, y_points, color=tableau20[colour_num])

    plt.title('CartPole', y=1.02, fontsize=20)
    plt.xlabel('Global time steps', fontsize=20)
    plt.ylabel('Score', fontsize=20)
    plt.savefig(name_of_file, bbox_inches="tight")



def plot_score_all_run_graph_cart_pole_eligibilty_time(y_max, x_max ,colour_num, k, name_of_file):
    tableau20 = [(31, 119, 180), (174, 199, 232), (255, 127, 14), (255, 187, 120),
                 (44, 160, 44), (152, 223, 138), (214, 39, 40), (255, 152, 150),
                 (148, 103, 189), (197, 176, 213), (140, 86, 75), (196, 156, 148),
                 (227, 119, 194), (247, 182, 210), (127, 127, 127), (199, 199, 199),
                 (188, 189, 34), (219, 219, 141), (23, 190, 207), (158, 218, 229)]
    
    for i in range(len(tableau20)):
        r, g, b = tableau20[i]
        tableau20[i] = (r / 255., g / 255., b / 255.)    
 
    plt.figure(figsize=(16, 10))
          
    ax = plt.subplot(111)
    ax.spines["top"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
        
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()

    plt.ylim(0, y_max)
    plt.xlim(0, x_max + 5)
    plt.yticks(range(0, y_max, int(50)), [str(x) for x in range(0, y_max, 50)], fontsize=20)
    plt.xticks(fontsize=20)
      
    # Provide tick lines across the plot to help your viewers trace along    
    # the axis ticks. Make sure that the lines are light and small so they    
    # don't obscure the primary data lines.

    for y in range(0, y_max + 1, int(50)):
        plt.plot(range(0, int(x_max)), [y] * len(range(0, int(x_max))), "-", lw=0.5, color="black", alpha=0.3)
        #plt.plot(range(0, x_points[-1]), [y] * len(range(0, x_points[-1])), "-", lw=0.5, color="black", alpha=0.3)
      
    # Remove the tick marks; they are unnecessary with the tick lines we just plotted.    
    plt.tick_params(axis="both", which="both", bottom="off", top="off",
                    labelbottom="on", left="off", right="off", labelleft="on")

    #plt.plot(range(1000), a3c_2_agents_scores, color=tableau20[0])
    #plt.plot(range(1000), a3c_2_agents_scores[0], color=tableau20[0])
    #amount_of_agents = ['1 threads', '2 threads', '4 threads', '8 threads', '16 threads']
    #folders = ['cartpole_1_threads', 'cartpole_2_threads', 'cartpole_4_threads', 'cartpole_8_threads', 'cartpole_16_threads']

    
    data = get_score_and_time_all_agents_cartpole('eligibility' + '/')
    data = remove_last_elements(data, k)
    split_data = list(map(list, zip(*data)))
    x_points = split_data[0]
    x_points = x_points[::int(k / 2)]
    x_points = x_points[1::2]
    #x_points = x_points[::k]
    y_points = split_data[1]
    y_points = get_average_score(y_points, k)
    np.insert(x_points, 0, 0)
    np.insert(y_points, 0, 0)
    plt.plot(x_points, y_points, color=tableau20[colour_num])
    #plt.text(195000 + 0.5, int(y_points[-1]), amount_of_agents[j], fontsize=14, color=tableau20[colour_num * j])

    plt.title('CartPole', y=1.02, fontsize=20)
    plt.xlabel('Time in seconds', fontsize=20)
    plt.ylabel('Score', fontsize=20)
    plt.savefig(name_of_file, bbox_inches="tight")




def plot_score_eligibilty_vs_A3C(y_max, x_max ,colour_num, atari_game, k, name_of_file):
    tableau20 = [(31, 119, 180), (174, 199, 232), (255, 127, 14), (255, 187, 120),
                 (44, 160, 44), (152, 223, 138), (214, 39, 40), (255, 152, 150),
                 (148, 103, 189), (197, 176, 213), (140, 86, 75), (196, 156, 148),
                 (227, 119, 194), (247, 182, 210), (127, 127, 127), (199, 199, 199),
                 (188, 189, 34), (219, 219, 141), (23, 190, 207), (158, 218, 229)]
    
    for i in range(len(tableau20)):
        r, g, b = tableau20[i]
        tableau20[i] = (r / 255., g / 255., b / 255.)    
 
    plt.figure(figsize=(16, 10))
          
    ax = plt.subplot(111)
    ax.spines["top"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
        
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()

    plt.ylim(0, y_max)
    plt.xlim(0, x_max + 5)

    print(y_max)
    plt.yticks(range(0, y_max, int(50)), [str(x) for x in range(0, y_max, 50)], fontsize=20)
    plt.xticks(fontsize=20)
      
    # Provide tick lines across the plot to help your viewers trace along    
    # the axis ticks. Make sure that the lines are light and small so they    
    # don't obscure the primary data lines.

    for y in range(0, y_max + 1, int(50)):
        plt.plot(range(0, int(x_max)), [y] * len(range(0, int(x_max))), "-", lw=0.5, color="black", alpha=0.3)
        #plt.plot(range(0, x_points[-1]), [y] * len(range(0, x_points[-1])), "-", lw=0.5, color="black", alpha=0.3)
      
    # Remove the tick marks; they are unnecessary with the tick lines we just plotted.    
    plt.tick_params(axis="both", which="both", bottom="off", top="off",
                    labelbottom="on", left="off", right="off", labelleft="on")

    #plt.plot(range(1000), a3c_2_agents_scores, color=tableau20[0])
    #plt.plot(range(1000), a3c_2_agents_scores[0], color=tableau20[0])
    amount_of_agents = ['1 threads', '2 threads', '4 threads', '8 threads', '16 threads']
    folders = ['cartpole_1_threads', 'cartpole_2_threads', 'cartpole_4_threads', 'cartpole_8_threads', 'cartpole_16_threads']
    j = 0

    data = get_score_and_counter_all_agents_cartpole(atari_game + '/' + folders[j])
    data = remove_last_elements(data, k)
    split_data = list(map(list, zip(*data)))
    x_points = split_data[0]
    x_points = x_points[::int(k / 2)]
    x_points = x_points[1::2]
    #x_points = x_points[::k]
    y_points = split_data[1]
    y_points = get_average_score(y_points, k)
    np.insert(x_points, 0, 0)
    np.insert(y_points, 0, 0)
    plt.plot(x_points, y_points, color=tableau20[4 + (j * 2)])

    data = get_score_and_counter_all_agents_cartpole('eligibility' + '/')
    print(len(data))
    data = remove_last_elements(data, k)
    split_data = list(map(list, zip(*data)))
    x_points = split_data[0]
    x_points = x_points[::int(k / 2)]
    x_points = x_points[1::2]
    #x_points = x_points[::k]
    y_points = split_data[1]
    y_points = get_average_score(y_points, k)
    np.insert(x_points, 0, 0)
    np.insert(y_points, 0, 0)
    plt.plot(x_points, y_points, color=tableau20[2])




    plt.title('CartPole', y=1.02, fontsize=20)
    plt.xlabel('Global time steps', fontsize=20)
    plt.ylabel('Score', fontsize=20)
    plt.savefig(name_of_file, bbox_inches="tight")
    #plt.show()



def plot_score_eligibilty_vs_A3C_time(y_max, x_max ,colour_num, atari_game, k, name_of_file):
    tableau20 = [(31, 119, 180), (174, 199, 232), (255, 127, 14), (255, 187, 120),
                 (44, 160, 44), (152, 223, 138), (214, 39, 40), (255, 152, 150),
                 (148, 103, 189), (197, 176, 213), (140, 86, 75), (196, 156, 148),
                 (227, 119, 194), (247, 182, 210), (127, 127, 127), (199, 199, 199),
                 (188, 189, 34), (219, 219, 141), (23, 190, 207), (158, 218, 229)]
    
    for i in range(len(tableau20)):
        r, g, b = tableau20[i]
        tableau20[i] = (r / 255., g / 255., b / 255.)    
 
    plt.figure(figsize=(16, 10))
          
    ax = plt.subplot(111)
    ax.spines["top"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
        
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()


    #plt.plot(range(1000), a3c_2_agents_scores, color=tableau20[0])
    #plt.plot(range(1000), a3c_2_agents_scores[0], color=tableau20[0])
    amount_of_agents = ['1 threads', '2 threads', '4 threads', '8 threads', '16 threads']
    folders = ['cartpole_1_threads', 'cartpole_2_threads', 'cartpole_4_threads', 'cartpole_8_threads', 'cartpole_16_threads']
    j = 0

    data = get_score_and_time_all_agents_cartpole(atari_game + '/' + folders[j])
    data = remove_last_elements(data, k)
    split_data = list(map(list, zip(*data)))
    x_points = split_data[0]
    x_points = x_points[::int(k / 2)]
    x_points = x_points[1::2]
    #x_points = x_points[::k]
    y_points = split_data[1]
    y_points = get_average_score(y_points, k)
    np.insert(x_points, 0, 0)
    np.insert(y_points, 0, 0)



    plt.ylim(0, y_max)
    plt.xlim(0, int(x_points[-1]) + 5)

    print(y_max)
    plt.yticks(range(0, y_max, int(50)), [str(x) for x in range(0, y_max, 50)], fontsize=20)
    plt.xticks(fontsize=20)
      
    # Provide tick lines across the plot to help your viewers trace along    
    # the axis ticks. Make sure that the lines are light and small so they    
    # don't obscure the primary data lines.

    for y in range(0, y_max + 1, int(50)):
        plt.plot(range(0, int(x_max)), [y] * len(range(0, int(x_max))), "-", lw=0.5, color="black", alpha=0.3)
        #plt.plot(range(0, x_points[-1]), [y] * len(range(0, x_points[-1])), "-", lw=0.5, color="black", alpha=0.3)
      
    # Remove the tick marks; they are unnecessary with the tick lines we just plotted.    
    plt.tick_params(axis="both", which="both", bottom="off", top="off",
                    labelbottom="on", left="off", right="off", labelleft="on")





    plt.plot(x_points, y_points, color=tableau20[4 + (j * 2)])

    data = get_score_and_time_all_agents_cartpole('eligibility' + '/')
    data = remove_last_elements(data, k)
    split_data = list(map(list, zip(*data)))
    x_points = split_data[0]
    x_points = x_points[::int(k / 2)]
    x_points = x_points[1::2]
    y_points = split_data[1]
    y_points = get_average_score(y_points, k)
    np.insert(x_points, 0, 0)
    np.insert(y_points, 0, 0)
    plt.plot(x_points, y_points, color=tableau20[2])




    plt.title('CartPole', y=1.02, fontsize=20)
    plt.xlabel('Time in seconds', fontsize=20)
    plt.ylabel('Score', fontsize=20)
    plt.savefig(name_of_file, bbox_inches="tight")
    #plt.show()





def plot_score_all_run_graph_cart_pole_time(y_max, x_max ,colour_num, atari_game, k, name_of_file):
    tableau20 = [(31, 119, 180), (174, 199, 232), (255, 127, 14), (255, 187, 120),
                 (44, 160, 44), (152, 223, 138), (214, 39, 40), (255, 152, 150),
                 (148, 103, 189), (197, 176, 213), (140, 86, 75), (196, 156, 148),
                 (227, 119, 194), (247, 182, 210), (127, 127, 127), (199, 199, 199),
                 (188, 189, 34), (219, 219, 141), (23, 190, 207), (158, 218, 229)]
    
    for i in range(len(tableau20)):
        r, g, b = tableau20[i]
        tableau20[i] = (r / 255., g / 255., b / 255.)    
 
    plt.figure(figsize=(16, 10))
          
    ax = plt.subplot(111)
    ax.spines["top"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
        
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()

    plt.ylim(0, y_max)
    plt.xlim(0, x_max + 5)

    print(y_max)
    plt.yticks(range(0, y_max, int(50)), [str(x) for x in range(0, y_max, 50)], fontsize=20)
    plt.xticks(fontsize=20)
      
    # Provide tick lines across the plot to help your viewers trace along    
    # the axis ticks. Make sure that the lines are light and small so they    
    # don't obscure the primary data lines.

    for y in range(0, y_max + 1, int(50)):
        plt.plot(range(0, int(x_max)), [y] * len(range(0, int(x_max))), "-", lw=0.5, color="black", alpha=0.3)
        #plt.plot(range(0, x_points[-1]), [y] * len(range(0, x_points[-1])), "-", lw=0.5, color="black", alpha=0.3)
      
    # Remove the tick marks; they are unnecessary with the tick lines we just plotted.    
    plt.tick_params(axis="both", which="both", bottom="off", top="off",
                    labelbottom="on", left="off", right="off", labelleft="on")

    #plt.plot(range(1000), a3c_2_agents_scores, color=tableau20[0])
    #plt.plot(range(1000), a3c_2_agents_scores[0], color=tableau20[0])
    amount_of_agents = ['1 threads', '2 threads', '4 threads', '8 threads', '16 threads']
    folders = ['cartpole_1_threads', 'cartpole_2_threads', 'cartpole_4_threads', 'cartpole_8_threads', 'cartpole_16_threads']

    for j in range(len(folders)):
        data = get_score_and_time_all_agents_cartpole(atari_game + '/' + folders[j])
        data = remove_last_elements(data, k)
        split_data = list(map(list, zip(*data)))
        x_points = split_data[0]
        x_points = x_points[::int(k / 2)]
        x_points = x_points[1::2]
        y_points = split_data[1]
        y_points = get_average_score(y_points, k)
        np.insert(x_points, 0, 0)
        np.insert(y_points, 0, 0)

        plt.plot(x_points, y_points, color=tableau20[colour_num * j])
        if j == 3:
            plt.text(x_max+ 5.5, int(y_points[-1] + 7), amount_of_agents[j], fontsize=20, color=tableau20[colour_num * j])
        elif j == 2:
            plt.text(x_max+ 5.5, int(y_points[-1] - 7), amount_of_agents[j], fontsize=20, color=tableau20[colour_num * j])
        else:
            plt.text(x_max+ 5.5, int(y_points[-1]), amount_of_agents[j], fontsize=20, color=tableau20[colour_num * j])

    plt.title('CartPole', y=1.02, fontsize=20)
    plt.xlabel('Time in seconds', fontsize=20)
    plt.ylabel('Score', fontsize=20)
    plt.savefig(name_of_file, bbox_inches="tight")
    #plt.show()




########################################
#              TO DOOO                 #
########################################

'''
def find_max_score_at_step():
    folders = ['cartpole_1_threads', 'cartpole_2_threads', 'cartpole_4_threads', 'cartpole_8_threads', 'cartpole_16_threads']
    j = 0
    data = get_score_and_time_all_agents_cartpole('cartpole_true'+ '/' + folders[j])
    data = remove_last_elements(data, k)
    split_data = list(map(list, zip(*data)))
    a = max(data,key=itemgetter(0))[0]
    b = max(data,key=itemgetter(1))[0]
    print(a)
    print(b)



def find_max_score_at_time():
    data = get_score_and_time_all_agents_cartpole('cartpole_true' + '/' + folders[j])
    split_data = list(map(list, zip(*data)))

'''


































def plot_score_all_run_graph_spcaeinvaders_counter_score(y_max, x_max ,colour_num, atari_game, k, name_of_file):
    tableau20 = [(31, 119, 180), (174, 199, 232), (255, 127, 14), (255, 187, 120),
                 (44, 160, 44), (152, 223, 138), (214, 39, 40), (255, 152, 150),
                 (148, 103, 189), (197, 176, 213), (140, 86, 75), (196, 156, 148),
                 (227, 119, 194), (247, 182, 210), (127, 127, 127), (199, 199, 199),
                 (188, 189, 34), (219, 219, 141), (23, 190, 207), (158, 218, 229)]
    
    for i in range(len(tableau20)):
        r, g, b = tableau20[i]
        tableau20[i] = (r / 255., g / 255., b / 255.)    
 
    plt.figure(figsize=(16, 10))
          
    ax = plt.subplot(111)
    ax.spines["top"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
        
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()

    plt.ylim(0, y_max)
    plt.xlim(0, x_max + 5)

    print(y_max)
    plt.yticks(range(0, y_max, int(50)), [str(x) for x in range(0, y_max, 50)], fontsize=20)
    plt.xticks(fontsize=20)
      
    # Provide tick lines across the plot to help your viewers trace along    
    # the axis ticks. Make sure that the lines are light and small so they    
    # don't obscure the primary data lines.

    for y in range(0, y_max + 1, int(200)):
        plt.plot(range(0, int(x_max)), [y] * len(range(0, int(x_max))), "-", lw=0.5, color="black", alpha=0.3)
        #plt.plot(range(0, x_points[-1]), [y] * len(range(0, x_points[-1])), "-", lw=0.5, color="black", alpha=0.3)
      
    # Remove the tick marks; they are unnecessary with the tick lines we just plotted.    
    plt.tick_params(axis="both", which="both", bottom="off", top="off",
                    labelbottom="on", left="off", right="off", labelleft="on")

    #plt.plot(range(1000), a3c_2_agents_scores, color=tableau20[0])
    #plt.plot(range(1000), a3c_2_agents_scores[0], color=tableau20[0])
    amount_of_agents = ['1 threads', '2 threads', '4 threads', '8 threads', '16 threads']
    folders = ['spaceInvaders_16_threads']

    for j in range(len(folders)):
        data = get_score_and_counter_all_agents_spaceinvaders_aliens(atari_game + '/' + folders[j])
        data = remove_last_elements(data, k)
        split_data = list(map(list, zip(*data)))
        x_points = split_data[0]
        if k != 1:
            x_points = x_points[::int(k / 2)]
            x_points = x_points[1::2]
        y_points = split_data[1]
        if k != 1:
            y_points = get_average_score(y_points, k)
        np.insert(x_points, 0, 0)
        np.insert(y_points, 0, 0)

        plt.plot(x_points, y_points, color=tableau20[colour_num * j])

        plt.text(x_max + 0.5, int(y_points[-1]), amount_of_agents[j], fontsize=20, color=tableau20[colour_num * j])

    plt.title('SpaceInvaders', y=1.02, fontsize=20)
    plt.xlabel('Global time steps', fontsize=20)
    plt.ylabel('Score', fontsize=20)
    plt.savefig(name_of_file, bbox_inches="tight")
    #plt.show()













import matplotlib.pyplot as plt
import numpy as np
import os
import re

def get_data_points_from_file(fname, folder):
    data = np.genfromtxt(folder + '/' + fname, delimiter=',')
    return data

# Data point is [episode, score, time in seconds]
def separate_data(data):
    return data[:,0], data[:,1], data[:,2]

def get_files_containing_regex(filelist, regex):
    pattern = re.compile(regex)
    matches = []
    for f in filelist:
        match = re.findall(pattern, f)
        if match:
            matches.append(match[0])
    return matches

def tex_gen(data):
    pass

def test():
    return 1, 2




def plot_time_for_one_agent(folder, number_of_agents):
    filenames = os.listdir(folder)
    total_score  = 0
    total_time = 0
    counter = 0
    for i in range(number_of_agents):
        agent_folders = get_files_containing_regex(filenames, '.+agent_' + str(i) +'.+')
        print(agent_folders)
        for s in agent_folders:
            counter += 1
            data = get_data_points_from_file(s, folder)
            episodes, scores, times = separate_data(data)
            total_score += np.array(scores)
            total_time += np.array(times)
    total_score = total_score / counter
    total_time = total_time / counter
    total_time = [ int(x) for x in total_time ]
    plot_time_graph(total_score, total_time)




def get_variance_from_an_agent(folder, number_of_agents, k, eps):
    filenames = os.listdir(folder)
    score_list = []
    for i in range(number_of_agents):
        agent_folders = get_files_containing_regex(filenames, '.+agent_' + str(i) +'.+')
        for s in agent_folders:
            data = get_data_points_from_file(s, folder)
            episodes, scores, times = separate_data(data)
            score_list.append(scores)
    mean_list = 0
    min_list = 0
    max_list = 0

    for i in range(len(score_list)):
        mean_list += np.array(score_list[i])
    mean_list = mean_list / (len(score_list))


    mean_chunk_list = np.mean(mean_list.reshape(-1, k), axis=1)

    min_values_list = []

    for i in range(len(score_list)): 
        min_values = np.min(np.array(score_list[i]).reshape(-1, k), axis=1)
        min_values_list.append(min_values)

    min_values_list2 = []
    iterations = int(eps / k)
    for i in range(iterations):
        min_value = min(x[i] for x in min_values_list)
        min_values_list2.append(min_value)



    max_values_list = []
    for i in range(len(score_list)): 
        max_values = np.max(np.array(score_list[i]).reshape(-1, k), axis=1)
        max_values_list.append(max_values)

    max_values_list2 = []
    for i in range(iterations):
        max_value = max(x[i] for x in max_values_list)
        max_values_list2.append(max_value)


    plot_variance_graph(mean_chunk_list, min_values_list2, max_values_list2, eps, k)



def get_average_score_for_every_agent_in_folder_k(folder, max_agents, k):
    total_score = 0
    for i in range(max_agents):
        score = get_average_score_from_an_agent(folder, i)
        total_score += np.array(score)
    average_score = total_score / max_agents
    mean_chunk_list = np.mean(average_score.reshape(-1, k), axis=1)
    return mean_chunk_list




def get_average_score_from_an_agent(folder, number_of_agents):
    filenames = os.listdir(folder)
    agent_folders = get_files_containing_regex(filenames, '.+agent_' + str(number_of_agents) +'.+')
    print(agent_folders)
    total_score = 0
    counter = 0
    for s in agent_folders:
        counter += 1
        data = get_data_points_from_file(s, folder)
        episodes, scores, times = separate_data(data)
        total_score += np.array(scores)
    total_score = total_score / counter
    return total_score


def get_average_score_for_every_agent_in_folder(folder, max_agents):
    total_score = 0
    for i in range(max_agents):
        score = get_average_score_from_an_agent(folder, i)
        total_score += np.array(score)
    average_score = total_score / max_agents
    return average_score


def plot_variance_graph(mean, mini, maxi, episodes, k):
        # These are the "Tableau 20" colors as RGB.    
    tableau20 = [(31, 119, 180), (174, 199, 232), (255, 127, 14), (255, 187, 120),    
                 (44, 160, 44), (152, 223, 138), (214, 39, 40), (255, 152, 150),    
                 (148, 103, 189), (197, 176, 213), (140, 86, 75), (196, 156, 148),    
                 (227, 119, 194), (247, 182, 210), (127, 127, 127), (199, 199, 199),    
                 (188, 189, 34), (219, 219, 141), (23, 190, 207), (158, 218, 229)]    
      
    # Scale the RGB values to the [0, 1] range, which is the format matplotlib accepts.    
    for i in range(len(tableau20)):    
        r, g, b = tableau20[i]    
        tableau20[i] = (r / 255., g / 255., b / 255.)    
      
    # You typically want your plot to be ~1.33x wider than tall. This plot is a rare    
    # exception because of the number of lines being plotted on it.    
    # Common sizes: (10, 7.5) and (12, 9)    
    plt.figure(figsize=(16, 10))    
      
    # Remove the plot frame lines. They are unnecessary chartjunk.    
    ax = plt.subplot(111)    
    ax.spines["top"].set_visible(False)    
    ax.spines["bottom"].set_visible(False)    
    ax.spines["right"].set_visible(False)    
    ax.spines["left"].set_visible(False)    
      
    # Ensure that the axis ticks only show up on the bottom and left of the plot.    
    # Ticks on the right and top of the plot are generally unnecessary chartjunk.    
    ax.get_xaxis().tick_bottom()    
    ax.get_yaxis().tick_left()    
      
    # Limit the range of the plot to only where the data is.    
    # Avoid unnecessary whitespace.    
    plt.ylim(0, 205)    
    plt.xlim(0, episodes + 5)
      
    # Make sure your axis ticks are large enough to be easily read.    
    # You don't want your viewers squinting to read your plot.    
    plt.yticks(range(0, 201, 10), [str(x) for x in range(0, 201, 10)], fontsize=14)    
    plt.xticks(fontsize=14)    
      
    # Provide tick lines across the plot to help your viewers trace along    
    # the axis ticks. Make sure that the lines are light and small so they    
    # don't obscure the primary data lines.    
    for y in range(10, 201, 10):    
        plt.plot(range(0, episodes), [y] * len(range(0, episodes)), "-", lw=0.5, color="black", alpha=0.3)    
      
    # Remove the tick marks; they are unnecessary with the tick lines we just plotted.    
    plt.tick_params(axis="both", which="both", bottom="off", top="off",    
                    labelbottom="on", left="off", right="off", labelleft="on")    
      
    amount_of_agents = ['mean', 'min', 'max']
    epsi = (int(episodes / k))
    plt.plot(range(0, episodes, k), mean, color=tableau20[0])
    plt.text(episodes + 0.5, mean[-1], amount_of_agents[0], fontsize=14, color=tableau20[0])    
    plt.plot(range(0, episodes, k), mini, color=tableau20[5])    
    plt.text(episodes + 0.5, mini[-1], amount_of_agents[1], fontsize=14, color=tableau20[5])    
    plt.plot(range(0, episodes, k), maxi, color=tableau20[10])    
    plt.text(episodes + 0.5, maxi[-1], amount_of_agents[2], fontsize=14, color=tableau20[10])    


 
    plt.title('Average result of different amounts of threads for the CartPole problem', y=1.02)
    plt.xlabel('Episodes')
    plt.ylabel('Average of the average of the agents')
      
    # Always include your data source(s) and copyright notice! And for your    
    # data sources, tell your viewers exactly where the data came from,    
    # preferably with a direct link to the data. Just telling your viewers    
    # that you used data from the "U.S. Census Bureau" is completely useless:    
    # the U.S. Census Bureau provides all kinds of data, so how are your    
    # viewers supposed to know which data set you used?    
      
    # Finally, save the figure as a PNG.    
    # You can also save it as a PDF, JPEG, etc.    
    # Just change the file extension in this call.    
    # bbox_inches="tight" removes all the extra whitespace on the edges of your plot.    
    #plt.show()
    plt.savefig("avg_score_pr_episode.png", bbox_inches="tight")






def plot_graph(input_scores, episodes, k):
        # These are the "Tableau 20" colors as RGB.    
    tableau20 = [(31, 119, 180), (174, 199, 232), (255, 127, 14), (255, 187, 120),    
                 (44, 160, 44), (152, 223, 138), (214, 39, 40), (255, 152, 150),    
                 (148, 103, 189), (197, 176, 213), (140, 86, 75), (196, 156, 148),    
                 (227, 119, 194), (247, 182, 210), (127, 127, 127), (199, 199, 199),    
                 (188, 189, 34), (219, 219, 141), (23, 190, 207), (158, 218, 229)]    
      
    # Scale the RGB values to the [0, 1] range, which is the format matplotlib accepts.    
    for i in range(len(tableau20)):    
        r, g, b = tableau20[i]    
        tableau20[i] = (r / 255., g / 255., b / 255.)    
      
    # You typically want your plot to be ~1.33x wider than tall. This plot is a rare    
    # exception because of the number of lines being plotted on it.    
    # Common sizes: (10, 7.5) and (12, 9)    
    plt.figure(figsize=(16, 10))    
      
    # Remove the plot frame lines. They are unnecessary chartjunk.    
    ax = plt.subplot(111)    
    ax.spines["top"].set_visible(False)    
    ax.spines["bottom"].set_visible(False)    
    ax.spines["right"].set_visible(False)    
    ax.spines["left"].set_visible(False)    
      
    # Ensure that the axis ticks only show up on the bottom and left of the plot.    
    # Ticks on the right and top of the plot are generally unnecessary chartjunk.    
    ax.get_xaxis().tick_bottom()    
    ax.get_yaxis().tick_left()    
      
    # Limit the range of the plot to only where the data is.    
    # Avoid unnecessary whitespace.    
    plt.ylim(0, 501)    
    plt.xlim(0, episodes + 5)
      
    # Make sure your axis ticks are large enough to be easily read.    
    # You don't want your viewers squinting to read your plot.    
    plt.yticks(range(0, 501, 100), [str(x) for x in range(0, 501, 100)], fontsize=14)    
    plt.xticks(fontsize=14)    
      
    # Provide tick lines across the plot to help your viewers trace along    
    # the axis ticks. Make sure that the lines are light and small so they    
    # don't obscure the primary data lines.    
    for y in range(0, 501, 50):    
        plt.plot(range(0, episodes), [y] * len(range(0, episodes)), "-", lw=0.5, color="black", alpha=0.3)    
      
    # Remove the tick marks; they are unnecessary with the tick lines we just plotted.    
    plt.tick_params(axis="both", which="both", bottom="off", top="off",    
                    labelbottom="on", left="off", right="off", labelleft="on")    
      
    amount_of_agents = ['1 threads', '2 threads', '4 threads', '8 threads', '16 threads']

    #plt.plot(range(1000), a3c_2_agents_scores, color=tableau20[0])
    #plt.plot(range(1000), a3c_2_agents_scores[0], color=tableau20[0])    
    epsi = (int(episodes / k))
    for rank in range(len(input_scores)):
        # Plot each line separately with its own color, using the Tableau 20    
        # color set in order.
        plt.plot(range(0, episodes, k), input_scores[rank], color=tableau20[rank])    
      
        # Add a text label to the right end of every line. Most of the code below    
        # is adding specific offsets y position because some labels overlapped.    
        #y_pos = input_scores[rank][-1] - 0.5
        y_pos = input_scores[rank][-1]
        # Again, make sure that all labels are large enough to be easily read    
        # by the viewer.    
        plt.text(episodes + 0.5, y_pos, amount_of_agents[rank], fontsize=14, color=tableau20[rank])    

    # matplotlib's title() call centers the title on the plot, but not the graph,    
    # so I used the text() call to customize where the title goes.    
      
    # Make the title big enough so it spans the entire plot, but don't make it    
    # so big that it requires two lines to show.    
      
    # Note that if the title is descriptive enough, it is unnecessary to include    
    # axis labels; they are self-evident, in this plot's case.    
    #plt.text(1995, 93, "Percentage of Bachelor's degrees conferred to women in the U.S.A."    
    #       ", by major (1970-2012)", fontsize=17, ha="center")    
    plt.title('Average score using different amounts of threads for Spaceinvaders', y=1.02)
    plt.xlabel('Episodes')
    plt.ylabel('Average score')
      
    # Always include your data source(s) and copyright notice! And for your    
    # data sources, tell your viewers exactly where the data came from,    
    # preferably with a direct link to the data. Just telling your viewers    
    # that you used data from the "U.S. Census Bureau" is completely useless:    
    # the U.S. Census Bureau provides all kinds of data, so how are your    
    # viewers supposed to know which data set you used?    
      
    # Finally, save the figure as a PNG.    
    # You can also save it as a PDF, JPEG, etc.    
    # Just change the file extension in this call.    
    # bbox_inches="tight" removes all the extra whitespace on the edges of your plot.    
    #plt.show()
    plt.savefig("avg_score_pr_episode.png", bbox_inches="tight")  


def plot_time_graph(input_scores, input_times):
        # These are the "Tableau 20" colors as RGB.    
    tableau20 = [(31, 119, 180), (174, 199, 232), (255, 127, 14), (255, 187, 120),    
                 (44, 160, 44), (152, 223, 138), (214, 39, 40), (255, 152, 150),    
                 (148, 103, 189), (197, 176, 213), (140, 86, 75), (196, 156, 148),    
                 (227, 119, 194), (247, 182, 210), (127, 127, 127), (199, 199, 199),    
                 (188, 189, 34), (219, 219, 141), (23, 190, 207), (158, 218, 229)]    
      
    # Scale the RGB values to the [0, 1] range, which is the format matplotlib accepts.    
    for i in range(len(tableau20)):    
        r, g, b = tableau20[i]    
        tableau20[i] = (r / 255., g / 255., b / 255.)    
      
    # You typically want your plot to be ~1.33x wider than tall. This plot is a rare    
    # exception because of the number of lines being plotted on it.    
    # Common sizes: (10, 7.5) and (12, 9)    
    plt.figure(figsize=(16, 10))    
      
    # Remove the plot frame lines. They are unnecessary chartjunk.    
    ax = plt.subplot(111)    
    ax.spines["top"].set_visible(False)    
    ax.spines["bottom"].set_visible(False)    
    ax.spines["right"].set_visible(False)    
    ax.spines["left"].set_visible(False)    
      
    # Ensure that the axis ticks only show up on the bottom and left of the plot.    
    # Ticks on the right and top of the plot are generally unnecessary chartjunk.    
    ax.get_xaxis().tick_bottom()    
    ax.get_yaxis().tick_left()    
      
    # Limit the range of the plot to only where the data is.    
    # Avoid unnecessary whitespace.    
    plt.ylim(0, 205)    
    plt.xlim(0, input_times[-1] + 5)
      
    # Make sure your axis ticks are large enough to be easily read.    
    # You don't want your viewers squinting to read your plot.    
    plt.yticks(range(0, 201, 10), [str(x) for x in range(0, 201, 10)], fontsize=14)    
    plt.xticks(fontsize=14)    
      
    # Provide tick lines across the plot to help your viewers trace along    
    # the axis ticks. Make sure that the lines are light and small so they    
    # don't obscure the primary data lines.  
    for y in range(10, 201, 10):    
        plt.plot(range(0, input_times[-1]), [y] * len(range(0, input_times[-1])), "-", lw=0.5, color="black", alpha=0.3)    
      
    # Remove the tick marks; they are unnecessary with the tick lines we just plotted.    
    plt.tick_params(axis="both", which="both", bottom="off", top="off",    
                    labelbottom="on", left="off", right="off", labelleft="on")    
      
    amount_of_agents = ['1 threads', '2 threads', '4 threads', '8 threads', '16 threads']

    #plt.plot(range(1000), a3c_2_agents_scores, color=tableau20[0])
    #plt.plot(range(1000), a3c_2_agents_scores[0], color=tableau20[0])    
    plt.plot(input_times, input_scores, color=tableau20[1])
    y_pos = input_scores[-1]
    plt.text(input_times[-1] + 0.5, y_pos, amount_of_agents[1], fontsize=14, color=tableau20[1])    

    # matplotlib's title() call centers the title on the plot, but not the graph,    
    # so I used the text() call to customize where the title goes.    
      
    # Make the title big enough so it spans the entire plot, but don't make it    
    # so big that it requires two lines to show.    
      
    # Note that if the title is descriptive enough, it is unnecessary to include    
    # axis labels; they are self-evident, in this plot's case.    
    #plt.text(1995, 93, "Percentage of Bachelor's degrees conferred to women in the U.S.A."    
    #       ", by major (1970-2012)", fontsize=17, ha="center")    
    plt.title('Average result of different amounts of threads for the CartPole problem', y=1.02)
    plt.xlabel('Time in seconds')
    plt.ylabel('Average of the average of the agents')
      
    # Always include your data source(s) and copyright notice! And for your    
    # data sources, tell your viewers exactly where the data came from,    
    # preferably with a direct link to the data. Just telling your viewers    
    # that you used data from the "U.S. Census Bureau" is completely useless:    
    # the U.S. Census Bureau provides all kinds of data, so how are your    
    # viewers supposed to know which data set you used?    
      
    # Finally, save the figure as a PNG.    
    # You can also save it as a PDF, JPEG, etc.    
    # Just change the file extension in this call.    
    # bbox_inches="tight" removes all the extra whitespace on the edges of your plot.    
    #plt.show()
    plt.savefig("avg_score_pr_episode.png", bbox_inches="tight")  



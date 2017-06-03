import matplotlib.pyplot as plt
import pandas as pd  
import os 
import re
from data_handler import * 

avg_scores = []
folders = ['a3c_1_agents', 'a3c_2_agents', 'a3c_3_agents', 'a3c_4_agents']
for f in folders:
    filenames = os.listdir(f)
    if f == 'a3c_4_agents':
        agent_1_runs = get_files_containing_regex(filenames, '.+agent_1.+') 
        total_score_1 = 0
        for s in agent_1_runs: 
            data = get_data_points_from_file(s, f)
            episodes, scores, times = separate_data(data)
            total_score_1 += np.array(scores)


        agent_2_runs = get_files_containing_regex(filenames, '.+agent_2.+') 
        total_score_2 = 0
        for s in agent_2_runs: 
            data = get_data_points_from_file(s, f)
            episodes, scores, times = separate_data(data)
            total_score_2 += np.array(scores)
    
        agent_3_runs = get_files_containing_regex(filenames, '.+agent_3.+') 
        total_score_3 = 0
        for s in agent_3_runs: 
            data = get_data_points_from_file(s, f)
            episodes, scores, times = separate_data(data)
            total_score_3 += np.array(scores)
        
        agent_4_runs = get_files_containing_regex(filenames, '.+agent_4.+') 
        total_score_4 = 0
        for s in agent_4_runs: 
            data = get_data_points_from_file(s, f)
            episodes, scores, times = separate_data(data)
            total_score_4 += np.array(scores)
        avg_scores.append((total_score_1 + total_score_2 + total_score_3 + total_score_4) / (len(agent_1_runs) + len(agent_2_runs) + len(agent_3_runs) + len(agent_4_runs)))

    elif f == 'a3c_3_agents':
        agent_1_runs = get_files_containing_regex(filenames, '.+agent_1.+') 
        total_score_1 = 0
        for s in agent_1_runs: 
            data = get_data_points_from_file(s, f)
            episodes, scores, times = separate_data(data)
            total_score_1 += np.array(scores)


        agent_2_runs = get_files_containing_regex(filenames, '.+agent_2.+') 
        total_score_2 = 0
        for s in agent_2_runs: 
            data = get_data_points_from_file(s, f)
            episodes, scores, times = separate_data(data)
            total_score_2 += np.array(scores)
    
        agent_3_runs = get_files_containing_regex(filenames, '.+agent_3.+') 
        total_score_3 = 0
        for s in agent_3_runs: 
            data = get_data_points_from_file(s, f)
            episodes, scores, times = separate_data(data)
            total_score_3 += np.array(scores)
        avg_scores.append((total_score_1 + total_score_2 + total_score_3) / (len(agent_1_runs) + len(agent_2_runs) + len(agent_3_runs)))
   
    elif f == 'a3c_2_agents':
        agent_1_runs = get_files_containing_regex(filenames, '.+2000.+agent_1.+') 
        total_score_1 = 0
        for s in agent_1_runs: 
            data = get_data_points_from_file(s, f)
            episodes, scores, times = separate_data(data)
            total_score_1 += np.array(scores)


        agent_2_runs = get_files_containing_regex(filenames, '.+2000.+agent_2.+') 
        total_score_2 = 0
        for s in agent_1_runs: 
            data = get_data_points_from_file(s, f)
            episodes, scores, times = separate_data(data)
            total_score_2 += np.array(scores)
        avg_scores.append((total_score_1 + total_score_2) / (len(agent_1_runs) + len(agent_2_runs)))

#    if f == 'a3c_3_agents' or f == 'a3c_4_agents':
#        short_runs = get_files_containing_regex(filenames, '.+2000.+')
    else:
        short_runs = get_files_containing_regex(filenames, '.+2000.+')
        long_runs = get_files_containing_regex(filenames, '.+4000.+')
    
        # Represent data as graphs and tex for short runs
        all_scores = []
        total_score = 0
        for s in short_runs:
            data = get_data_points_from_file(s, f)
            episodes, scores, times = separate_data(data)
            total_score += np.array(scores)
        avg_scores.append(total_score / len(short_runs))


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
plt.xlim(0, 2005)
  
# Make sure your axis ticks are large enough to be easily read.    
# You don't want your viewers squinting to read your plot.    
plt.yticks(range(0, 201, 10), [str(x) for x in range(0, 201, 10)], fontsize=14)    
plt.xticks(fontsize=14)    
  
# Provide tick lines across the plot to help your viewers trace along    
# the axis ticks. Make sure that the lines are light and small so they    
# don't obscure the primary data lines.    
for y in range(10, 201, 10):    
    plt.plot(range(0, 2000), [y] * len(range(0, 2000)), "-", lw=0.5, color="black", alpha=0.3)    
  
# Remove the tick marks; they are unnecessary with the tick lines we just plotted.    
plt.tick_params(axis="both", which="both", bottom="off", top="off",    
                labelbottom="on", left="off", right="off", labelleft="on")    
  
amount_of_agents = ['1 thread', '2 threads', '3 threads', '4 threads']
  
for rank, column in enumerate(amount_of_agents):    
    # Plot each line separately with its own color, using the Tableau 20    
    # color set in order.    
    plt.plot(range(2000),    
            avg_scores[rank],    
            lw=1.5, color=tableau20[rank])    
  
    # Add a text label to the right end of every line. Most of the code below    
    # is adding specific offsets y position because some labels overlapped.    
    y_pos = avg_scores[rank][-1] - 0.5    
    if column == "1 thread":    
        y_pos -= 5    
    elif column == "2 threads":    
        y_pos -= 0.5    
    elif column == "Communications\nand Journalism":    
        y_pos += 0.75    
    elif column == "Art and Performance":    
        y_pos -= 0.25    
    elif column == "Agriculture":    
        y_pos += 1.25    
    elif column == "Social Sciences and History":    
        y_pos += 0.25    
    elif column == "Business":    
        y_pos -= 0.75    
    elif column == "Math and Statistics":    
        y_pos += 0.75    
    elif column == "Architecture":    
        y_pos -= 0.75    
    elif column == "Computer Science":    
        y_pos += 0.75    
    elif column == "Engineering":    
        y_pos -= 0.25    
  
    # Again, make sure that all labels are large enough to be easily read    
    # by the viewer.    
    plt.text(2000.5, y_pos, column, fontsize=14, color=tableau20[rank])    
  
# matplotlib's title() call centers the title on the plot, but not the graph,    
# so I used the text() call to customize where the title goes.    
  
# Make the title big enough so it spans the entire plot, but don't make it    
# so big that it requires two lines to show.    
  
# Note that if the title is descriptive enough, it is unnecessary to include    
# axis labels; they are self-evident, in this plot's case.    
#plt.text(1995, 93, "Percentage of Bachelor's degrees conferred to women in the U.S.A."    
#       ", by major (1970-2012)", fontsize=17, ha="center")    
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

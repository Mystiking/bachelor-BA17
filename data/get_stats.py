import numpy as np
import matplotlib.pyplot as plt
import re, os

def combine_files(filenames):
    total = np.loadtxt(filenames[0], delimiter=',')
    for f in filenames[1:]:
        tmp = np.loadtxt(f, delimiter=',')
        total = np.append(total, tmp, axis=0)
    result = []
    for ts, s, t, ak in total:
        result.append((ts, s, t, ak))

    #print(result)
    return sorted(result, key=lambda a: a[3])

def get_k_mean_score(l, k):
    c = 0
    result_scores = []
    result_timesteps = []
    scores = []
    timesteps = []
    for ts, s, t, ak in l:
        scores.append(s)
        timesteps.append(ak)
        if c % k == 0:
            result_scores.append(sum(scores) / len(scores))
            result_timesteps.append(sum(timesteps) / len(timesteps))
            scores = []
            timesteps = []
        c += 1
    return result_scores, result_timesteps

def get_k_mean_time(l, k):
    c = 0
    result_scores = []
    result_times = []
    scores = []
    times= []
    for ts, s, t, ak in l:
        scores.append(s)
        times.append(t)
        if c % k == 0:
            result_scores.append(sum(scores) / len(scores))
            result_times.append(sum(times) / len(times))
            scores = []
            times = []
        c += 1
    return result_scores, result_times


def get_matching_files(folder, regex):
    files = os.listdir(folder)
    result = []
    for f in files:
        try:
            if len(re.findall(regex, f)) > 0 :
                
                result.append(folder + (re.findall(regex, f))[0])
        except:
            pass
    return result

def pretty_plot(xs, ys, title, xlabel, ylabel, name_of_file, ymax = None):
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

    x_max = np.max(xs)
    y_max = np.max(ys) if ymax is None else ymax
    print(y_max)
    print(x_max)
    print(int(x_max))

    plt.ylim(0, y_max)
    plt.xlim(0, x_max + 5)

    plt.yticks(range(0, int(y_max), int(50)), [str(x) for x in range(0, int(y_max), 50)], fontsize=30)
    plt.xticks(fontsize=30)
    for x in range(len(xs)):
        plt.plot(xs[x], ys[x], color=tableau20[10], lw=1.5)

    for y in range(0, int(y_max) + 1, int(50)):
        plt.plot(range(0, int(x_max), 10), [y for x in range(0, int(x_max), 10)], color="black", lw=2.5, alpha=0.3)
      
    plt.tick_params(axis="both", which="both", bottom="off", top="off",
                    labelbottom="on", left="off", right="off", labelleft="on")

    plt.xlabel(xlabel, fontsize=30)
    plt.ylabel(ylabel, fontsize=30)
    plt.title(title, fontsize=30)
    plt.savefig(name_of_file, bbox_inches="tight")
    #plt.show()


#files = (get_matching_files('spaceInvaders/spaceInvaders_16_threads/', 'space.+'))
files = (get_matching_files('cartpole_true/cartpole_8_threads/', 'cartpole.+'))
l = combine_files(files)
#print(l)
r1, r2 = (get_k_mean_score(l, 5))
pretty_plot([r2], [r1], "CartPole", "Timesteps", "Score", "cartpole_.png", 201)
#r1, r2 = (get_k_mean_time(l, 5))
#print(r2)
#pretty_plot([r2], [r1], "CartPole", "Time in seconds", "Score", "test.png", 200)

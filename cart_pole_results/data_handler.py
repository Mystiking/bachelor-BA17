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

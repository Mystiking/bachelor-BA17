import numpy as np
import os
import tensorflow as tf

flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_integer('threads', 1, 'folder')



def get_order(filenames, folder):
    start_times_and_timesteps = []
    for f in filenames:
        tmp = open(folder + f, 'r')
        first_line = tmp.readline().split(',')
        time = first_line[2]
        timesteps = first_line[0]
        start_times_and_timesteps.append((f, float(time), int(timesteps)))
    
    start_times_and_timesteps.sort(key=lambda a : a[2])
    print("Sorted by timesteps:")
    for f, t, ts in start_times_and_timesteps:
        print(f, t, ts)
    return start_times_and_timesteps 

def get_differences(l):
    to_be_added = [(l[0][0], 0)]
    dif = l[0][1]
    for f, t, ts in l[1:]:
        dif += (t - dif)
        to_be_added.append((f, dif))
    for f, d in to_be_added:
        print(f, d)
    return to_be_added

def fix_file(f, d, new_f):
    writer = open(new_f, 'w')
    reader = open(f, 'r')
    for line in reader.readlines():
        line = line.split(',')
        time = float(line[2]) + d
        writer.write("{},{},{},{}".format(line[0], line[1], time, line[3]))
    writer.close()
    reader.close()
    print("Success!")

def fix_files(difs, folder):
    for f, d in difs:
       fix_file(folder + f, 1, folder + 'test' + f) 

fnames = os.listdir('spaceInvaders_{}_threads'.format(FLAGS.threads))
order = get_order(fnames, 'spaceInvaders_{}_threads/'.format(FLAGS.threads))
difs = get_differences(order)
fix_files(difs, 'spaceInvaders_{}_threads/'.format(FLAGS.threads))

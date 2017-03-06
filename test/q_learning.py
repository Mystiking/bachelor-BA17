import numpy as np
import random
import math
import draw
import matplotlib.pyplot as plt

actions = ['l', 'r', 'u', 'd']
states = [{} for i in range(0, 48)]
for s in states:
    s['l'] = 0
    s['r'] = 0
    s['u'] = 0
    s['d'] = 0

def get_max_keys(d):
    res = []
    for key in d:
        if not res:
            res.append(key)
        else:
            for k in res:
                if d[key] > d[k]:
                    res.remove(k)
                    res.append(key)
                elif d[key] == d[k] and key not in res:
                    res.append(key)
    return res

def move(s, a):
    if a == 'l':
        if s % 12 == 0:
            return s
        else:
            return s - 1
    if a == 'r':
        if s % 12 == 11:
            return s
        else:
            return s + 1
    if a == 'u':
        if s // 12 == 0:
            return s
        else:
            return s - 12
    if a == 'd':
        if s // 12 == 3:
            return s
        else:
            return s + 12

episodes = 1000 
epsilon = 0.1
alpha = 0.5
gamma = 0.9
# Moves list
ms = []
while(episodes > 0):
    episodes -= 1
    print("Episode", episodes)
    rand = random.random()
    # Starting state
    state = 36
    if rand <= epsilon:
        a = actions[math.floor(rand*4)]
    else:
        rewards = []
        for r in states[state]:
           rewards.append(states[state][r])
        best_actions = np.argwhere(rewards == np.amax(rewards))
        best_actions = [x for [x] in best_actions]
        a = actions[np.random.choice(best_actions)]
    d = draw.Draw(1024, 1024)
    d.fill_rectangle(3, 11, 4, 12, 0, 255, 0)
    d.fill_rectangle(3, 0, 4, 12, 0, 0, 255)
    for x in range(10):
        d.fill_rectangle(3, 1 + x, 4, 12, 255, 0, 0)
    d.draw_grid(4, 12)
    total_r = 0
    while True:
        rand = random.random()
        if rand <= epsilon:
            a = actions[math.floor(rand*40)]
        else:
            best_actions = get_max_keys(states[state])
            # Choose A from S unsing policy derived from Q
            a = np.random.choice(best_actions)
        # Take action A and observe R, S'
        new_state = int(move(state, a))
        if new_state in range(37, 47):
            reward = -100
            new_state = 36
        else:
            reward = -1

        a_optimal = np.random.choice(get_max_keys(states[new_state]))
        states[state][a] += alpha * (reward + gamma * states[new_state][a_optimal] - states[state][a])
        
        ### Drawing
        if a == 'u':
            d.draw_up_arrow(state // 12, state % 12, 4, 12)
        elif a == 'd':
            d.draw_down_arrow(state // 12, state % 12, 4, 12)
        elif a == 'l':
            d.draw_left_arrow(state // 12, state % 12, 4, 12)
        elif a == 'r':
            d.draw_right_arrow(state // 12, state % 12, 4, 12)

        state = new_state
        total_r += reward
        if state == 47:
            d.write("q_learning/q_learning_" + str(episodes) + ".png")
            ms.append(total_r)
            break

def k_average(rewards, k):
    result = []
    for r in range(len(rewards)):
        avg = 0
        amnt = 0
        for x in range(-4, 6):
            if r - x >= 0 and r - x < len(rewards):
                avg += rewards[r - x]
                amnt += 1
        result.append(avg / amnt)
    return result
ms = k_average(ms, 10)
plt.plot([i for i in range(len(ms))], ms, 'r')
plt.ylim([-100, 0])
plt.show()


import numpy as np
import random
import math
import draw
import matplotlib.pyplot as plt

actions = ['l', 'r', 'u', 'd', 'lu', 'ld', 'ru', 'rd']
wind = np.zeros((10, 3))
for w in range(len(wind)):
    if w % 10 == 3 or w % 10 == 4 or w % 10 == 5 or w % 10 == 8:
        wind[w] = [0, 1, 2]
    elif w % 10 == 6 or w % 10 == 7:
        wind[w] = [1, 2, 3]
    else:
        wind[w] = [0, 0, 0]

states = [{} for i in range(0, 70)]
for s in states:
    s['l'] = 0
    s['r'] = 0
    s['u'] = 0
    s['d'] = 0
    s['ld'] = 0
    s['rd'] = 0
    s['lu'] = 0
    s['ru'] = 0

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

def add_wind(s, wind):
    w = 10 * np.random.choice(wind[s % 10])
    if s - w < 0:
        return s % 10
    else:
        return s - w 


def move(s, a, wind):
    if a == 'l':
        if s % 10 == 0:
            return add_wind(s, wind)
        else:
            return add_wind(s - 1, wind)
    if a == 'r':
        if s % 10 == 9:
            return add_wind(s, wind)
        else:
            return add_wind(s + 1, wind)
    if a == 'u':
        if s // 10 == 0:
            return add_wind(s, wind)
        else:
            return add_wind(s - 10, wind)
    if a == 'd':
        if s // 10 == 6:
            return add_wind(s, wind)
        else:
            return add_wind(s + 10, wind)
    if a == 'ld':
        if s % 10 == 0 or s // 10 == 6:
            return add_wind(s, wind)
        else:
            return add_wind(s + 9, wind)
    if a == 'rd':
        if s % 10 == 9 or s // 10 == 6:
            return add_wind(s, wind)
        else:
            return add_wind(s + 11, wind)
    if a == 'lu':
        if s % 10 == 0 or s // 10 == 0:
            return add_wind(s, wind)
        else:
            return add_wind(s - 11, wind)
    if a == 'ru':
        if s % 10 == 9 or s // 10 == 0:
            return add_wind(s, wind)
        else:
            return add_wind(s - 9, wind)


episodes = 1000 
epsilon = 0.1
alpha = 1.0
gamma = 0.9
# Moves list
ms = []
while(episodes > 0):
    episodes -= 1
    rand = random.random()
    # Starting state
    state = 30
    if rand <= epsilon:
        a = actions[math.floor(rand*8)]
    else:
        rewards = []
        for r in states[state]:
           rewards.append(states[state][r])
        best_actions = np.argwhere(rewards == np.amax(rewards))
        best_actions = [x for [x] in best_actions]
        a = actions[np.random.choice(best_actions)]
    d = draw.Draw(1024, 1024)
    d.draw_grid(7, 10)
    d.fill_rectangle(3, 7, 7, 10, 0, 255, 0)
    d.fill_rectangle(3, 0, 7, 10, 0, 0, 255)
    moves = 0
    while True:
        rand = random.random()
        if rand <= epsilon:
            a = actions[math.floor(rand*8)]
        else:
            rewards = []
            best_actions = get_max_keys(states[state])
            a = np.random.choice(best_actions)
        new_state = int(move(state, a, wind))
        a_new  = np.random.choice(get_max_keys(states[new_state]))
        reward = -1 + states[new_state][a_new]

        states[state][a] += alpha * (reward + gamma * states[new_state][a_new] - states[state][a])
        
        ### Drawing
        if a == 'u':
            d.draw_up_arrow(state // 10, state % 10, 7, 10)
        elif a == 'd':
            d.draw_down_arrow(state // 10, state % 10, 7, 10)
        elif a == 'l':
            d.draw_left_arrow(state // 10, state % 10, 7, 10)
        elif a == 'r':
            d.draw_right_arrow(state // 10, state % 10, 7, 10)
        elif a == 'ld':
            d.draw_left_down_arrow(state // 10, state % 10, 7, 10)
        elif a == 'rd':
            d.draw_right_down_arrow(state // 10, state % 10, 7, 10)
        elif a == 'ru':
            d.draw_right_up_arrow(state // 10, state % 10, 7, 10)
        elif a == 'lu':
            d.draw_left_up_arrow(state // 10, state % 10, 7, 10)
        state = new_state
        moves += 1
        if state == 37:
            d.write("sarsa/sarsa_" + str(episodes) + ".png")
            ms.append(moves)
            break

plt.plot([i for i in range(1000)], ms, 'r')
#plt.ylim(0, 30)

plt.show()


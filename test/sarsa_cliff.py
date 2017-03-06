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

episodes = 500 
epsilon = 0.3
alpha = 0.5
gamma = 0.9
# Moves list
ms = []
while(episodes > 0):
    episodes -= 1
    rand = random.random()
    # Starting state
    state = 36
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
    d.fill_rectangle(3, 11, 4, 12, 0, 255, 0)
    d.fill_rectangle(3, 0, 4, 12, 0, 0, 255)
    for x in range(10):
        d.fill_rectangle(3, 1 + x, 4, 12, 255, 0, 0)
    d.draw_grid(4, 12)
    moves = 0
    while True:
        new_state = int(move(state, a))
        rand = random.random()
        if rand <= epsilon:
            a_new = actions[math.floor(rand*4)]
        else:
            a_new  = np.random.choice(get_max_keys(states[new_state]))
        if new_state in range(37, 47):
            reward = -100
            new_state = 36
        else:
            reward = -1

        states[state][a] += alpha * (reward + gamma * states[new_state][a_new] - states[state][a])
        
        ### Drawing
        if a == 'u':
            d.draw_up_arrow(state // 12, state % 12, 4, 12)
        elif a == 'd':
            d.draw_down_arrow(state // 12, state % 12, 4, 12)
        elif a == 'l':
            d.draw_left_arrow(state // 12, state % 12, 4, 12)
        elif a == 'r':
            d.draw_right_arrow(state // 12, state % 12, 4, 12)
        elif a == 'ld':
            d.draw_left_down_arrow(state // 10, state % 10, 7, 10)
        elif a == 'rd':
            d.draw_right_down_arrow(state // 10, state % 10, 7, 10)
        elif a == 'ru':
            d.draw_right_up_arrow(state // 10, state % 10, 7, 10)
        elif a == 'lu':
            d.draw_left_up_arrow(state // 10, state % 10, 7, 10)
        state = new_state
        a = a_new
        moves += 1
        if state == 47:
            d.write("sarsa/sarsa_cliff_" + str(episodes) + ".png")
            ms.append(moves)
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

### Drawing the policy

dp = draw.Draw(1024, 1024)
dp.draw_grid(4, 12)
for s in range(len(states)):
    max_keys = get_max_keys(states[s])
    if 'l' in max_keys:
        dp.draw_left_arrow(s // 12, s % 12, 4, 12)
    if 'r' in max_keys:
        dp.draw_right_arrow(s // 12, s % 12, 4, 12)
    if 'u' in max_keys:
        dp.draw_up_arrow(s // 12, s % 12, 4, 12)
    if 'd' in max_keys:
        dp.draw_down_arrow(s // 12, s % 12, 4, 12)
dp.fill_rectangle(3, 11, 4, 12, 0, 255, 0)
dp.fill_rectangle(3, 0, 4, 12, 0, 0, 255)
for x in range(10):
    dp.fill_rectangle(3, 1 + x, 4, 12, 255, 0, 0)
dp.write('sarsa_cliff_policy_optimal.png')
#ms = k_average(ms, 10)
#plt.plot([i for i in range(len(ms))], ms, 'r')
#plt.ylim([-100, 0])
#plt.show()


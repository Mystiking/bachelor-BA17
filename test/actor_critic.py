import draw
import numpy as np

def softmax(s):
    x = np.array([s['u'], s['d'], s['l'], s['r']])
    return np.exp(x - np.max(x)) / np.sum(np.exp(x - np.max(x)))

def getBestAction(x, actions):
    return np.random.choice(actions, p=x)

def getReward(s):
    if s == 3:
        return 1
    elif s == 7:
        return -1
    else:
        return r

def move(s, a):
    if a == 'l':
        if s % 4 == 0 or s == 6:
            return s
        else:
            return s - 1
    if a == 'r':
        if s % 4 == 3 or s == 4:
            return s
        else:
            return s + 1
    if a == 'u':
        if s // 4 == 0 or s == 9:
            return s
        else:
            return s - 4
    if a == 'd':
        if s // 4 == 2 or s == 1:
            return s
        else:
            return s + 4


# Discounts
alpha = 0.1
beta = 1.0
gamma = 0.9
_lambda = 0.81 

# Constants
episodes = 100
r = -0.04

actions = ['u', 'd', 'l' , 'r']


states = [{} for i in range(0, 12)]
for s in states:
    s['u'] = 0
    s['l'] = 0
    s['d'] = 0 
    s['r'] = 0
    s['reward'] = 0

states[3]['reward'] = 1
states[7]['reward'] = -1

e_traces_s = [0 for i in range(0, 12)]
e_traces_as = [{} for i in range(0, 12)]
for e in e_traces_as:
    e['u'] = 0
    e['l'] = 0
    e['d'] = 0 
    e['r'] = 0


while episodes > 0:
    current_state = 8
    d = draw.Draw(1024, 1024)
    d.draw_grid(3, 4)
    d.fill_rectangle(0, 3, 4, 3, 0, 255, 0)
    d.fill_rectangle(1, 1, 4, 3, 0, 0, 0)
    d.fill_rectangle(1, 3, 4, 3, 255, 0, 0)

    while current_state != 3 and current_state != 7:
        for i in range(len(e_traces_s)):
            if i == current_state:
                e_traces_s[i] = 1 + gamma * _lambda * e_traces_s[i]
            else:
                e_traces_s[i] = gamma * _lambda * e_traces_s[i]
        action = getBestAction(softmax(states[current_state]), actions)
        next_state = move(current_state, action) 
        td_error = getReward(next_state) + gamma*states[next_state]['reward'] - states[current_state]['reward']
        states[current_state]['reward'] = states[current_state]['reward'] + alpha * td_error * e_traces_s[current_state]
        for i in range(len(e_traces_as)):
            if i == current_state:
                e_traces_as[i][action] = 1 + alpha * _lambda * e_traces_as[i][action]
                for a in actions:
                    if action != a:
                        e_traces_as[i][a] = alpha * _lambda * e_traces_as[i][a]
            else:
                for a in actions:
                    e_traces_as[i][a] = alpha * _lambda * e_traces_as[i][a]
    
        states[current_state][action] = states[current_state][action] + alpha * td_error * e_traces_as[current_state][action]

        if action == 'u':
            d.draw_up_arrow(current_state // 4, current_state % 4, 3, 4)
        elif action == 'd':
            d.draw_down_arrow(current_state // 4, current_state % 4, 3, 4)
        elif action == 'l':
            d.draw_left_arrow(current_state // 4, current_state % 4, 3, 4)
        elif action == 'r':
            d.draw_right_arrow(current_state // 4, current_state % 4, 3, 4)
        current_state = next_state
    d.write("actor_critic/actor_critic_test{}.png".format(episodes))
    episodes -= 1

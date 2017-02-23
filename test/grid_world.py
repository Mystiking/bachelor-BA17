import numpy as np

world = np.zeros((4,4))
actions = ['l', 'r', 'u', 'd']
states = [i for i in range(0, 16)]
policy = [[0.25, 0.25, 0.25, 0.25] for i in range(0, 16)]

print(world)
print(actions)
print(states)
print(policy)

def move(s, a):
    if a == 'l':
        if s % 4 == 0:
            return s
        else:
            return s - 1
    if a == 'r':
        if s % 4 == 3:
            return s
        else:
            return s + 1
    if a == 'u':
        if s // 4 == 0:
            return s
        else:
            return s - 4
    if a == 'd':
        if s // 4 == 3:
            return s
        else:
            return s + 4



def policy_evaluation(S, pi, actions, world):
    theta = 0.001
    gamma = 1
    new_world = np.copy(world)
    while True:
        delta = 0
        for s in S:
            if s == 0 or s == 15:
                pass
            else:
                index = (s // 4, s % 4)
                v = world[index]
                S_new = []
                for a in actions:
                    S_new.append(move(s, a)) 
                val = 0
                for s_n in range(len(S_new)):
                    val += pi[s][s_n] * (-1 + gamma * world[S_new[s_n] // 4, S_new[s_n] % 4])
                new_world[index] = val
                delta = max(delta, np.abs(v - new_world[index]))
        world = np.copy(new_world)
        if delta < theta:
            return world

def reward(s, a, world):
    next_state = move(s, a)
    return -1 + world[next_state // 4, next_state % 4]

def policy_improvement(S, pi, actions, world, k):
    while k > 0:
        k -= 1
        world = policy_evaluation(S, pi, actions, world)
        print(world)
        stable = True
        for s in S:
            old_action = pi[s]
            rewards = []
            for a in actions:
                r = reward(s, a, world)
                rewards.append(r)
            new_action = np.argwhere(rewards == np.amax(rewards))
            new_action.reshape((new_action.shape[0], ))
            for i in range(4):
                if i in new_action:
                    pi[s][i] = 1 / len(new_action)
                else:
                    pi[s][i] = 0
            if old_action != pi[s]:
                stable = False
        if stable:
            return world, pi

w, p = policy_improvement(states, policy, actions, world, 9)
for x in range(len(p)):
    print('State {}: policy {}'.format(x, p[x]))


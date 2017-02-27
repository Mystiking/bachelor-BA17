import numpy as np

vs = np.zeros(400)
S = []
for i in range(20):
    for j in range(20):
        S.append((i, j))


def evaluation(S, pi, gamma, vals):
    theta = 0.1
    while True:
        delta = 0
        for s in range(len(S)):
            v = vals[s]
            vals[s] = 
        if delta < theta:
            break

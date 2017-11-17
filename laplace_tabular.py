import sys
import numpy as np
from numpy import linalg as LA
import pickle
import env
import options
import plot_utils


np.set_printoptions(precision=2)

# 10 x 10 grid
max_row = 10
max_col = 10

max_actions = 4 # right, left, down, up

states_rc = []
for r in range(max_row):
    for c in range(max_row):
        states_rc.append((r, c))

total_states = len(states_rc)

adjacency = np.zeros((total_states, total_states), dtype = np.int)

# set env dimension
command = "dim:{},{}".format(max_row, max_col)
env.env_message(command)

# set no goal
env.env_message("no_goal")

# Compute adjacency matrix (take all possible actions from every state)
for state in range(total_states):
    for a in range(max_actions):
        command = "start_state:{}".format(state)
        env.env_message(command)
        
        result = env.env_step(a)
        next_state = result["state"][0]
        if next_state != state:
            adjacency[state][next_state] = 1


D = np.zeros((total_states, total_states), dtype = np.int)

row_sum = np.sum(adjacency, axis=1)
for state in range(total_states):
   D[state][state] = row_sum[state]

diff = D - adjacency
sq_D = np.sqrt(D) # Diagonal matrix so element-wise operation is ok
L = np.matmul(sq_D, np.matmul(diff, sq_D))

# extract eigenvalue(w), eigenvector(v)
w, v = LA.eig(L)
#plot_utils.print_eigen(w, v)
print w

# TODO: Add options incrementally
# Getting the most interesting eigen vector iv
idx = np.argmin(w)
iv = v[:, idx]
print iv

opt = options.Option(iv, max_row, max_col, alpha=0.1, epsilon=0.1, discount=1)
opt.learn(100000)
opt.display_policy()

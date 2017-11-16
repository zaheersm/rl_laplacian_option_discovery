import sys
import numpy as np
from numpy import linalg as LA
import pickle
import env
import options
import plot_utils
# shouldn't use this anymore
def pprint_pi(pi, max_row, max_col):
    action_set = ['R', 'L', 'D', 'U', 'T']
    count = 0
    for r in range((max_row)):
        for c in range((max_col)):
            sys.stdout.write(action_set[pi[count]] + ' ')
            count+=1
        print '\n'

def print_eigen(e_vals, e_vecs):
    for idx in range(len(e_vals)):
        print 'Eigen Value: ', e_vals[idx]
        print 'Eigen Vector: \n', e_vecs[idx]
        print '-'*20

np.set_printoptions(precision=2)

# 10 x 10 grid
max_row = 10
max_col = 10

max_actions = 4

states_rc = []
for r in range(max_row):
    for c in range(max_row):
        states_rc.append((r, c))

total_states = len(states_rc)

adjacency = np.zeros((total_states, total_states), dtype = np.int)

command = "dim:{},{}".format(max_row, max_col)
env.env_message(command)
env.env_message("no_goal")
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

w, v = LA.eig(L)
#print_eigen(w, v)

print w
# Getting the most interesting eigen vector iv
idx = np.argmin(w)
iv = v[:, idx]
print iv

opt = options.Option(iv, max_row, max_col, alpha=0.1, epsilon=0.1, discount=1)
opt.learn(10000)
opt.display_policy()

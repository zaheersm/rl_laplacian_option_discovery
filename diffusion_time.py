import numpy as np
import environment

import sys

def pprint_value(V, max_row, max_col):
    for r in range(max_row):
        for c in range(max_col):
            val = '{:3.2f}'.format(V[r][c])
            sys.stdout.write(str(val) + ' ')
        print '\n'

def load_option_policies(num_options=4):
    policies = []
    for idx in range(num_options):
        in_name = 'option_archive/option{}.txt'.format(idx)
        opt = np.loadtxt(in_name, dtype=np.int)
        opt = opt.reshape((10,10))
        policies.append(opt)
    return policies

def simulate_opt(opt, states_rc, start_state, goal_state):
    env = environment.GridEnvironment(max_row, max_col, (-1, -1))
    env.set_start_state(states_rc[start_state])
    env.start()
    current_state = start_state
    cr, cc = states_rc[current_state]
    action = int(opt[cr][cc])
    steps = 0
    while action != TERMINAL:
        if current_state == goal_state:
            return steps, current_state
        result = env.step(action)
        current_state = result["state"][0]
        steps += 1
        cr, cc = states_rc[current_state]
        action = int(opt[cr][cc])
    return steps, current_state

max_row, max_col = 10, 10
default_action_set = [(0,1), (0, -1), (1, 0), (-1, 0)] # R, L, D, U
TERMINAL = 4
states_rc = [(r, c) for r in range(max_row)
             for c in range(max_col)]

opt_policies = load_option_policies(200)

cache = {}
for opt_id in range(len(opt_policies)):
    for g_id in range(len(states_rc)):
        for s_id in range(len(states_rc)):
            steps, ns = simulate_opt(opt_policies[opt_id], states_rc,
                                     s_id, g_id)
            cache[(opt_id, s_id, g_id)] = (steps, ns)

MAX_OPTIONS = 200
diffusion_time = np.zeros((MAX_OPTIONS + 1,))
options_range = range(MAX_OPTIONS + 1)[::-1]
for num_options in options_range:
    means = []
    for g_id, GOAL in enumerate(states_rc):
        # print g_id
        V = np.zeros((max_row, max_col))
        tolerance = 0.01
        env = environment.GridEnvironment(max_row, max_col,  (-1, -1))
        delta = np.inf
        while delta > tolerance:
            delta = 0
            for s_id, s in enumerate(states_rc):
                if s == GOAL: continue
                r, c = s[0], s[1]
                v = V[r][c]
                va = 0
                p = 1./(len(default_action_set) + num_options)
                for action in range(len(default_action_set)):
                    env.set_current_state(s_id)
                    result = env.step(action)
                    ns = states_rc[result["state"][0]]
                    nr, nc = ns[0], ns[1]
                    if ns == GOAL:
                        va += (p)*(1)*(0 + (1)*V[nr][nc])
                    else:
                        va += (p)*(1)*(1 + (1)*V[nr][nc])
                for opt_id in range(num_options):
                    opt = opt_policies[opt_id]
                    steps, ns = cache[(opt_id, s_id, g_id)]
                    ns = states_rc[ns]
                    nr, nc = ns[0], ns[1]
                    va += (p)*(1)*(steps + (1)*V[nr][nc])
                V[r][c] = va
                delta = max(delta, abs(V[r][c] - v))
                #sys.stdout.write(str(delta) + '\r')
                #sys.stdout.flush()
                #pprint_value(V, max_row, max_col)	
        #print np.mean(V)
        means.append(np.mean(V))
        sys.stdout.write('. ')
        if (g_id + 1) % 10 == 0:
            sys.stdout.write('\n')
        sys.stdout.flush()
    out = 'Num Options: {} | Diffusion Time: {}'.format(num_options,
                                                        np.mean(means))
    print out
    diffusion_time[num_options] = np.mean(means)
    # Saving in every iteration
    savename = 'data_files/diffusion_time_values_0_200.txt'
    np.savetxt(savename, diffusion_time, fmt='%f')

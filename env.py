import numpy as np

max_row = 10
max_col = 10

states_rc = [] 
for r in range(max_row):
    for c in range(max_row):
        states_rc.append((r, c))

# right, left, up, down
action_set = [(0,1), (0,-1), (1,0), (-1,0)]
MAX_ACTION = len(action_set)

goal_state = (0, 0)

def env_init():
    pass

def env_start():
    global current_state, states_rc
    current_state = np.asarray([rand_in_range(len(states_rc))])
    return current_state

def env_step(action):
    global current_state, goal_state, action_set, MAX_ACTION

    if not action < MAX_ACTION:
        print "Invalid action taken!!"
        print "action: ", action
        print "current_state: ", current_state
        exit(1)
    action = action_set[action]

    s = current_state[0]
    s = states_rc[s]
    nr = min(max_row - 1, max(0, s[0] + action[0]))
    nc = min(max_col - 1, max(0, s[1] + action[1]))
    s = (nr, nc)
    reward = -1.0
    is_terminal = False
    if np.array_equal(s, goal_state):
        is_terminal = True
        current_state = None
    else:
        current_state[0] = states_rc.index(s)
    
    result = {"reward" : reward, "state" : current_state,
            "isTerminal" : is_terminal}
    
    return result

def env_cleanup():
    return

def env_message(in_message):
    global current_state, goal_state, max_row, max_col, states_rc
    # Helper messages to help in adjacency matrix
    if in_message.startswith("start_state"):
        current_state = np.asarray([int(in_message.split(":")[1])])
    elif in_message.startswith("no_goal"):
        goal_state = (-1, -1)
    elif in_message.startswith("dim"):
        dims = in_message.split(":")[1].split(",")
        max_row, max_col = int(dims[0]), int(dims[1])
	states_rc = []
	for r in range(max_row):
            for c in range(max_row):
                states_rc.append((r, c))
    return ""

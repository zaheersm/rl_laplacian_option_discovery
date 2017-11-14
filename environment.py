import pickle
import numpy as np

class Environment(object):
    
    def __init__ (self, reward_vector=None, max_row=3, max_col=3, goal_state=(0,0)):
        states_rc = []
        for r in range(max_row):
            for c in range(max_col):
                states_rc.append((r,c))
        self.states_rc = states_rc

        self.max_row, self.max_col = max_row, max_col

        self.goal_state = goal_state

        # Need to figure out a termination technique
        self.action_set = [(0,1), (0,-1), (1,0), (-1,0), (-1, -1)]
        self.MAX_ACTION = len(self.action_set)

        if reward_vector is None:
            # Reward is -1.0 everywhere
            self.reward_vector = np.zeros((len(self.states_rc))) * -1
        else:
            self.reward_vector = reward_vector
        #print self.reward_vector

    def start(self):
        start_state = np.random.randint(len(self.states_rc))
        self.current_state = np.asarray([start_state])
        #self.current_state[0] = self.states_rc.index((3,0))
        # Returning a copy of the current state
        return np.copy(self.current_state)

    def step(self, action):
        if not action < self.MAX_ACTION:
            print "Invalid action taken!!"
            print "action: ", action
            print "current_state", self.current_state

        action = self.action_set[action]
        if action == (-1, -1):
            self.current_state = None
            result = {"reward": 0, "state":None, "isTerminal":True}
            return result
        s = self.current_state[0]
        # Getting the coordinate representation of the state
        s = self.states_rc[s]
        nr = min(self.max_row - 1, max(0, s[0] + action[0]))
        nc = min(self.max_col - 1, max(0, s[1] + action[1]))
        ns = (nr, nc)
        # Going back to the integer representation
        s = self.states_rc.index(s)
        ns = self.states_rc.index(ns)
        reward = self.reward_vector[ns] - self.reward_vector[s]
        self.current_state[0] = ns
        """
        #TODO: Never terminates
        is_terminal = False
        if np.array_equal(ns, self.states_rc.index(self.goal_state)):
            is_terminal = True
            self.current_state = None
        """
        result = {"reward" : reward, "state" : self.current_state,
                "isTerminal" : False}

        return result

    def cleanup(self):
        return

    def message(self, in_message):
    	# Helper messages to help in adjacency matrix
	if in_message.startswith("start_state"):
	    self.current_state = np.asarray([int(in_message.split(":")[1])])
	elif in_message.startswith("no_goal"):
	    self.goal_state = (-1, -1)
	elif in_message.startswith("dim"):
	    dims = in_message.split(":")[1].split(",")
	    max_row, max_col = int(dims[0]), int(dims[1])
	    states_rc = []
	    for r in range(max_row):
		for c in range(max_row):
		    states_rc.append((r, c))
	    self.max_row, self.max_col = max_row, max_col
            self.states_rc = states_rc
	elif in_message.startswith("reward_vec"):
	    self.reward_vector = pickle.loads(in_message.split(":")[1])
	return ""

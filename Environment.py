import numpy as np
import pickle


# Possible default actions in tabular environment
default_action_set = [(0, 1), (0, -1), (1, 0), (-1, 0)] # R, L, D, U

TERMINATE_ACTION = (0,0)


class GridEnvironment(object):

    def __init__(self, max_row, max_col,
                 goal_state, obstacle_vector = None, reward_vector = None):

        states_rc = [(r, c) for r in range(max_row) for c in range(max_col)]
        self.states_rc = states_rc # all possible states (r,c)
        self.max_row, self.max_col = max_row, max_col

        self.goal_state = goal_state

        self.action_set = default_action_set
        self.default_max_actions = len(self.action_set) # will stay fixed
        self.max_actions = len(self.action_set) # can increase

        if reward_vector is None:
            # Reward is 0.0 everywhere, and 1.0 in goal state
            self.reward_vector = np.zeros((len(self.states_rc))) * 0.0
            goal_idx = self.states_rc.index(goal_state)
            self.reward_vector[goal_idx] = 1.0
        else:
            self.reward_vector = reward_vector

        self.current_state = None

    def start(self):
        start_state = np.random.randint(len(self.states_rc))
        self.current_state = np.asarray([start_state])
        # Returning a copy of the current state
        return np.copy(self.current_state)

    def step(self, action):
        if not action < self.max_actions:
            print "Invalid action taken!!"
            print "action: ", action
            print "current_state", self.current_state

        action = self.action_set[action]
        if action == TERMINATE_ACTION:
            self.current_state = None
            result = {"reward": 0, "state": None, "isTerminal": True}
            return result
        s = self.current_state[0]
        # Getting the coordinate representation of the state
        s = self.states_rc[s]
        nr = min(self.max_row - 1, max(0, s[0] + action[0]))
        nc = min(self.max_col - 1, max(0, s[1] + action[1]))
        ns = (nr, nc) 

        # Going back to the integer representation
        s = self.states_rc.index(s)
        ns = self.states_rc.index(ns) # next state

        reward = self.reward_vector[ns] - self.reward_vector[s]
        self.current_state[0] = ns

        
        if self.goal_state != (-1,-1) and \
            self.states_rc.index(self.goal_state) == self.current_state[0]:
            
            self.current_state = None
            result = {"reward": reward, "state": None, "isTerminal": True}
            return result

        else:
            result = {"reward": reward, "state": self.current_state,
                      "isTerminal": False}
            return result

    def cleanup(self):

        return

    def message(self, in_message):
        if in_message.startswith("set start_state"):
            self.current_state = np.asarray([int(in_message.split(":")[1])])

        elif in_message.startswith("set terminate_action"):
            self.action_set.append(TERMINATE_ACTION)
            self.max_actions = len(self.action_set)

        elif in_message.startswith("set no_goal"):
            self.goal_state = (-1, -1)

        elif in_message.startswith("set dim"):
            dims = in_message.split(":")[1].split(",")
            max_row, max_col = int(dims[0]), int(dims[1])
            states_rc = [(r, c) for r in range(max_row)
                         for c in range(max_col)]
            self.max_row, self.max_col = max_row, max_col
            self.states_rc = states_rc

        elif in_message.startswith("set eigen_purpose"):
            # TODO: pickle fails for eigenvectors which has 1e-10 format
            # self.reward_vector = pickle.loads(in_message.split(":")[1])
            # handling manually
            self.reward_vector = pickle.loads(in_message[18:])

        elif in_message.startswith("get max_row"):
            return self.max_row

        elif in_message.startswith("get max_col"):
            return self.max_col

        elif in_message.startswith("get default_max_actions"):
            return self.default_max_actions
        elif in_message.startswith("get max_actions"):
            return self.max_actions
        else:
            print("Invalid env message: " + in_message)
            exit()
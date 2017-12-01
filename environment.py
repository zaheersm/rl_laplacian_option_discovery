import numpy as np
import pickle


# Possible default actions in tabular environment
default_action_set = [(0, 1), (0, -1), (1, 0), (-1, 0)] # R, L, D, U

TERMINATE_ACTION = (0,0)


class GridEnvironment(object):

    def __init__(self, max_row=10, max_col=10, goal_state=(0,0),
                 obstacle_vector = None, reward_vector = None):

        states_rc = [(r, c) for r in range(max_row) for c in range(max_col)]
        self.states_rc = states_rc # all possible states (r,c)
        self.max_row, self.max_col = max_row, max_col

        # use exploring starts if start_state is None
        self.start_state = None 
        self.goal_state = goal_state

        self.action_set = default_action_set[:]
        self.default_max_actions = len(self.action_set) # will stay fixed
        self.max_actions = len(self.action_set) # can increase

        if reward_vector is None:
            # Reward is 0.0 everywhere, and 1.0 in goal state
            self.reward_vector = np.zeros((len(self.states_rc))) * 0.0
            try: # It is possible that there's no goal state e.g. (-1,-1)
                goal_idx = self.states_rc.index(goal_state)
                self.reward_vector[goal_idx] = 1.0
            except ValueError:
                pass
        else:
            self.reward_vector = reward_vector

        self.current_state = None

    def start(self):

        # exploring starts
        if self.start_state == None:
            start_state_int = np.random.randint(len(self.states_rc))
        # start state is specified
        else:
            start_state_int = self.start_state
        self.current_state = np.asarray([start_state_int])

        # Returning a copy of the current state
        return np.copy(self.current_state)

    def step(self, action):
        if not action < self.max_actions:
            print "Invalid action taken!!"
            print "action: ", action
            print "current_state", self.current_state

        action = self.action_set[action]

        # if terminate action
        if action == TERMINATE_ACTION:
            self.current_state = None
            result = {"reward": 0, "state": None, "isTerminal": True}
            return result

        else:
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

        # check terminal
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
        self.current_state = None
        return

    # Getter and Setter functions
    def set_start_state(self, start_state):
        self.start_state = start_state

    def set_goal_state(self, goal_state):
        self.goal_state = goal_state

    def add_terminate_action(self):
        self.action_set.append(TERMINATE_ACTION)
        self.max_actions = len(self.action_set)

    def get_grid_dimension(self):
        return self.max_row, self.max_col

    def get_default_max_actions(self):
        return self.default_max_actions

    def set_current_state(self, current_state):
        self.current_state = np.asarray([current_state])

    def set_eigen_purpose(self, eigen_purpose):
        self.reward_vector = eigen_purpose

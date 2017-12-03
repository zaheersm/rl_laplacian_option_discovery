import numpy as np
import pickle


# Possible default actions in tabular environment
default_action_set = [(0, 1), (0, -1), (1, 0), (-1, 0)] # R, L, D, U

TERMINATE_ACTION = (0,0)


def initialize_env(env_type):

    if env_type == "Grid":
        return GridEnvironment()
    elif env_type == "4-Room":
        return RoomEnvironment()
    elif env_type == "I-Maze":
        return I_MazeEnvironment()
    else:
        print("Invalid Environment: " + env_type)
        exit()

# returns relevant environment information
def parse_env(env_grid):
    max_row = len(env_grid)
    max_col = len(env_grid[0])
    obstacles = []
    start_state = (-1,-1)
    goal_state = (-1,-1)
    for r in range(max_row):
        for c in range(max_col):
            if env_grid[r][c] == 'X':
                obstacles.append((r,c))
            elif env_grid[r][c] == 'S':
                start_state = (r,c)
            elif env_grid[r][c] == 'G':
                goal_state = (r,c)

    return max_row, max_col, start_state, goal_state, obstacles

class BaseEnvironment(object):

    def __init__(self, max_row, max_col, start_state,
                 goal_state, obstacle_vector = [], reward_vector = None):
        states_rc = [(r, c) for r in range(max_row) for c in range(max_col)]
        self.states_rc = states_rc # all possible states (r,c)
        self.max_row, self.max_col = max_row, max_col

        # use exploring starts if start_state is None
        self.start_state = start_state 
        self.goal_state = goal_state

        self.action_set = default_action_set[:]
        self.default_max_actions = len(self.action_set) # will stay fixed
        self.max_actions = len(self.action_set) # can increase

        self.obstacle_vector = obstacle_vector

        if reward_vector is None:
            # Reward is 0.0 everywhere, and 1.0 in goal state
            self.reward_vector = np.zeros((len(self.states_rc))) * 0.0
            try: # It is possible that there's no goal state e.g. (-1,-1)
                goal_idx = self.states_rc.index(goal_state)
                self.reward_vector[goal_idx] = 1.0
            except ValueError:
                pass

            for obs_state in obstacle_vector:
                obs_idx = self.states_rc.index(obs_state)
                self.reward_vector[obs_idx] = float('-inf')

        else:
            self.reward_vector = reward_vector

        self.current_state = None

    def start(self):

        # exploring starts
        if self.start_state == (-1,-1):
            valid_state_idx = [idx for idx, state in enumerate(self.states_rc) if state not in self.obstacle_vector]
            start_state_int = np.random.choice(valid_state_idx)
            #start_state_int = np.random.randint(len(self.states_rc))
        # start state is specified
        else:
            start_state_int = self.states_rc.index(self.start_state)
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


            # if s or ns is an obstacle, don't move
            if s in self.obstacle_vector or ns in self.obstacle_vector:

                # Going back to the integer representation
                s = self.states_rc.index(s)
                ns = self.states_rc.index(s) # same state
                reward = 0 #- 0.001 # small step penalty
            else:

                # Going back to the integer representation
                s = self.states_rc.index(s)
                ns = self.states_rc.index(ns) # next state
                reward = self.reward_vector[ns] - self.reward_vector[s] #- 0.001 # small step penalty

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


class GridEnvironment(BaseEnvironment):
    def __init__(self):
        grid_env = [
            '         G',
            '          ',
            '          ',
            '          ',
            '          ',
            '          ',
            '          ',
            '          ',
            '          ',
            'S         ',
        ]
        max_row, max_col, start_state, goal_state, obstacles = parse_env(grid_env)

        BaseEnvironment.__init__(self, max_row, max_col, start_state,
                 goal_state, obstacles)


class RoomEnvironment(BaseEnvironment):
    def __init__(self):
        room_env = [
            '     X    G',
            '     X     ',
            '           ',
            '     X     ',
            '     X     ',
            'X XXXX     ',
            '     XXX XX',
            '     X     ',
            '     X     ',
            '           ',
            'S    X     '
        ]

        max_row, max_col, start_state, goal_state, obstacles = parse_env(room_env)

        BaseEnvironment.__init__(self, max_row, max_col, start_state,
                 goal_state, obstacles)



class I_MazeEnvironment(BaseEnvironment):
    def __init__(self):
        I_maze_env = [
            ' XXXXXXXXXXXXXG',
            '               ',
            'SXXXXXXXXXXXXX '
        ]

        max_row, max_col, start_state, goal_state, obstacles = parse_env(I_maze_env)

        BaseEnvironment.__init__(self, max_row, max_col, start_state,
                 goal_state, obstacles)



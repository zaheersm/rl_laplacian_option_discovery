import numpy as np
import pickle

# Possible default actions in tabular environment
default_action_set = [(0, 1), (0, -1), (1, 0), (-1, 0)] # R, L, D, U

TERMINATE_ACTION = (0,0)

class QAgent(object):

    def __init__(self, max_row, max_col):
        """
        Q[row][col][a] would represent the action value for
        state [row, col] and the action 'action_set[a]'

        That is, an action has an integer representation which
        can be converted into a tuple representation by indexing
        action_set using the integer
        """
        self.action_set = default_action_set[:]
        self.default_max_actions = len(self.action_set) # will stay fixed
        self.max_actions = len(self.action_set) # can increase

        self.Q = np.zeros((max_row, max_col, self.max_actions))
        self.states_rc = [(r, c) for r in range(max_row)
                          for c in range(max_col)]

        self.last_state, self.last_action = -1, -1
        self.steps = 0
        self.max_row, self.max_col = max_row, max_col


    def start(self, state):

        # Saving the state as last_state
        self.last_state = state[0]
        # Getting the cartesian form of the states
        row, col = self.states_rc[state[0]]

        if np.random.uniform() < self.epsilon:
            ca = np.random.randint(self.max_actions)
        else:
            q = self.Q[row][col]
            # Breaking ties randomly
            ca = np.random.choice(np.flatnonzero(q == q.max()))

        action = self.action_set[ca]
        self.last_action = ca

        # Updating steps
        self.steps = 1

        # Returning integer representation of the action
        return self.last_action

    def step(self, reward, state):
        """
        Arguments: reward: floting point, state: integer
        Returns: action: list with two integers [row, col]
        """
        current_state = self.states_rc[state[0]]
        # crow: row of current state
        crow = current_state[0]
        # ccol: col of current state
        ccol = current_state[1]
        if np.random.uniform() < self.epsilon:
            ca = np.random.randint(self.max_actions)
        else:
            q = self.Q[crow][ccol]
            # Breaking ties randomly
            ca = np.random.choice(np.flatnonzero(q == q.max()))

        action = self.action_set[ca]
        # Getting the coordinates of the last state
        lrow, lcol = self.states_rc[self.last_state]
        # la: integer representation of last action
        la = self.last_action

        # target
        target = reward + (self.discount)*(self.Q[crow][ccol].max())
        # Update: New Estimate = Old Estimate + StepSize[Target - Old Estimate]
        self.Q[lrow][lcol][la] += self.alpha*(target - self.Q[lrow][lcol][la])

        # Updating last_state and last_action
        self.last_state = self.states_rc.index(current_state)
        self.last_action = ca

        # Updating steps
        self.steps += 1

        return self.last_action

    def end(self, reward):
        """
        Arguments: reward: floating point
        Returns: Nothing
        """
        lrow, lcol = self.states_rc[self.last_state]
        la = self.last_action
        # We know that the agent has transitioned in the terminal state
        # for which all action values are 0
        target = reward + 0
        self.Q[lrow][lcol][la] += self.alpha*(target - self.Q[lrow][lcol][la])
        # Resetting last_state and last_action for the next episode
        self.last_state, self.last_action = -1, -1
        return

    def cleanup(self):
        # clean up

        self.Q = np.zeros((self.max_row, self.max_col, self.max_actions))
        self.last_state, self.last_action = -1, -1
        self.steps = 0

        return


    def message(self, in_message):
        """
        Arguments: in_message: string
        returns: The value function as a string.
        This function is complete. You do not need to add code here.
        """
        if (in_message == 'ValueFunction'):
            return pickle.dumps(np.max(self.Q, axis=1), protocol=0)

        elif in_message.startswith("set eigen_option"):
            eigenoption = pickle.loads(in_message.split(":")[1])

            self.max_actions += 1
            self.Q = np.zeros((self.max_row, self.max_col, self.max_actions))

            # TODO: add to option set
            

        elif in_message.startswith("set terminate_action"):
            self.action_set.append(TERMINATE_ACTION)
            self.max_actions = len(self.action_set)
            self.Q = np.zeros((self.max_row, self.max_col, self.max_actions))

        elif in_message == ("get steps"):
            return str(self.steps)

        elif in_message.startswith("set alpha"):
            self.alpha = float(in_message.split(":")[1])

        elif in_message.startswith("set epsilon"):
            self.epsilon = float(in_message.split(":")[1])

        elif in_message.startswith("set discount"):
            self.discount = float(in_message.split(":")[1])

        elif in_message.startswith("set dim"):
            dims = in_message.split(":")[1].split(",")
            max_rows, max_cols = int(dims[0]), int(dims[1])
            self.__init__(max_rows, max_cols)

        elif in_message.startswith("get policy"):
            pi = self._policy()
            return pickle.dumps(pi, protocol=0)

        else:
            print("Invalid agent message: " + in_message)
            exit()

    def _policy(self):
        pi = np.zeros((len(self.states_rc,)), dtype=np.int)

        for idx, state in enumerate(self.states_rc):
            row, col = state[0], state[1]
            q = self.Q[row][col]
            # Taking last max to break ties inorder to prefer Terminate action
            ca = np.flatnonzero(q == q.max())[-1]
            pi[idx] = ca # each state will have related optimal action idx

        return pi


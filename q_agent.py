#!/usr/bin/env python

import numpy as np
import pickle
from sys import exit
import matplotlib.pyplot as plt


class Agent(object):
    def __init__(self, total_rows=7, total_cols=10):
        """
        Hint: Initialize the variables that need to be reset before each run begins
        Returns: nothing
        """
        """
        Initializing a 7 x 10 x 4 for Q
        Q[row][col][a] would represent the action value for
        state [row, col] and the action 'action_set[a]'

        That is, an action has an integer representation which
        can be converted into a list representation by indexing
        action_set using the integer
        """
        self.action_set = [(0,1), (0,-1), (1,0), (-1,0), (-1, -1)]
        self.MAX_ACTIONS = len(self.action_set)

        self.Q = np.zeros((total_rows, total_cols, self.MAX_ACTIONS))
        states_rc = []
        for r in range(total_rows):
            for c in range(total_cols):
                states_rc.append((r,c))

        self.states_rc = states_rc
        # Remember internally stored state and action is in cartesian form
        self.last_state = np.zeros((2,), dtype=np.int)
        self.last_action = (-1, -1)
        self.steps = 0
        self.total_rows, self.total_cols = total_rows, total_cols
        self.epsilon = 0.1
        self.alpha = 0.1
        self.discount = 1.0

    def start(self, state):
        """
        Arguments: state: numpy array
        Returns: action: integer
        """
        # Getting the cartesian form of the states
        state = self.states_rc[state[0]]
        self.last_state[0], self.last_state[1] = state[0], state[1]
        row = self.last_state[0]
        col = self.last_state[1]

        if np.random.uniform() < self.epsilon:
            action = self.action_set[np.random.randint(self.MAX_ACTIONS)]
        else:
            q = self.Q[row][col]
            # Breaking ties randomly
            ca = np.random.choice(np.flatnonzero(q == q.max()))
            action = self.action_set[ca]

        self.last_action = (action[0], action[1])
        # Updating steps
        self.steps += 1
        # Returning integer representation of the action
        return self.action_set.index(action)

    def step(self, reward, state): # returns NumPy array, reward: floating point, this_observation: NumPy array
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
            # ca: current action's integer representation
            ca = np.random.randint(self.MAX_ACTIONS)
            # List representation of the action
            action = self.action_set[ca]
        else:
            q = self.Q[crow][ccol]
            # Breaking ties randomly
            ca = np.random.choice(np.flatnonzero(q == q.max()))
            action = self.action_set[ca]
        
        # lrow, lcol: row and co of last state
        lrow, lcol = self.last_state[0], self.last_state[1]
        # la: integer representation of last action
        la = self.action_set.index(self.last_action)

        # target
        target = reward + (self.discount)*(self.Q[crow][ccol].max())
        # Update: New Estimate = Old Estimate + StepSize[Target - Old Estimate]
        self.Q[lrow][lcol][la] = self.Q[lrow][lcol][la] + self.alpha*(target - self.Q[lrow][lcol][la])

        # Updating last_state and last_action
        self.last_state[0], self.last_state[1] = current_state[0], current_state[1]
        self.last_action = (action[0], action[1])

        # Updating steps
        self.steps+=1
        
        return self.action_set.index(self.last_action)

    def end(self, reward):
        """
        Arguments: reward: floating point
        Returns: Nothing
        """
        lrow, lcol = self.last_state[0], self.last_state[1]
        la = self.action_set.index(self.last_action)
        # We know that the agent has transitioned in the terminal state 
        # for which all action values are 0
        target = reward + 0
        self.Q[lrow][lcol][la] = self.Q[lrow][lcol][la] + self.alpha*(target - self.Q[lrow][lcol][la])
        # Resetting last_state and last_action for the next episode
        self.last_state = np.zeros((2,), dtype=np.int)
        self.last_action = [-1, -1]
        return

    def cleanup(self):
        """
        This function is not used
        """
        # clean up
        return

    def message(self, in_message): # returns string, in_message: string
        """
        Arguments: in_message: string
        returns: The value function as a string.
        This function is complete. You do not need to add code here.
        """
        # should not need to modify this function. Modify at your own risk
        if (in_message == 'ValueFunction'):
            return pickle.dumps(np.max(self.Q, axis=1), protocol=0)
        elif in_message == ("steps"):
            return str(self.steps)
        elif in_message.startswith("alpha"):
            self.alpha = float(in_message.split(",")[1])
        elif in_message.startswith("dim"):
            dims = in_message.split(":")[1].split(",")
            total_rows, total_cols = int(dims[0]), int(dims[1])
            self.__init__(total_rows, total_cols)
        elif in_message.startswith("policy"):
            pi = self._policy()
            return pickle.dumps(pi, protocol=0)
        else:
            return "I don't know what to return!!"

    def _policy(self):
        
        pi = np.zeros((len(self.states_rc,)), dtype=np.int)
        for idx, state in enumerate(self.states_rc):
            row, col = state[0], state[1]
            q = self.Q[row][col]
            ca = np.flatnonzero(q == q.max())[-1] # take last max to break ties
            pi[idx] = ca

        return pi

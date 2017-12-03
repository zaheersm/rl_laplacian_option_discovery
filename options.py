import numpy as np
import pickle
import rlglue
import environment
import agents
import plot_utils


class Options(object):

    def __init__(self, alpha=0.1, epsilon=1.0, discount=0.1):

        # Configuring Environment
        self.env = environment.GridEnvironment()

        self.env.set_goal_state((-1, -1)) # (-1, -1) implies no goal_state
        self.env.set_start_state((-1, -1)) # no start_state
        self.env.add_terminate_action()
        self.max_row, self.max_col = self.env.get_grid_dimension()

        # Configuring Agent
        self.agent = agents.QAgent(self.max_row, self.max_col)
        self.agent.add_terminate_action()
        self.agent.set_alpha(alpha)
        self.agent.set_epsilon(epsilon)
        self.agent.set_discount(discount)

        self.glue = rlglue.RLGlue(self.env, self.agent)

        self.eigenvectors = None
        self.eigenoptions = []

        # compute eigen
        self.compute_eigen()

    def compute_eigen(self):

        # Need to exclude Terminate Action
        default_max_actions = self.env.get_default_max_actions()

        # get all possible (r,c) states in env
        states_rc = []
        for r in range(self.max_row):
            for c in range(self.max_col):
                states_rc.append((r, c))

        total_states = len(states_rc)

        # Compute adjacency matrix (take all possible actions from every state)
        adjacency = np.zeros((total_states, total_states), dtype = np.int)
        for state in range(total_states):
            for a in range(default_max_actions):
                # Take a specified action from a given start state
                self.env.set_current_state(state)
                result = self.glue.environment.step(a)
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

        # extract eigenvalues(w), eigenvectors(v)
        w, v = np.linalg.eig(L)
        v = v.T # switch axes to correspond to eigenvalue index

        # sort in order of increasing eigenvalue
        # self.eigenoptions will be computed lazily
        indexes = range(len(w))
        indexes.sort(key=w.__getitem__)
        eigenvalues = np.asarray(map(w.__getitem__, indexes))
        self.eigenvectors = np.asarray(map(v.__getitem__, indexes))
        # Adding eigenvectors in the opposite directions
        shape = self.eigenvectors.shape
        shape = (shape[0] * 2, shape[1])
        eigenvectors = np.zeros(shape)
        for idx in range(len(self.eigenvectors)):
            v1 = self.eigenvectors[idx] * 1
            # v2 is the opposite eigenvector
            v2 = self.eigenvectors[idx] * -1
            eigenvectors[idx*2] = v1
            eigenvectors[idx*2 + 1] = v2

        self.eigenvectors = eigenvectors

    def learn_next_eigenoption(self, steps=100000):

        # learn next option
        new_option_idx = len(self.eigenoptions)
        if new_option_idx == len(self.eigenvectors):
            print("All eigenoptions have already been computed")
            return

        # set reward vector
        self.env.set_eigen_purpose(self.eigenvectors[new_option_idx])
        #plot_utils.print_eigen(self.eigenvectors[new_option_idx])

        # Learn policy
        while steps >= 0:
            is_terminal = self.glue.episode(steps)
            if is_terminal is True:
                ep_steps = self.agent.get_steps()
            else:
                break
            steps -= ep_steps

        eigenoption = self.agent.get_policy()
        self.eigenoptions.append(eigenoption)
        self.glue.cleanup() # reset Q(S,A) and reward vector

        # return newly learned policy
        return self.eigenoptions[-1]

    def get_eigenoptions(self):        
        return self.eigenoptions

    # display eigenoption at the idx
    def display_eigenoption(self, display = True, savename='', idx = -1):
        # default return latest learned eigenoption
        if len(self.eigenoptions) < 1 or idx not in range(-1, len(self.eigenoptions)):
            print "The eigenoption has not been learnt for this option yet"
            return

        plot_utils.plot_pi(self.eigenoptions[idx], self.max_row,
                           self.max_col, display, savename)

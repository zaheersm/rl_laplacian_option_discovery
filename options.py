import numpy as np
import pickle
import RLGlue
import Environment
import Agents
import plot_utils


class Options(object):

    def __init__(self, internal_env, internal_agent, alpha=0.1, epsilon=0.1, discount=0.1):

        # currently internal_env/agent is just a copy of external_env/agent
        self.glue = RLGlue.RLGlue(internal_env, internal_agent)

        # set "no goal"
        self.glue.env_message("set no_goal")

        # add 'terminate' action
        self.glue.env_message("set terminate_action")
        self.glue.agent_message("set terminate_action")

        # set alpha
        command = "set alpha:{}".format(alpha)
        self.glue.agent_message(command)

        # set epsilon
        command = "set epsilon:{}".format(epsilon)
        self.glue.agent_message(command)

        # set discount
        command = "set discount:{}".format(discount)
        self.glue.agent_message(command)

        self.eigenvalues = None
        self.eigenvectors = None
        self.eigenoptions = []

        # compute eigen
        self.compute_eigen()


    def compute_eigen(self):

        # TODO: Later need to handle states with obstacles (exclude them)
        # Send env_message to get all accessible states
        self.max_row = self.glue.env_message("get max_row")
        self.max_col = self.glue.env_message("get max_col")

        # Need to exclude Terminate Action
        default_max_actions = self.glue.env_message("get default_max_actions") # 4

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

                # Take a specified action from a given start state to get next state
                command = "set start_state:{}".format(state)
                self.glue.env_message(command)
                
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

        # dunno why zip doesn't work for 8x8 grid and larger
        # self.eigenvalues, self.eigenvectors = zip(*sorted(zip(w,v)))

        # alternative sort method
        indexes = range(len(w))
        indexes.sort(key=w.__getitem__)
        self.eigenvalues = np.asarray(map(w.__getitem__, indexes))
        self.eigenvectors = np.asarray(map(v.__getitem__, indexes))

        # DEBUG: check the most interesting eigen vector (smallest eigenvalue)
        # print(self.eigenvalues.shape)
        # print(self.eigenvectors.shape)
        # idx = np.argmin(self.eigenvalues)

        # print idx
        # print self.eigenvectors[0,:]
        # print self.eigenvectors[1,:] # pickle fails
        # print self.eigenvectors[2,:] # pickle fails
        # print self.eigenvectors[3,:]



    def learn_eigenoption(self, steps=100000):

        # learn next option
        new_option_idx = len(self.eigenoptions)
        if new_option_idx == len(self.eigenvalues):
            print("All eigenoptions have already been computed")
            return

        # set reward vector
        command = "set eigen_purpose:" + pickle.dumps(self.eigenvectors[new_option_idx], protocol=0)
        #print(command[:18])

        self.glue.env_message(command)

        # Learn policy
        while steps >= 0:
            is_terminal = self.glue.episode(steps)
            if is_terminal is True:
                ep_steps = int(self.glue.agent_message("get steps"))
            else:
                break
            steps -= ep_steps
        self.glue.cleanup() # Currently does nothing: should reset Q(S,A) and reward vector
        self.eigenoptions.append(pickle.loads(self.glue.agent_message("get policy")))

        # return newly learned policy
        return self.eigenoptions[-1]

    def get_eigenoptions(self):        
        return self.eigenoptions

    def display_eigenoption(self, idx = -1):
        # default return latest learned eigenoption
        if len(self.eigenoptions) < 1 or idx not in range(-1, len(self.eigenoptions)):
            print "The eigenoption has not been learnt for this option yet"
            return
        plot_utils.plot_pi(self.eigenoptions[idx], self.max_row, self.max_col)


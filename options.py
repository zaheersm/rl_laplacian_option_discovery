import numpy as np
import pickle
import RLGlue
import Environment
import Agents
import plot_utils


class Option(object):

    def __init__(self, eigen_purpose, max_row, max_col, alpha=0.1,
                 epsilon=0.1, discount=0.1):
        self.glue = RLGlue.RLGlue("environment", "q_agent")
        # Setting max_row and max_col for agent and environment
        command = "dim:{},{}".format(max_row, max_col)
        self.glue.env_message(command)
        self.glue.agent_message(command)

        # Setting the eigen purpose
        command = "eigen_purpose:" + pickle.dumps(eigen_purpose, protocol=0)
        self.glue.env_message(command)
        command = "alpha:{}".format(alpha)
        self.glue.agent_message(command)
        command = "epsilon:{}".format(epsilon)
        self.glue.agent_message(command)
        command = "discount:{}".format(discount)
        self.glue.agent_message(command)
        self.max_row = max_row
        self.max_col = max_col
        self.pi = None

    def learn(self, steps=100000):
        # Returns the policy
        while steps >= 0:
            is_term = self.glue.episode(steps)
            if is_term is True:
                ep_steps = int(self.glue.agent_message("steps"))
            else:
                break
            steps -= ep_steps
        self.glue.cleanup()
        self.pi = pickle.loads(self.glue.agent_message("policy"))
        return self.pi

    def get_policy(self, steps=100000):
        if self.pi is None:
            self.pi = self.learn(steps)
        return self.pi

    def display_policy(self):
        if self.pi is None:
            print "The policy has not been learnt for this option yet"
            return
        plot_utils.plot_pi(self.pi, self.max_row, self.max_col)


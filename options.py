import numpy as np
import pickle

import plot_utils


class Option(object):

    def __init__(self, eigen_purpose, max_row, max_col, alpha=0.1,
                 epsilon=0.1, discount=0.1):
        self.glue = RLGlue("environment", "q_agent")
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


class RLGlue(object):

    def __init__(self, env_name, agent_name):
        """
        Arguments
        ---------
        env_name : string
            filename of the environment module
        agent_name : string
            filename of the agent module
        """
        self.environment = Environment()
        self.agent = QAgent()
        self.total_reward = 0.0
        self.num_steps = 0
        self.num_episodes = 0

    def start(self):
        """
        Returns
        -------
        observation : dict
            dictionary containing what the first state and action were
        """

        self.total_reward = 0.0
        self.num_steps = 1

        last_state = self.environment.start()
        last_action = self.agent.start(last_state)
        observation = {"state": last_state, "action": last_action}

        self.last_action = last_action

        return observation

    def step(self):
        """
        Returns
        -------
        result : dict
            dictionary with keys {reward,state,action,isTerminal}
        """
        result = self.environment.step(self.last_action)
        self.total_reward += result['reward']
        if result['isTerminal'] is True:
            self.num_episodes += 1
            self.agent.end(result['reward'])
            result['action'] = None
        else:
            self.num_steps += 1
            self.last_action = self.agent.step(result['reward'],
                                               result['state'])
            result['action'] = self.last_action
        return result

    def cleanup(self):

        self.environment.cleanup()
        self.agent.cleanup()

    def agent_message(self, message):
        """
        Arguments
        ---------
        message : string
            the message to send to the agent

        Returns
        -------
        the_agent_response : string
            the agent's response to the message
        """
        if message is None:
            message_to_send = ""
        else:
            message_to_send = message

        the_agent_response = self.agent.message(message_to_send)
        if the_agent_response is None:
            return ""

        return the_agent_response

    def env_message(self, message):
        """
        Arguments
        ---------
        message : string
            the message to send to the environment
        Returns
        -------
        the_env_response : string
            the environment's response to the message
        """
        if message is None:
            message_to_send = ""
        else:
            message_to_send = message

        the_env_response = self.environment.message(message_to_send)
        if the_env_response is None:
            return ""

        return the_env_response

    def episode(self, max_steps_this_episode):
        """
        Arguments
        ---------
        max_steps_this_episode : int

        Returns
        -------
        is_terminal : bool
        """
        is_terminal = False

        self.start()
        while (not is_terminal) and \
                ((max_steps_this_episode == 0) or
                 (self.num_steps < max_steps_this_episode)):
            rl_step_result = self.step()
            is_terminal = rl_step_result['isTerminal']

        return is_terminal


class QAgent(object):

    def __init__(self, max_rows=10, max_cols=10):
        """
        Q[row][col][a] would represent the action value for
        state [row, col] and the action 'action_set[a]'

        That is, an action has an integer representation which
        can be converted into a tuple representation by indexing
        action_set using the integer
        """
        # action_set = [right, left, down, up, terminate]
        self.action_set = [(0, 1), (0, -1), (1, 0), (-1, 0), (-1, -1)]
        self.MAX_ACTIONS = len(self.action_set)

        self.Q = np.zeros((max_rows, max_cols, self.MAX_ACTIONS))
        self.states_rc = [(r, c) for r in range(max_rows)
                          for c in range(max_cols)]

        self.last_state, self.last_action = -1, -1
        self.steps = 0
        self.max_rows, self.max_cols = max_rows, max_cols
        self.epsilon, self.alpha, self.discount = 0.1, 0.1, 1.0
        self.max_rows, self.max_cols = max_rows, max_cols

    def start(self, state):

        # Saving the state as last_state
        self.last_state = state[0]
        # Getting the cartesian form of the states
        row, col = self.states_rc[state[0]]

        if np.random.uniform() < self.epsilon:
            ca = np.random.randint(self.MAX_ACTIONS)
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
            ca = np.random.randint(self.MAX_ACTIONS)
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
        """
        This function is not used
        """
        # clean up
        return

    def message(self, in_message):
        """
        Arguments: in_message: string
        returns: The value function as a string.
        This function is complete. You do not need to add code here.
        """
        if (in_message == 'ValueFunction'):
            return pickle.dumps(np.max(self.Q, axis=1), protocol=0)
        elif in_message == ("steps"):
            return str(self.steps)
        elif in_message.startswith("alpha"):
            self.alpha = float(in_message.split(":")[1])
        elif in_message.startswith("epsilon"):
            self.epsilon = float(in_message.split(":")[1])
        elif in_message.startswith("discount"):
            self.discount = float(in_message.split(":")[1])
        elif in_message.startswith("dim"):
            dims = in_message.split(":")[1].split(",")
            max_rows, max_cols = int(dims[0]), int(dims[1])
            self.__init__(max_rows, max_cols)
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
            # Taking last max to break ties inorder to prefer Terminate action
            ca = np.flatnonzero(q == q.max())[-1]
            pi[idx] = ca

        return pi


class Environment(object):

    def __init__(self, reward_vector=None, max_row=3, max_col=3,
                 goal_state=(0, 0)):

        states_rc = [(r, c) for r in range(max_row) for c in range(max_col)]
        self.states_rc = states_rc

        self.max_row, self.max_col = max_row, max_col

        self.goal_state = goal_state

        # Need to figure out a termination technique
        self.action_set = [(0, 1), (0, -1), (1, 0), (-1, 0), (-1, -1)]
        self.MAX_ACTION = len(self.action_set)

        if reward_vector is None:
            # Reward is -1.0 everywhere
            self.reward_vector = np.zeros((len(self.states_rc))) * -1
        else:
            self.reward_vector = reward_vector

    def start(self):
        start_state = np.random.randint(len(self.states_rc))
        self.current_state = np.asarray([start_state])
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
        ns = self.states_rc.index(ns)
        reward = self.reward_vector[ns] - self.reward_vector[s]
        self.current_state[0] = ns
        result = {"reward": reward, "state": self.current_state,
                  "isTerminal": False}

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
            states_rc = [(r, c) for r in range(max_row)
                         for c in range(max_col)]
            self.max_row, self.max_col = max_row, max_col
            self.states_rc = states_rc
        elif in_message.startswith("eigen_purpose"):
            self.reward_vector = pickle.loads(in_message.split(":")[1])
        return ""

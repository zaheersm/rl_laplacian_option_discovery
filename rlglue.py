import numpy as np

class RLGlue(object):

    def __init__(self, env_instance, agent_instance):
        """
        Arguments
        ---------
        env_name : string
            filename of the environment module
        agent_name : string
            filename of the agent module
        """
        self.environment = env_instance
        self.agent = agent_instance
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

    def cleanup(self):

        self.environment.cleanup()
        self.agent.cleanup()

        self.total_reward = 0.0
        self.num_steps = 0
        self.num_episodes = 0

    def get_num_steps(self):
        return self.num_steps

    def get_num_episodes(self):
        return self.num_episodes

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



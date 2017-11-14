#!/usr/bin/env python
from importlib import import_module

class RLGlue(object):

    def __init__ (self, env_name, agent_name):
        """
        Arguments
        ---------
        env_name : string
            filename of the environment module
        agent_name : string
            filename of the agent module
        """
        env_module = import_module(env_name)
        agent_module = import_module(agent_name)
        self.environment = env_module.Environment()
        self.agent = agent_module.Agent()
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
	
	self.total_reward = 0.0;
	self.num_steps = 1;

	last_state = self.environment.start()
        last_action = self.agent.start(last_state)
	observation = {"state":last_state, "action":last_action}

	self.last_action = last_action

	return observation

    def agent_start(self, state):
	"""
	Arguments
	---------
	state : numpy array
	    the initial state the agent is starting in

	Returns
	-------
	int : the action taken by the agent
	"""
	return self.agent.start(state)

    def agent_step(self, reward, state):
	"""
	Arguments
	---------
	observation : dict
	    a dictionary containing the reward and next state resulting from
	    the agent's most-recent action

	Returns
	-------
	int : the action taken by the agent
	"""
	return self.agent.step(reward,state)

    def agent_end(self, reward):
	"""
	Arguments
	---------
	reward : float
	    the final reward received by the agent
	"""
	self.agent.end(reward)

    def env_start(self):
	"""
	Returns
	-------
	numpy array : the initial state
	"""
	self.total_reward = 0.0
	self.num_steps = 1

	return self.environment.start()

    def env_step(self, action): # returns (floating point, NumPy array, Boolean), action: NumPy array
	"""
	Arguments
	---------
	action : int
	    the most recent action taken by the agent

	Returns
	-------
	result : dict
	    dictionary with keys {reward,state,isTerminal}
	"""
        result = self.environment.step(action)

	self.total_reward += result['reward']

	if result['isTerminal'] == True:
	    self.num_episodes += 1
	else:
	    self.num_steps += 1

	return result
    
    def step(self):
	"""
	Returns
	-------
	result : dict
	    dictionary with keys {reward,state,action,isTerminal}
	"""
        result = self.environment.step(self.last_action)
	self.total_reward += result['reward'];
	if result['isTerminal'] == True:
	    self.num_episodes += 1
	    self.agent.end(result['reward'])
	    result['action'] = None
	else:
	    self.num_steps += 1
	    self.last_action = self.agent.step(result['reward'],result['state'])
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
	while (not is_terminal) and ((max_steps_this_episode == 0) or (self.num_steps < max_steps_this_episode)):
	    rl_step_result = self.step()
            is_terminal = rl_step_result['isTerminal']

	    # if (num_steps == (max_steps_this_episode)):
		# print 'not ended'

	return is_terminal

    def total_return():
	""" returns floating point """
	return self.total_reward

    def num_steps():
	""" returns integer """
	return self.num_steps

    def num_episodes():
	""" returns integer """
	return self.num_episodes

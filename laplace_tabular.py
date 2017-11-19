import sys
import numpy as np
import pickle
import copy

import RLGlue
import Environment
import Agents
import options

np.set_printoptions(precision=2)

# Experiment parameter
num_episodes = 100
alpha = 0.1
epsilon = 0.1
discount = 1.0

# TODO: Add random seed for random start state / random action selection
# state_random_seed_arr = []
# for i in range(num_episodes):
# 	random_seed_arr.append(np.random.randint(1000))



# Set up environment
max_row = 10
max_col = 10
obstacles = [] # TODO: handle obstacles
goal_state = (0,9)

# Specify name of env and agent
env_name = "GridEnvironment"
agent_name = "QAgent" 

external_env = getattr(Environment, env_name)(max_row, max_col, goal_state)
external_agent = getattr(Agents, agent_name)(max_row, max_col)

internal_env = copy.copy(external_env)
internal_agent = copy.copy(external_agent)


# Baseline Agent
glue = RLGlue.RLGlue(external_env, external_agent)

# set alpha
command = "set alpha:{}".format(alpha)
glue.agent_message(command)

# set epsilon
command = "set epsilon:{}".format(epsilon)
glue.agent_message(command)

# set discount
command = "set discount:{}".format(discount)
glue.agent_message(command)

ave_steps = 0
for ep in range(num_episodes):
	# run episode 
	glue.episode(0)
	ave_steps += glue.get_num_steps() 
ave_steps/=float(num_episodes)
print("Base agent: " + str(ave_steps) + " steps")
glue.cleanup()


exit()

# Compute eigenoptions
opt = options.Options(internal_env, internal_agent, alpha=alpha, epsilon=epsilon, discount=discount)


eigenoption = opt.learn_eigenoption(100000)

# display newest learned option
#opt.display_eigenoption()

# TODO: add options incrementally to actions set
#glue.agent_message("set eigen_option:" + pickle.dumps(self.eigenvectors[new_option_idx], protocol=0))







import sys
import numpy as np
import pickle
import copy

import RLGlue
import Environment
import Agents
import options

np.set_printoptions(precision=2)



# Set up environment
max_row = 10
max_col = 10
obstacles = [] # TODO: handle obstacles

# Specify name of env and agent
env_name = "GridEnvironment"
agent_name = "QAgent" 

external_env = getattr(Environment, env_name)(max_row, max_col, (0,9))
external_agent = getattr(Agents, agent_name)(max_row, max_col)

internal_env = copy.copy(external_env)
internal_agent = copy.copy(external_agent)


opt = options.Options(internal_env, internal_agent, alpha=0.1, epsilon=0.1, discount=1.0)


###### DEBUG: learn one option
opt.learn_eigenoption(100000)

# display newest learned option
opt.display_eigenoption()


opt.learn_eigenoption(100000)

# display newest learned option
opt.display_eigenoption()

opt.learn_eigenoption(100000)

# display newest learned option
opt.display_eigenoption()

opt.learn_eigenoption(100000)

# display newest learned option
opt.display_eigenoption()

exit()
##############################

# glue = RLGlue.RLGlue(external_env, external_agent)

# # set alpha
# command = "set alpha:{}".format(alpha)
# glue.agent_message(command)

# # set epsilon
# command = "set epsilon:{}".format(epsilon)
# glue.agent_message(command)

# # set discount
# command = "set discount:{}".format(discount)
# glue.agent_message(command)




# TODO: add options incrementally to actions set

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

# Specify name of external env and agent
env_name = "Environment"
agent_name = "QAgent" 

# getattr(Environment, 'Environment')(@@@) # call by string
external_env = Environment.Environment(max_row, max_col, (0,9))
external_agent = Agents.QAgent(max_row, max_col)

internal_env = copy.copy(external_env)
internal_agent = copy.copy(external_agent)
opt = options.Options(internal_env, internal_agent, alpha=0.1, epsilon=0.1, discount=1.0)


###### DEBUG: learn one option
# add options incrementally to actions set
opt.learn_eigenoption(100000)

# display newest learned option
opt.display_eigenoption()

exit()
##############################

#rlglue = RLGlue.RLGlue(external_env, external_agent)



import sys
import numpy as np
import pickle
import options
import plot_utils
import copy
# choose environment
#import env_grid as env
import Environment

np.set_printoptions(precision=2)


# Specify name of external env and agent
env_name = "Environment"
agent_name = "QAgent" # unnecessary for now bc we always use QAgent

# Set up environment
max_row = 10
max_col = 10
external_env = Environment.Environment(max_row, max_col, (0,9))
internal_env = copy.copy(external_env)


opt = options.Options(internal_env, alpha=0.1, epsilon=0.1, discount=1.0)

# add options incrementally to actions set
opt.learn_eigenoption(10000)

opt.display_eigenoption()



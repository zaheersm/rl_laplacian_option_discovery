import sys
import os
import numpy as np
import pickle
import copy

import rlglue
import environment
import agents
import options


def createFolder(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print ('Error: Creating directory. ' +  directory)

np.set_printoptions(precision=2)


# Experiment parameter
alpha = 0.1
epsilon = 0.1 # 1.0
discount = 1.0
save_dir = './I_maze_env_option_archive'
num_options = 200 # 200, 242, 90

# Specify name of env and agent
env_name = "I_MazeEnvironment"
agent_name = "QAgent" 

external_env = getattr(environment, env_name)()

max_row, max_col = external_env.get_dim()
external_agent = getattr(agents, agent_name)(max_row, max_col)


internal_env = copy.copy(external_env)
internal_agent = copy.copy(external_agent)


createFolder(save_dir)
# Compute eigenoptions
opt = options.Options(internal_env, internal_agent, alpha=alpha, epsilon=epsilon, discount=discount)

for idx in range(num_options):
    print("==== processing options: "+str(idx) + " ====")
    eigenoption = opt.learn_next_eigenoption(100000)
    savename = save_dir + '/option{}.txt'.format(idx)
    np.savetxt(savename, np.array(eigenoption), fmt='%d')
    # display or save newest learned option
    savename = save_dir + '/option{}.png'.format(idx)
    opt.display_eigenoption(display=False, savename=savename, idx=idx)



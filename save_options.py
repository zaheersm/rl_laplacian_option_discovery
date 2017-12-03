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


save_dir = './grid_env_test'
num_options = 200

explore_env = environment.GridEnvironment()

max_row, max_col = explore_env.get_grid_dimension() # get dimension of the environment
explore_agent = agents.OptionExploreQAgent(max_row=max_row, max_col=max_col)

explore_agent.set_alpha(0.1)
explore_agent.set_discount(0.9)
explore_glue = rlglue.RLGlue(explore_env, explore_agent)

# Option object would learn eigen-options for the enviornment
opt = options.Options(alpha=0.1, epsilon=1.0, discount=0.9)


createFolder(save_dir)

for idx in range(num_options):
    print("==== learning option: "+str(idx) + " ====")
    eigenoption = opt.learn_next_eigenoption(100000)
    savename = save_dir+'/option{}.txt'.format(idx)
    np.savetxt(savename, np.array(eigenoption), fmt='%d')
    # display or save newest learned option
    savename = save_dir+'/option{}.png'.format(idx)
    opt.display_eigenoption(display=False, savename=savename, idx=idx)



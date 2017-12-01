import sys
import numpy as np
import pickle
import copy

import rlglue
import environment
import agents
import options

np.set_printoptions(precision=2)

explore_env = environment.GridEnvironment(max_row=10, max_col=10,
                                          goal_state=(0,9))
explore_env.set_start_state((9,0))
explore_agent = agents.OptionExploreQAgent(max_row=10, max_col=10)
explore_agent.set_alpha(0.1)
explore_agent.set_discount(0.9)
explore_glue = rlglue.RLGlue(explore_env, explore_agent)

# Option object would learn eigen-options for the enviornment
opt = options.Options(alpha=0.1, epsilon=1.0, discount=0.9)

for idx in range(200):
    print("==== learning option: "+str(idx) + " ====")
    eigenoption = opt.learn_next_eigenoption(100000)
    savename = 'option_archive/option{}.txt'.format(idx)
    np.savetxt(savename, np.array(eigenoption), fmt='%d')
    # display or save newest learned option
    savename = 'option_archive/option{}.png'.format(idx)
    opt.display_eigenoption(display=False, savename=savename, idx=idx)

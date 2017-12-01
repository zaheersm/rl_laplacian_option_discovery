import sys
import numpy as np
import pickle
import copy

import rlglue
import environment
import agents
import options

# Setting up explore_agent which would learn Q-values using options
explore_env = environment.GridEnvironment(max_row=10, max_col=10,
                                          goal_state=(0,9))
explore_env.set_start_state((9,0))
explore_agent = agents.OptionExploreQAgent(max_row=10, max_col=10)
explore_agent.set_alpha(0.1)
explore_agent.set_discount(0.9)
explore_glue = rlglue.RLGlue(explore_env, explore_agent)

# Setting up reward_agent which would use Q-values learnt by explore_agent
# to accumulate reward
reward_env = environment.GridEnvironment(max_row=10, max_col=10,
                                         goal_state=(0,9))
reward_env.set_start_state((9,0))
reward_agent = agents.QAgent(max_row=10, max_col=10)
reward_agent.set_alpha(0.1)
reward_agent.set_epsilon(0.0)
reward_agent.set_discount(0.9)
reward_glue = rlglue.RLGlue(reward_env, reward_agent)

# Option object would learn eigen-options for the enviornment
opt = options.Options(alpha=0.1, epsilon=1.0, discount=0.9)

# Experiment
np.set_printoptions(precision=2)

# Experiment parameter
num_runs = 50
num_episodes = 100
num_options = 200

# Starting from the agent with primitive actions, we incrementally add options
# in explore_agent
results = np.zeros((num_options+1, num_episodes))

current_num_options = 0
for i in [0,2,4,8,64,128, 200]:
    print('Explore Agent with ' + str(i) + ' options...')
    # add option
    while current_num_options < i:
        eigenoption = opt.learn_next_eigenoption(100000)
        # display or save newest learned option
	# opt.display_eigenoption(display=False,
        #                        savename='option'+str(i)+'.png', idx=i-1)
        explore_agent.add_eigenoption(eigenoption)
        current_num_options += 1
    cum_reward = np.zeros(num_episodes)
    for run in range(num_runs):
        for ep in range(num_episodes):
            # run episode
            explore_glue.episode(100)
            learned_Q = explore_agent.get_Q()
            reward_agent.set_Q(learned_Q)
            reward_glue.episode(100)
            cum_reward[ep] += reward_glue.get_total_reward()
            reward_glue.cleanup()
        explore_glue.cleanup()
    cum_reward /= float(num_runs)
    results[i] = cum_reward
np.save('data_files/average_return', results)

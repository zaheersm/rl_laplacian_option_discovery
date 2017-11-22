import sys
import numpy as np
import pickle
import copy

import rlglue
import environment
import agents
import options

np.set_printoptions(precision=2)

# Experiment parameter
num_runs = 100
num_episodes = 250

alpha = 0.1
epsilon = 0.0
discount = 0.9

num_options = 100

# TODO: Add random seed for random start state / random action selection
# state_random_seed_arr = []
# for i in range(num_episodes):
# 	random_seed_arr.append(np.random.randint(1000))



# Set up environment
max_row = 10
max_col = 10
obstacles = [] # TODO: handle obstacles
start_state = (9,0)
goal_state = (0,9)

# Specify name of env and agent
env_name = "GridEnvironment"
agent_name = "QAgent" 

external_env = getattr(environment, env_name)(max_row, max_col, goal_state)
external_agent = getattr(agents, agent_name)(max_row, max_col)
internal_env = copy.deepcopy(external_env)
internal_agent = copy.deepcopy(external_agent)


# Baseline Agent
glue = rlglue.RLGlue(external_env, external_agent)
glue.env_message("set start_state:"+pickle.dumps(start_state, protocol=0))
glue.agent_message("set alpha:{}".format(alpha))
glue.agent_message("set epsilon:{}".format(epsilon))
glue.agent_message("set discount:{}".format(discount))


# Compute eigenoptions
print("Computing eigenvalues")
opt = options.Options(internal_env, internal_agent, alpha=alpha, epsilon=epsilon, discount=discount)



#### Test Agent #### does not learn, and takes greedy action
test_env = getattr(environment, env_name)(max_row, max_col, goal_state)
test_agent = getattr(agents, agent_name)(max_row, max_col)
glue_test = rlglue.RLGlue(test_env, test_agent)

glue_test.env_message("set start_state:"+pickle.dumps(start_state, protocol=0))
glue_test.agent_message("set alpha:{}".format(alpha))
glue_test.agent_message("set epsilon:{}".format(epsilon))
glue_test.agent_message("set discount:{}".format(discount))
glue_test.agent_message('TEST ON') # disable learning and set epsilon = 0.0 if not already


### Experiment
# starting from base agent, we incrementally add options
results = np.zeros((num_options+1, num_episodes))

current_num_options = 0
for i in [0,2,4,8,64]:
	print('Computing Agent with ' + str(i) + ' options...')
	# add option
	while current_num_options < i:
		eigenoption = opt.learn_next_eigenoption(100000)
		# display or save newest learned option
		#opt.display_eigenoption(display=False, savename='option'+str(i)+'.png', idx=i-1)
		glue.agent_message("set eigen_option:" + pickle.dumps(eigenoption, protocol=0))

		current_num_options += 1
		
	# learn 
	cum_reward = np.zeros(num_episodes)
	for run in range(num_runs):
		for ep in range(num_episodes):
			# run episode 
			glue.episode(100)
			learned_Q = pickle.loads(glue.agent_message('Get Q'))

			glue_test.agent_message('Set Q:'+pickle.dumps(learned_Q))
			glue_test.episode(100)
			cum_reward[ep] += glue_test.get_total_reward()
			glue_test.cleanup()
		glue.cleanup()
		
	cum_reward /= float(num_runs)
	results[i] = cum_reward
	print(i, results[i])

np.save('ave_return', results)


#  ---------------------------- 1 AGENT ----------------------------------#

# import libraries
import environment
import environment_prey
import time
import copy
import numpy as np
import utils as utl
import matplotlib.pyplot as plt
import pickle


# GRID SIZE
N = 10

# Q-LEARNING PARAMS
epsilon = 0.1
#max_epsilon = 0.1
#min_epsilon = 0.00001
#epsilon_decay_rate = 0.0001
alpha = 0.1
gamma = 0.9

# TRAINING PARAMS
n_episodes = 20000 # num of episodes
max_steps = 5000 # max steps of each episode

#epsilon_list = [epsilon] # for decreasing epsilon
#alpha_list = [alpha] # for decreasing alpha



# -------------------------- RUN EXPERIMENTS ----------------------------#
'''
# create environment by passing the grid size
env = environment.Environment(N) # with fixed prey
#env = environment_prey.Environment(N) # with moving prey


model_name = "single_Agent_" + "Fixed_Prey_" + "Grid_" + str(N)
#model_name = "single_Agent_" + "Moving_Prey_" + "Grid_" + str(N)


# RUN THE FUNCTIONS

time_goal = utl.run_multiple_episodes(n_episodes, env, max_steps, epsilon, alpha)

# save environment object into file for later retrieval
with open(model_name + "_env", "wb") as fp:
    pickle.dump(env, fp)

# save TG into file for later retrieval
with open(model_name + "_list", "wb") as fp:
    pickle.dump(time_goal, fp)

# save TG into file for later retrieval
with open(model_name + "_list", "wb") as fp:
    pickle.dump(time_goal, fp)

# PLot TG over the episodes
utl.plot_time_to_goal(model_name, n_episodes, time_goal)
plt.clf()

# Plot TG averaged every 100 episodes
utl.plot_time_to_goal(model_name, n_episodes,  time_goal, avg=1)
plt.clf()
'''


# --------------------------------- COMPARISONS -------------------------------#

'''
# load environment
model_name = "single_Agent_" + "Fixed_Prey_" + "Grid_" + str(N)
with open(model_name + "_env", "rb") as fp:
    env = pickle.load(fp)
    # reset the environment before simulation

# retrieve Q-table
print(env.preds[0].Q)

# run simulation
#utl.run_simulation(env)
'''

'''
# load TG lists for combined plot
# load model with fixed prey
model_name1 = "single_Agent_" + "Fixed_Prey_" + "Grid_" + str(N)
with open(model_name1 + "_list", "rb") as fp:
    tg1 = pickle.load(fp)

# load model with moving prey
model_name2 = "single_Agent_" + "Moving_Prey_" + "Grid_" + str(N)
with open(model_name2 + "_list", "rb") as fp:
    tg2 = pickle.load(fp)

# Plot comparison of TGs 
utl.plot_TG_comparison(model_name1, model_name2, tg1, tg2, n_episodes)
plt.clf()
'''


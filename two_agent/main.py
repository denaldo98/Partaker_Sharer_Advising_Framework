#  ---------------------------- 2 AGENTS WITH COMMUNICATION ----------------------------------#

# import libraries
import environment_comm
import environment_comm_prey
import environment_no_comm_prey
import environment_no_comm
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
#max_epsilon = 1
#min_epsilon = 0.00001
#epsilon_decay_rate = 0.001
alpha = 0.1 
gamma = 0.9

# TRAINING PARAMS
n_episodes = 20000 # num of episodes
max_steps = 5000 # max steps of each episode

#epsilon_list = [epsilon] # for decreasing epsilon
#alpha_list = [alpha] # for decreasing alpha

# -------------------------- RUN EXPERIMENTS ----------------------------#

# create environment by passing the grid size
#env = environment_no_comm.Environment(N)
#env = environment_no_comm_prey.Environment(N)
#env = environment_comm.Environment(N)
env = environment_comm_prey.Environment(N)


#model_name = "2_Agents_" + "no_comm_" + "Fixed_Prey_" + "Grid_" + str(N)
#model_name = "2_Agents_" + "no_comm_" + "Moving_Prey_" + "Grid_" + str(N)
#model_name = "2_Agents_" + "comm_" + "Fixed_Prey_" + "Grid_" + str(N)
model_name = "2_Agents_" + "comm_" + "Moving_Prey_" + "Grid_" + str(N)
#model_name = "single_Agent_" + "Moving_Prey_" + "Grid_" + str(N)


# RUN THE FUNCTIONS

time_goal, b_ask1_list, b_give1_list = utl.run_multiple_episodes(n_episodes, env, max_steps, epsilon, alpha)

# save TG into file for later retrieval
with open(model_name + "_list", "wb") as fp:
    pickle.dump(time_goal, fp)

# PLot TG over the episodes
utl.plot_time_to_goal(model_name, n_episodes, time_goal)
plt.clf()

# Plot TG averaged every 100 episodes
utl.plot_time_to_goal(model_name, n_episodes,  time_goal, avg=1)
plt.clf()

# plot budgets only for PSAF models
if b_ask1_list != 0:
    # Plot b_ask1
    utl.plot_budget(model_name, "b_ask", n_episodes, b_ask1_list)
    plt.clf()

    # Plot b_give1
    utl.plot_budget(model_name, "b_give", n_episodes, b_give1_list)
    plt.clf()

# run simulation
#utl.run_simulation(env)



# --------------------------------- COMPARISONS -------------------------------#

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

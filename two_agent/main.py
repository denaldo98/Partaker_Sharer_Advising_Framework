#  ---------------------------- 2 AGENTS WITH COMMUNICATION ----------------------------------#

# import libraries
import numpy as np
import utils as utl
import matplotlib.pyplot as plt
import pickle

# import environments
import environment_comm
import environment_comm_prey
import environment_no_comm_prey
import environment_no_comm


# GRID SIZE
N = 10

# Q-LEARNING PARAMS
epsilon = 0.1 
alpha = 0.1 
gamma = 0.9

# TRAINING PARAMS
n_episodes = 20000 # num of episodes
max_steps = 5000 # max steps of each episode (if goal not reached)


# -------------------------- RUN EXPERIMENTS ----------------------------#

# CREATE ENVIRONMENT by passing the grid size
#env = environment_no_comm.Environment(N) # non communicating, fixed prey
#env = environment_no_comm_prey.Environment(N) # non communicating, moving prey
#env = environment_comm.Environment(N) # communicating, fixed prey
env = environment_comm_prey.Environment(N) # communicating, moving prey

# define NAME OF THE MODEL for saving files with correct names
#model_name = "2_Agents_" + "no_comm_" + "Fixed_Prey_" + "Grid_" + str(N)
#model_name = "2_Agents_" + "no_comm_" + "Moving_Prey_" + "Grid_" + str(N)
#model_name = "2_Agents_" + "comm_" + "Fixed_Prey_" + "Grid_" + str(N)
model_name = "2_Agents_" + "comm_" + "Moving_Prey_" + "Grid_" + str(N)


# RUN THE FUNCTIONS defined in utils.py

# perform 1 TRAINING of n_episodes
#time_goal, b_ask1_list, b_give1_list = utl.run_multiple_episodes(n_episodes, env, max_steps, epsilon, alpha)

# perform 1 TRAINING of n_episodes with GRAPHICAL SIMULATION
time_goal, b_ask1_list, b_give1_list = utl.run_training_simulation(n_episodes, env, max_steps, epsilon, alpha)

# perform 1 TRAINING of n_episodes with GRAPHICAL SIMULATION EVERY 5000 EPISODES
#time_goal, b_ask1_list, b_give1_list = utl.run_training_simulation2(n_episodes, env, max_steps, epsilon, alpha)

# repat TRAINING process n_processes time (pay attention to the name of the environment)
# env_type parameter in the repeat_process() function can be : "comm_prey", "no_comm_prey", "comm"
#n_processes = 20
#time_goal, b_ask1_list, b_give1_list = utl.repeat_process(n_processes, n_episodes, max_steps, epsilon, alpha, "no_comm_prey", N)


'''
# SAVE environment object into file for later retrieval
# useful when performing 1 training of n_episodes to later see what the agents have learned
with open(model_name + "_env", "wb") as fp:
    pickle.dump(env, fp)

'''
'''
# save TG  into file for later retrieval
with open(model_name + "_TG_list", "wb") as fp:
    pickle.dump(time_goal, fp)

# save budgets
with open(model_name + "_b_ask_list", "wb") as fp:
    pickle.dump(b_ask1_list, fp)

with open(model_name + "_b_give_list", "wb") as fp:
    pickle.dump(b_give1_list, fp)
'''


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



'''
# ---------- load environment for retrieving Q-table or for running simulation with the learned policy ---------- #
# select the name of the model to load
model_name = "2_Agents_" + "comm_" + "Moving_Prey_" + "Grid_" + str(N)
with open(model_name + "_env", "rb") as fp:
    env = pickle.load(fp)

# retrieve Q-table
#print(env.preds[0].Q)

# run simulation after training
utl.run_simulation(env, with_budget=0)



# ------------------------------------ COMPARISONS of different models learning ----------------------------------#

# ----------load TG lists for combined plot----------- #
# load model with fixed prey
model_name1 = "2_Agents_" + "comm_" + "Fixed_Prey_" + "Grid_" + str(N)
with open(model_name1 + "_TG_list", "rb") as fp: # TG_list of 1st model
    tg1 = pickle.load(fp)

# load model with moving prey
model_name2 = "2_Agents_" + "comm_" + "Moving_Prey_" + "Grid_" + str(N)
with open(model_name2 + "_TG_list", "rb") as fp: # TG_list of 2nd model
    tg2 = pickle.load(fp)

# Plot comparison of TGs 
utl.plot_TG_comparison(model_name1, model_name2, tg1, tg2, n_episodes, avg = 1) # avg=1 to obtain avg every 100 episodes
plt.clf()


# ---------- load BUDGET lists for combined plot---------- #
# load model with fixed prey
model_name1 = "2_Agents_" + "comm_" + "Fixed_Prey_" + "Grid_" + str(N)
with open(model_name1 + "_b_ask_list", "rb") as fp: # b_ask list of 1st model
    bg1 = pickle.load(fp)

# load model with moving prey
model_name2 = "2_Agents_" + "comm_" + "Moving_Prey_" + "Grid_" + str(N)
with open(model_name2 + "_b_ask_list", "rb") as fp: # b_ask list of 2nd model
    bg2 = pickle.load(fp)

# Plot comparison of budgets 
utl.plot_budget_comparison(model_name1, model_name2, bg1, bg2, n_episodes, avg = 1) # avg=1 to obtain avg every 100 episodes
plt.clf()
'''
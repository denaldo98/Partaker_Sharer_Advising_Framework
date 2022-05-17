# import libraries
import environment_comm
import environment_no_comm 
import utils_comm as utl
import time
import copy
import numpy as np

import matplotlib.pyplot as plt
plt.style.use('seaborn-darkgrid')
plt.rcParams["figure.figsize"] = (12, 6)

ACTION_TO_STRING = ("down", "up", "left", "right", "stay")

# GRID SIZE
N = 12

# Q-LEARNING PARAMS
epsilon = 0.1 # decreasing
max_epsilon = 1
min_epsilon = 0.00001
epsilon_decay_rate = 0.001

alpha = 0.1 # fixed, for now
#alpha = 1

gamma = 0.9



# TRAINING PARAMS
n_episodes = 20000 # num of  episodes
max_steps = 5000 # max steps of each episode



#epsilon_list = [epsilon] # for decreasing epsilon
#alpha_list = [alpha] # for decreasing alpha


def run_multiple_episodes(n_episodes, env, epsilon, alpha):

    time_goal = []
    b_ask_list1 = [] # list of b_ask of 1st predator (every episode)

    print("STARTING TRAINING")

    for episode in range(n_episodes):
        env.reset(epsilon, alpha) # randomly initialize the state of the env
        goal = 0 # encodes the reaching of a goal state (1 if goal state)
        performed_steps = 0 # TG

        # run episode until reaching goal state or max_steps
        while ((goal != 1) and (performed_steps < max_steps)):
            goal, actions, b_ask1 = env.transition() # returns 1 if goal state is reached
            performed_steps += 1

        # AFTER EACH EPISODE
        # reduce epsilon
        #epsilon = min_epsilon + (max_epsilon - min_epsilon) * np.exp(-epsilon_decay_rate * episode)
        #epsilon_list.append(epsilon)

        # append to lists
        time_goal.append(performed_steps)
        b_ask_list1.append(b_ask1)

        # reduce alpha

        if episode % 50 == 0:
            print(f"Completed {episode} / {n_episodes} episodes")

    return time_goal, b_ask_list1
        



# RUN THE PROCESS 200 TIMES without communication
tg_list_no_comm = [] # tg list of each process iteration
b_ask1_list_no_comm = [] # b_ask1 list of each process iteration

print("\n\n\n....STARTING TRAINING WITHOUT COMMUNICATION....\n\n")
start = time.time()

processes = 3 # number of times to repeat the training process
for i in range(processes):
    print("\n\n\nPROCESS ITERATION NUMBER: {}\n".format(i+1))
    start_time = time.time()
    # create environment by passing the grid size
    env = environment_no_comm.Environment(N)
    #epsilon = 1
    tg, b_ask1 = run_multiple_episodes(n_episodes, env, epsilon, alpha)
    print("\n\nPROCESS ITERATION REQUIRED TIME: {} seconds".format(time.time()-start_time))

    averages_tg = [] # average every 100 episodes
    for j in range(int(n_episodes/100)):
            averages_tg.append(np.mean(tg[j*100: j*100 + 100]))
    tg_list_no_comm.append(averages_tg)
    
    averages_b1 = []
    for j in range(int(n_episodes/100)):
            averages_b1.append(np.mean(b_ask1[j*100: j*100 + 100]))
    b_ask1_list_no_comm.append(averages_b1)

print("\n\n\n\nTRAINING FINISHED")
total_time = time.time() - start
print("\nTOTAL REQUIRED TIME: {} seconds, {} minutes".format(total_time, total_time/60))

# NOW, we need to perform average over the various processes:
time_to_goal_final_no_comm = [0 for i in range(int(n_episodes/100))] # intialize final list of times_to_goal

for i in range(int(n_episodes/100)):
    for j in range(processes):
        time_to_goal_final_no_comm[i] += tg_list_no_comm[j][i]

time_to_goal_final_no_comm = [time_to_goal_final_no_comm[i]/processes for i in range(int(n_episodes/100))]






# RUN THE PROCESS 200 TIMES with communication
tg_list_comm = [] # tg list of each process iteration
b_ask1_list_comm = [] # b_ask1 list of each process iteration

print("\n\n\n\n\n\n....STARTING TRAINING WITH COMMUNICATION....\n\n")
start = time.time()

for i in range(processes):
    print("\n\n\nPROCESS ITERATION NUMBER: {}\n".format(i+1))
    start_time = time.time()
    # create environment by passing the grid size
    env2 = environment_comm.Environment(N)
    #epsilon = 1
    tg, b_ask1 = run_multiple_episodes(n_episodes, env2, epsilon, alpha)
    print("\n\nPROCESS ITERATION REQUIRED TIME: {} seconds".format(time.time()-start_time))

    averages_tg = [] # average every 100 episodes
    for j in range(int(n_episodes/100)):
            averages_tg.append(np.mean(tg[j*100: j*100 + 100]))
    tg_list_comm.append(averages_tg)
    
    averages_b1 = []
    for j in range(int(n_episodes/100)):
            averages_b1.append(np.mean(b_ask1[j*100: j*100 + 100]))
    b_ask1_list_comm.append(averages_b1)

print("\n\n\n\nTRAINING FINISHED")
total_time = time.time() - start
print("\nTOTAL REQUIRED TIME: {} seconds, {} minutes".format(total_time, total_time/60))

# NOW, we need to perform average over the various processes:
time_to_goal_final_comm = [0 for i in range(int(n_episodes/100))] # intialize final list of times_to_goal

for i in range(int(n_episodes/100)):
    for j in range(processes):
        time_to_goal_final_comm[i] += tg_list_comm[j][i]

time_to_goal_final_comm = [time_to_goal_final_comm[i]/processes for i in range(int(n_episodes/100))]






# PLOT OF THE TIME TO GOAL OVER THE PROCESSES

plt.plot(range(int(n_episodes/100)), time_to_goal_final_comm, label="PSAF")
plt.plot(range(int(n_episodes/100)), time_to_goal_final_no_comm, label="multi-IQL")
plt.title("Time To Goal over the episodes (avg over 200 processes)")
plt.xlabel("Training Episodes (x100)")
plt.ylabel("Time To Goal")
plt.legend()
plt.tight_layout()
plt.show()




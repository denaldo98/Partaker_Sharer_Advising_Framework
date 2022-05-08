# 2 AGENTs

# import libraries
import string
import environment
import time
import copy
import numpy as np

import matplotlib.pyplot as plt
plt.style.use('seaborn-darkgrid')
plt.rcParams["figure.figsize"] = (12, 6)

ACTION_TO_STRING = ("down", "up", "left", "right", "stay")

# GRID SIZE
N = 10

# Q-LEARNING PARAMS
epsilon = 0.1
max_epsilon = 0.1
min_epsilon = 0.00001
epsilon_decay_rate = 0.0001

alpha = 0.1
#alpha = 1
gamma = 0.9

# create environment by passing the grid size
env = environment.Environment(N)

# num of TRAINING episodes
n_episodes = 40000

# max steps of each episode
max_steps = 5000

# Time to goal
time_goal = []

epsilon_list = [epsilon] # for decreasing epsilon
alpha_list = [alpha] # for decreasing alpha


# RUN 1 episode until reaching goal state
def run_one_episode(env, epsilon , alpha, ACTION_TO_STRING):

    start_time = time.time()
    print("STARTING TRAINING\n\n")

    env.reset(epsilon, alpha) # randomly initialize the state of the env

    goal = 0 # encodes the reaching of a goal state (1 if goal state)
    performed_steps = 0 # steps performed in the episode (TG)

    print("INITIAL STATE:")
    print(env) 
    print(env.pred_locs[0], env.pred_locs[1], env.prey_loc[0]) # STATE

    while goal != 1:   
        old_q_table = copy.copy(env.preds[0].Q)
        goal, act = env.transition() # PERFORM 1 TRANSITION
        new_q_table = env.preds[0].Q
        performed_steps +=1

        # reduce epsilon at each step
        epsilon = min_epsilon + (max_epsilon - min_epsilon) * np.exp(-epsilon_decay_rate * performed_steps)
        epsilon_list.append(epsilon)

        print("chosen action: {}".format(ACTION_TO_STRING[act[0]]))
        print("REWARD: {}".format(goal))
        print("\n\n new state:")
        print(env)
        print(env.pred_locs[0], env.pred_locs[1], env.prey_loc[0]) # NEW STATE
        print("OLD Q: {}\n".format(old_q_table))
        print("NEW_Q: {}\n".format(new_q_table))


    end_time = time.time()
    print("\n\nTRAINING FINISHED") 
    print("\nTotal required time: {} seconds, {} minutes".format(end_time - start_time, (end_time - start_time)/60))



def run_multiple_episodes(n_episodes, env, epsilon, alpha):

    start_time = time.time()
    print("STARTING TRAINING")

    for episode in range(n_episodes):
        
        env.reset(epsilon, alpha) # randomly initialize the state of the env
        
        goal = 0 # encodes the reaching of a goal state (1 if goal state)

        performed_steps = 0 # TG

        # run episode until reaching goal state or max_steps
        while ((goal != 1) and (performed_steps < max_steps)):

            goal, _ = env.transition() # returns 1 if goal state is reached

            performed_steps += 1
        
        time_goal.append(performed_steps)

        # reduce epsilon
        epsilon = min_epsilon + (max_epsilon - min_epsilon) * np.exp(-epsilon_decay_rate * episode)
        epsilon_list.append(epsilon)

        # reduce alpha

        if episode % 20 == 0:
            print(f"Completed {episode} / {n_episodes} episodes")

    end_time = time.time()

    print("\n\nTRAINING FINISHED") 
    print("\nTotal required time: {} seconds, {} minutes".format(end_time - start_time, (end_time - start_time)/60))

    # AVERAGE EVERY 100 episodes
    averages = []
    for i in range(int(n_episodes/100)):
        averages.append(np.mean(time_goal[i*100: i*100 + 100]))

    # PLOT THE TIME TO GOAL
    print("Plot of the Time To Goal over the episodes")
    plt.plot(range(n_episodes), time_goal)
    plt.title("Time To Goal over the episodes")
    plt.xlabel("Training Episodes")
    plt.ylabel("Time To Goal")
    #plt.ylim([0, 600])
    plt.tight_layout()
    plt.show()

    print("Plot of the avg Time To Goal over the episodes")
    plt.plot(range(int(n_episodes/100)), averages)
    plt.title("Time To Goal over the episodes")
    plt.xlabel("Training Episodes (x 100)")
    plt.ylabel("Time To Goal")
    #plt.ylim([0, 600])
    plt.tight_layout()
    plt.show()


    print("\nEpsilon list: {}".format(epsilon_list[19990:19999]))


# SIMULATION (with the learned Q-values)
def run_simulation(env):
    
    reward = "Initial configuration -->NO REWARD"
    action = np.array(["No action"])
    time_step = 0
    goal = 0
    print("\n\n\n\nNow watch environment proceed step-by-step")
    env.reset(epsilon=0, alpha=0)
    while not goal:
        print(env)
        print("Timestep: {}".format(time_step))
        #print("Predators and prey coordinates: {}".format(env.get_positions()))
        print("Predator coordinates: {}".format(env.pred_locs[0]))
        print("Prey coordinate: {}".format(env.prey_loc[0]))
        if action[0] != "No action":
            print("Chosen action: {}".format(ACTION_TO_STRING[int(action[0])]))
        #print("Action chosen by second predator: {}".format(action[1]))
        print("Reward: {}".format(reward))
        reward, action = env.transition()
        goal = reward
        time_step += 1
        input("Press enter to visualize next state")
    print(env)
    print("Goal state reached!")
    print("Performed steps: {}".format(time_step))






#run_one_episode(env, epsilon, alpha, ACTION_TO_STRING)
run_multiple_episodes(n_episodes, env, epsilon, alpha)

run_simulation(env)



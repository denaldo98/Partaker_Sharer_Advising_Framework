import numpy as np
import time
import matplotlib.pyplot as plt
plt.style.use('seaborn-darkgrid')
plt.rcParams["figure.figsize"] = (12, 6)
from numba import jit
import os
import time

# import environments
import environment_comm
import environment_no_comm
import environment_comm_prey
import environment_no_comm_prey


ACTION_TO_STRING = ("down", "up", "left", "right", "stay")


# auxiliary function to clear the output (for GRID VISUALIZATION)
def cls():
    os.system('cls' if os.name=='nt' else 'clear')


# ---------------- TRAINING FUNCTIONS -----------------#

# run many episodes
@jit
def run_multiple_episodes(n_episodes, env, max_steps, epsilon, alpha):
    
    # Time to goal of each episode
    time_goal = []

    # b_ask of 1st agent
    b_ask1_list = []

    # b_give of 1st agent
    b_give1_list = []
    
    start_time = time.time()
    print("STARTING TRAINING")

    for episode in range(n_episodes):

        # randomly initialize the state of the env
        env.reset(epsilon, alpha) 

        goal = 0 # 1 if goal state
        performed_steps = 0 # TG

        # run episode until reaching goal state or max_steps
        while ((goal != 1) and (performed_steps < max_steps)):
            goal, actions, prey_action, b_ask1, b_give1 = env.transition() # PERFORM 1 TRANSITION
            performed_steps += 1
        
        # Append TG of each episode
        time_goal.append(performed_steps)

        # append b_ask1 and b_give1 after each episode
        b_ask1_list.append(b_ask1)
        b_give1_list.append(b_give1)

        # show training episodes
        if episode % 50 == 0:
            print(f"Completed {episode} / {n_episodes} episodes")
        
    end_time = time.time()

    print("\n\nTRAINING FINISHED") 
    print("\nTotal required time: {} seconds, {} minutes".format(end_time - start_time, (end_time - start_time)/60))

    return time_goal, b_ask1_list, b_give1_list


# TRAINING SIMULATION OF AGENT COMMUNICATION
@jit
def run_training_simulation(n_episodes, env, max_steps, epsilon, alpha):

    # Time to goal of each episode
    time_goal = []

    # b_ask of 1st agent
    b_ask1_list = []

    # b_give of 1st agent
    b_give1_list = []
    
    start_time = time.time()
    print("STARTING TRAINING")

    for episode in range(n_episodes):
        print("\n\nEpisode: {}".format(episode))

        # # randomly initialize the state of the env
        env.reset(epsilon, alpha)

        goal = 0 # 1 if goal state
        performed_steps = 0 # TG

        reward = "Initial configuration -->NO REWARD"
        actions = np.array(["No action", "No action"])
        prey_action = "No action"
        print("Now watch TRAINING proceed step-by-step")

        # run episode until reaching goal state or max_steps
        while ((goal != 1) and (performed_steps < max_steps)):

            # print useful info
            print(env)
            print("Episode: {}".format(episode))
            print("Timestep: {}".format(performed_steps))
            print("Predator X coordinates: {}".format(env.pred_locs[0]))
            print("Predator Y coordinates: {}".format(env.pred_locs[1]))
            print("Prey O coordinate: {}".format(env.prey_loc[0]))
            print("b_ask of X: {}".format(env.preds[0].b_ask))
            print("b_give of X: {}".format(env.preds[0].b_give))
            print("b_ask of Y: {}".format(env.preds[1].b_ask))
            print("b_give of Y: {}".format(env.preds[1].b_give))
            if actions[0] != "No action":
                print("Action chosen by X: {}".format(ACTION_TO_STRING[int(actions[0])]))
                print("Action chosen by Y: {}".format(ACTION_TO_STRING[int(actions[1])]))
                print("Prey action: {}".format(ACTION_TO_STRING[prey_action]))
            print("Reward: {}".format(reward))
            reward, actions, prey_action, b_ask1, b_give1 = env.transition()
            goal = reward
            performed_steps += 1
            #time.sleep(0.25) # sleep to see plot
            cls() # clear previous output from terminal

        # Append TG of each episode
        time_goal.append(performed_steps)

        # append b_ask1 and b_give1 after each episode
        b_ask1_list.append(b_ask1)
        b_give1_list.append(b_give1)

        print(env)
        if goal:
            print("Goal state reached!")
        else:
            print("Max number of steps reached")

        print("Predator X coordinates: {}".format(env.pred_locs[0]))
        print("Predator Y coordinates: {}".format(env.pred_locs[1]))
        print("Prey O coordinate: {}".format(env.prey_loc[0]))
        print("b_ask of X: {}".format(env.preds[0].b_ask))
        print("b_give of X: {}".format(env.preds[0].b_give))
        print("b_ask of Y: {}".format(env.preds[1].b_ask))
        print("b_give of Y: {}".format(env.preds[1].b_give))
        print("Action chosen by X: {}".format(ACTION_TO_STRING[int(actions[0])]))
        print("Action chosen by Y: {}".format(ACTION_TO_STRING[int(actions[1])]))
        print("Prey action: {}".format(ACTION_TO_STRING[prey_action]))

        print("Reward: {}".format(reward))
        print("Performed steps: {}".format(performed_steps))

    end_time = time.time()
    print("\n\nTRAINING FINISHED") 
    print("\nTotal required time: {} seconds, {} minutes".format(end_time - start_time, (end_time - start_time)/60))

    return time_goal, b_ask1_list, b_give1_list


# SIMULATION (with the LEARNED POLICY)
def run_simulation(env, with_budget = 0):
    
    reward = "Initial configuration -->NO REWARD"
    action = np.array(["No action", "No action"])
    prey_action = "No action"
    time_step = 0
    goal = 0
    print("\n\n\n\nNow watch environment proceed step-by-step")

    # zero alpha end epsilon to show the learned policy
    env.reset(epsilon=0, alpha=0)
    
    # in case of agents with budget, set it to zero
    if with_budget:
        env.set_budget()
    # perform 1 full episode
    while not goal:
        print(env)
        print("Timestep: {}".format(time_step))
        print("Predator X coordinates: {}".format(env.pred_locs[0]))
        print("Predator Y coordinates: {}".format(env.pred_locs[1]))
        print("Prey O coordinate: {}".format(env.prey_loc[0]))
        if action[0] != "No action":
            print("Action chosen by X: {}".format(ACTION_TO_STRING[int(action[0])]))
            print("Action chosen by Y: {}".format(ACTION_TO_STRING[int(action[1])]))
            if prey_action != -1:
                print("Prey action: {}".format(ACTION_TO_STRING[prey_action]))

        print("Reward: {}".format(reward))
        reward, action, prey_action, b_ask1, b_give1 = env.transition()
        goal = reward
        time_step += 1
        input("Press enter to visualize next state")
    print(env)
    print("Goal state reached!")
    print("Performed steps: {}".format(time_step))


# repeat process many times (200 runs in the paper)
@jit
def repeat_process(n_processes, n_episodes, max_steps, epsilon, alpha, env_type, env_size):

    # TG list of each process iteration
    tg_list = []

    # b_ask1 list of each process iteration
    b_ask1_list = []

    # b_give1 of each process iteration
    b_give1_list = []

    start = time.time()
    print("STARTING TRAINING")
    
    # iterate over number of processes
    for i in range(n_processes):

        print("\n\n\nPROCESS ITERATION NUMBER: {}".format(i+1))
        start_time = time.time()

        # at each process iteration we re-create the environment
        if env_type == "comm_prey": # PSAF environment with moving prey
            env = environment_comm_prey.Environment(env_size)
        elif env_type == "no_comm_prey": # non communcating preds with moving prey
            env = environment_no_comm_prey.Environment(env_size)
        elif env_type == "comm": # PSAF environment with fixed prey
            env = environment_comm.Environment(env_size)
        else: # non communicating environment with fixed prey
            env = environment_no_comm.Environment(env_size)

        # TRAIN FOR 20.000 episodes
        max_steps2 = max_steps # to avoid reducing max_steps
        time_goal, b_ask1, b_give1 = run_multiple_episodes(n_episodes, env, max_steps2, epsilon, alpha)   

        # print training time of current RUN
        print("\nPROCESS ITERATION REQUIRED TIME: {} seconds, {} minutes".format(time.time()-start_time, (time.time()-start_time)/60))
        
        # append TG and budgets of each process run
        tg_list.append(time_goal)
        b_ask1_list.append(b_ask1)
        b_give1_list.append(b_give1)

    
    # end of processes runs
    print("\n\n\nTRAINING FINISHED")
    total_time = time.time() - start

    # TOTAL REQUIRED TIME
    print("\nTOTAL REQUIRED TIME: {} seconds, {} minutes".format(total_time, total_time/60))

    # perform averages over the various processes
    time_to_goal_final = [0 for i in range(int(n_episodes))] # intialize final list of TG
    b_ask1_final = [0 for i in range(int(n_episodes))] # intialize final list of b_ask1
    b_give1_final = [0 for i in range(int(n_episodes))] # intialize final list of b_give1

    # iterate over episodes
    for i in range(int(n_episodes)): 
        # iterate over processes
        for j in range(n_processes):
            # perform SUMS
            time_to_goal_final[i] += tg_list[j][i]
            b_ask1_final[i] += b_ask1_list[j][i]
            b_give1_final[i] += b_give1_list[j][i]  

    # now divide to obtain averages over processes' runs
    time_to_goal_final = [time_to_goal_final[i]/n_processes for i in range(int(n_episodes))]
    b_ask1_final = [b_ask1_final[i]/n_processes for i in range(int(n_episodes))]
    b_give1_final = [b_give1_final[i]/n_processes for i in range(int(n_episodes))]

    return time_to_goal_final, b_ask1_final, b_give1_final
    


# ---------------- PLOTTING FUNCTIONS -----------------#

def plot_time_to_goal(model_name, n_episodes, time_goal, avg = 0 ):

    if avg: # plot avg every 100 episodes
        averages = []
        for i in range(int(n_episodes/100)):
            averages.append(np.mean(time_goal[i*100: i*100 + 100]))   
        plt.plot(range(int(n_episodes/100)), averages)
        plt.title("Time To Goal averaged every 100 episodes_" + model_name)
        plt.xlabel("Training Episodes (x 100)")
        plt.ylabel("Time To Goal")
        plt.tight_layout()
        plt.savefig(model_name + "_avgTG")
    else: # plot TG over the episodes
        plt.plot(range(n_episodes), time_goal)
        plt.title("Time To Goal over the episodes_" + model_name)
        plt.xlabel("Training Episodes")
        plt.ylabel("Time To Goal")
        plt.tight_layout()
        plt.savefig(model_name + "_TG")

# PLOT comparison of TGs
def plot_TG_comparison(model_name1, model_name2, list1, list2, n_episodes, avg=1):
    if avg:
        averages1 = []
        averages2 = []
        for i in range(int(n_episodes/100)):
            averages1.append(np.mean(list1[i*100: i*100 + 100]))
            averages2.append(np.mean(list2[i*100: i*100 + 100]))  
        plt.plot(range(int(n_episodes/100)), averages1, label=model_name1)
        plt.plot(range(int(n_episodes/100)), averages2, label=model_name2)
        plt.title("Time To Goal averaged every 100 episodes")
        plt.xlabel("Training Episodes (x100)")
        plt.ylabel("Time To Goal")
        plt.legend()
        plt.tight_layout()
        plt.savefig("comparison" + "_TG_" + model_name1 + "_" + model_name2)
    else:
        plt.plot(range(n_episodes), list1, label=model_name1)
        plt.plot(range(n_episodes), list2, label=model_name2)
        plt.title("Time To Goal over the episodes")
        plt.xlabel("Training Episodes")
        plt.ylabel("Time To Goal")
        plt.legend()
        plt.tight_layout()
        plt.savefig("comparison" + "_avgTG_" + model_name1 + "_" + model_name2)


def plot_budget(model_name, type_of_budget, n_episodes, budget_list):
    plt.plot(range(n_episodes), budget_list)
    plt.title(type_of_budget + " first predator")
    plt.xlabel("Training Episodes")
    plt.ylabel(type_of_budget)
    plt.tight_layout()
    plt.savefig(model_name + "_" + type_of_budget)
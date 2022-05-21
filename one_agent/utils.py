import numpy as np
import time
import os

# for plots
import matplotlib.pyplot as plt
plt.style.use('seaborn-darkgrid')
plt.rcParams["figure.figsize"] = (12, 6)

# import environmnents
import environment
import environment_prey

ACTION_TO_STRING = ("down", "up", "left", "right", "stay")

# auxiliary function to clear the output (for GRID VISUALIZATION)
def cls():
    os.system('cls' if os.name=='nt' else 'clear')


# ---------------- TRAINING FUNCTIONS -----------------#

# run many episodes
def run_multiple_episodes(n_episodes, env, max_steps, epsilon, alpha):
    '''
    Perform the training over the environment env for n_episodes.
    - max_steps defines the maximum number of steps of each episode (if goal not reached)
    '''

    # Time to goal of each episode
    time_goal = []

    start_time = time.time()
    print("STARTING TRAINING")

    for episode in range(n_episodes):
        
        # randomly initialize the state of the env
        env.reset(epsilon, alpha) 
        
        goal = 0 # 1 if goal state
        performed_steps = 0 # TG

        # run episode until reaching goal state or max_steps
        while ((goal != 1) and (performed_steps < max_steps)):

            goal, actions, prey_action = env.transition() # PERFROM 1 TRANSITION
            performed_steps += 1
        
        # add TG of each episode
        time_goal.append(performed_steps)

        # show training episodes
        if episode % 50 == 0:
            print(f"Completed {episode} / {n_episodes} episodes")

    end_time = time.time()

    print("\n\nTRAINING FINISHED") 
    print("\nTotal required time: {} seconds, {} minutes".format(end_time - start_time, (end_time - start_time)/60))

    # return TG list
    return time_goal

# TRAINING SIMULATION 
def run_training_simulation(n_episodes, env, max_steps, epsilon, alpha):
    '''
    Perform the training over the environment env for n_episodes.
    The function is exactly the same as the one above with same parameters,
    but the TRAINING process is shown in the GRID wiht moving agent and prey
    and some information about chosen actions, coordinates at each time step 
    of the training
    '''
    # Time to goal of each episode
    time_goal = []
    
    start_time = time.time()
    print("STARTING TRAINING")

    for episode in range(n_episodes):
        print("\n\nEpisode: {}".format(episode))

        #  randomly initialize the state of the env
        env.reset(epsilon, alpha)

        goal = 0 # 1 if goal state
        performed_steps = 0 # TG

        reward = "Initial configuration -->NO REWARD"
        actions = np.array(["No action"])
        prey_action = "No action"
        print("Now watch TRAINING proceed step-by-step")

        # run episode until reaching goal state or max_steps
        while ((goal != 1) and (performed_steps < max_steps)):

            # print useful info
            print(env)
            print("Episode: {}".format(episode))
            print("Timestep: {}".format(performed_steps))
            print("Predator X coordinates: {}".format(env.pred_locs[0]))
            print("Prey O coordinate: {}".format(env.prey_loc[0]))
            if actions[0] != "No action":
                print("Action chosen by X: {}".format(ACTION_TO_STRING[int(actions[0])]))
                if prey_action != -1:
                    print("Prey action: {}".format(ACTION_TO_STRING[prey_action]))
            print("Reward: {}".format(reward))
            reward, actions, prey_action = env.transition()
            goal = reward
            performed_steps += 1
            #time.sleep(0.15) # sleep to see plot
            cls() # clear previous output from terminal

        # Append TG of each episode
        time_goal.append(performed_steps)

        print(env)
        if goal:
            print("Goal state reached!")
        else:
            print("Max number of steps reached")

        print("Predator X coordinates: {}".format(env.pred_locs[0]))
        print("Prey O coordinate: {}".format(env.prey_loc[0]))
        print("Action chosen by X: {}".format(ACTION_TO_STRING[int(actions[0])]))
        if prey_action != -1:
            print("Prey action: {}".format(ACTION_TO_STRING[prey_action]))

        print("Reward: {}".format(reward))
        print("Performed steps: {}".format(performed_steps))

    end_time = time.time()
    print("\n\nTRAINING FINISHED") 
    print("\nTotal required time: {} seconds, {} minutes".format(end_time - start_time, (end_time - start_time)/60))

    return time_goal


# SIMULATION (with the learned POLICY)
def run_simulation(env):
    '''
    Runs a single episode showing the environment grid.
    It assumes to take as a parameter env an already trained model.
    The purpose is to show how the agent behaves after having learned a policy
    PARAMETERS:
    - env = environment with already trained agent
    '''
    reward = "Initial configuration -->NO REWARD"
    action = np.array(["No action"])
    prey_action = "No action"
    time_step = 0
    goal = 0
    print("\n\n\n\nNow watch environment proceed step-by-step")

    # zero alpha end epsilon to show the learned policy
    env.reset(epsilon=0, alpha=0)

    # perform 1 full episode
    while not goal:
        print(env)
        print("Timestep: {}".format(time_step))
        print("Predator X coordinates: {}".format(env.pred_locs[0]))
        print("Prey O coordinate: {}".format(env.prey_loc[0]))
        if action[0] != "No action":
            print("Predator action: {}".format(ACTION_TO_STRING[int(action[0])]))
            if prey_action != -1:
                print("Prey action: {}".format(ACTION_TO_STRING[prey_action]))

        print("Reward: {}".format(reward))
        reward, action, prey_action = env.transition()
        goal = reward
        time_step += 1
        input("Press enter to visualize next state")
    print(env)
    print("Goal state reached!")

    print("Predator X coordinates: {}".format(env.pred_locs[0]))
    print("Prey O coordinate: {}".format(env.prey_loc[0]))
    print("Predator action: {}".format(ACTION_TO_STRING[int(action[0])]))
    if prey_action != -1:
        print("Prey action: {}".format(ACTION_TO_STRING[prey_action]))

    print("Reward: {}".format(reward))
    print("Performed steps: {}".format(time_step))

# repeat process many times (200 runs in the paper)
def repeat_process(n_processes, n_episodes, max_steps, epsilon, alpha, with_prey, env_size):
    '''
    - Repeat TRAINING PROCESS n_processes times.
    - The function calls the previously defined run_multiple_episodes() for n_processes iterations
    - with_prey = 1 if environment with MOVING prey
    - env_size is the size of the grid (int number)
    - The other parameters as before
    '''
    # TG list of each process iteration
    tg_list = []

    start = time.time()
    print("STARTING TRAINING")
    
    # iterate over number of processes
    for i in range(n_processes):

        print("\n\n\nPROCESS ITERATION NUMBER: {}".format(i+1))
        #start_time = time.time()

        # at each process iteration we re-create the environment
        if with_prey == 1: # environment with moving prey
            env = environment_prey.Environment(env_size)
        
        else: #  environment with fixed prey
            env = environment.Environment(env_size)

        # TRAIN FOR 20.000 episodes
        max_steps2 = max_steps # to avoid reducing max_steps
        time_goal = run_multiple_episodes(n_episodes, env, max_steps2, epsilon, alpha)   
        
        # append TG and budgets of each process run
        tg_list.append(time_goal)

    
    # end of processes runs
    print("\n\n\nTRAINING FINISHED")
    total_time = time.time() - start

    # TOTAL REQUIRED TIME
    print("\nTOTAL REQUIRED TIME: {} seconds, {} minutes".format(total_time, total_time/60))

    # perform averages over the various processes
    time_to_goal_final = [0 for i in range(int(n_episodes))] # intialize final list of TG

    # iterate over episodes
    for i in range(int(n_episodes)): 
        # iterate over processes
        for j in range(n_processes):
            # perform SUMS
            time_to_goal_final[i] += tg_list[j][i]  

    # now divide to obtain averages over processes' runs
    time_to_goal_final = [time_to_goal_final[i]/n_processes for i in range(int(n_episodes))]

    return time_to_goal_final
    


# ---------------- PLOTTING FUNCTIONS -----------------#

def plot_time_to_goal(model_name, n_episodes, time_goal, avg = 0 ):
    '''
    Plot TG over the episodes, or the TG averaged every 100 episods
    PARAMETERS:
    - model_name = name of the model to analyze
    - n_episodes = number of training episodes to show in the plot
    - time_goal = TG list
    - avg = 1 for plotting averaged TG every 100 episodes
    '''
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
    '''
    Compare TG lists of 2 different models
    Parameter avg as before
    '''
    if avg: # plot avg every 100 episodes
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
    else: # plot TG over the episodes
        plt.plot(range(n_episodes), list1, label=model_name1)
        plt.plot(range(n_episodes), list2, label=model_name2)
        plt.title("Time To Goal over the episodes")
        plt.xlabel("Training Episodes")
        plt.ylabel("Time To Goal")
        plt.legend()
        plt.tight_layout()
        plt.savefig("comparison" + "_avgTG_" + model_name1 + "_" + model_name2)





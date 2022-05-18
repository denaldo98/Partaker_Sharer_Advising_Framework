import numpy as np
import time
import matplotlib.pyplot as plt
plt.style.use('seaborn-darkgrid')
plt.rcParams["figure.figsize"] = (12, 6)

ACTION_TO_STRING = ("down", "up", "left", "right", "stay")

# ---------------- TRAINING FUNCTIONS -----------------#

# run many episodes
def run_multiple_episodes(n_episodes, env, max_steps, epsilon, alpha):

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

        # reduce epsilon
        #epsilon = min_epsilon + (max_epsilon - min_epsilon) * np.exp(-epsilon_decay_rate * episode)
        #epsilon_list.append(epsilon)

        # reduce alpha

        if episode % 50 == 0:
            print(f"Completed {episode} / {n_episodes} episodes")

    end_time = time.time()

    print("\n\nTRAINING FINISHED") 
    print("\nTotal required time: {} seconds, {} minutes".format(end_time - start_time, (end_time - start_time)/60))

    # return TG list
    return time_goal


# SIMULATION (with the learned Q-values)
def run_simulation(env):
    
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
        print("Predator coordinates: {}".format(env.pred_locs[0]))
        print("Prey coordinate: {}".format(env.prey_loc[0]))
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
    print("Performed steps: {}".format(time_step))



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





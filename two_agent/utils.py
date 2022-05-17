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

        # reduce epsilon
        #epsilon = min_epsilon + (max_epsilon - min_epsilon) * np.exp(-epsilon_decay_rate * episode)
        #epsilon_list.append(epsilon) 

        # reduce alpha

        if episode % 50 == 0:
            print(f"Completed {episode} / {n_episodes} episodes")
        
    end_time = time.time()

    print("\n\nTRAINING FINISHED") 
    print("\nTotal required time: {} seconds, {} minutes".format(end_time - start_time, (end_time - start_time)/60))

    return time_goal, b_ask1_list, b_give1_list


# SIMULATION (with the learned Q-values)
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
        print("Prey coordinate: {}".format(env.prey_loc[0]))
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



# ---------------- PLOTTING FUNCTIONS -----------------#

def plot_time_to_goal(model_name, n_episodes, time_goal, avg = 0 ):

    if avg: # plot avg every 100 episodes
        averages = []
        for i in range(int(n_episodes/100)):
            averages.append(np.mean(time_goal[i*100: i*100 + 100]))   
        plt.plot(range(int(n_episodes/100)), averages)
        plt.title("Time To Goal averaged every 100 episodes" + model_name)
        plt.xlabel("Training Episodes (x 100)")
        plt.ylabel("Time To Goal")
        plt.tight_layout()
        plt.savefig(model_name + "_avgTG")
    else: # plot TG over the episodes
        plt.plot(range(n_episodes), time_goal)
        plt.title("Time To Goal over the episodes" + model_name)
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
    plt.title(type_of_budget + "first predator")
    plt.xlabel("Training Episodes")
    plt.ylabel(type_of_budget)
    plt.tight_layout()
    plt.savefig(model_name + "_" + type_of_budget)
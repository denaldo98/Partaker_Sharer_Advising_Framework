import numpy as np
import matplotlib.pyplot as plt
plt.style.use('seaborn-darkgrid')
plt.rcParams["figure.figsize"] = (12, 6)

def plot_time_to_goal(n_episodes, time_goal, avg = 0):

    if avg: # plot avg every 100 episodes
        averages = []
        for i in range(int(n_episodes/100)):
            averages.append(np.mean(time_goal[i*100: i*100 + 100]))   
        plt.plot(range(int(n_episodes/100)), averages)
        plt.title("Time To Goal averaged every 100 episodes")
        plt.xlabel("Training Episodes (x 100)")
        plt.ylabel("Time To Goal")
        #plt.ylim([0, 600])
        plt.tight_layout()
        plt.show()

    else: # plot TG over the episodes
        plt.plot(range(n_episodes), time_goal)
        plt.title("Time To Goal over the episodes")
        plt.xlabel("Training Episodes")
        plt.ylabel("Time To Goal")
        #plt.ylim([0, 600])
        plt.tight_layout()
        plt.show()
    
def plot_b_ask(n_episodes, b_ask_list):
    plt.plot(range(n_episodes), b_ask_list)
    plt.title("b_ask 1st predator")
    plt.xlabel("Training Episodes")
    plt.ylabel("b_ask")
    #plt.ylim([0, 600])
    plt.tight_layout()
    plt.show()
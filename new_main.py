import matplotlib
import environment
import time
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('seaborn-darkgrid')
plt.rcParams["figure.figsize"] = (12, 6)



# define grid size
N = 4
 
# create environment by passing the grid size
env = environment.Environment(N)


# TRAINING PROCEDURE

# num of TRAINING episodes
n_episodes = 100

# max steps of each episode
max_steps = 5000

# Time to goal
time_goal = []



# TRAINING LOOP
print("STARTING TRAINING")

start_time = time.time()
for episode in range(n_episodes):
    
    env.reset() # randomly initialize the state of the env
    
    goal = 0 # encodes the reaching of a goal state (1 if goal state)

    performed_steps = 0

    # run episode until reaching goal state or max_steps
    while(not goal) and (performed_steps < max_steps):
        goal, _ = env.transition() # returns 1 if goal state is reached
        performed_steps +=1
    
    # at the end of each episode we add to the list the time to goal
    time_goal.append(performed_steps)

    if episode % 20 == 0:
        print(f"Completed {episode} / {n_episodes} episodes")

end_time = time.time()
print("TRAINING FINISHED") 
print("\nTotal required time: {} seconds, {} minutes".format(end_time - start_time, (end_time - start_time)/60))



# PLOT THE TIME TO GOAL
print("Plot of the Time To Goal over the episodes")
plt.plot(range(n_episodes), time_goal)
plt.title("Time To Goal over the episodes")
plt.xlabel("Training Episodes")
plt.ylabel("Time To Goal")
plt.tight_layout()
plt.show()


# SIMULATION (with the learned Q-values)
reward = "Initial configuration: NO REWARD"
action = np.array(["No action", "No action"])
time_step = 0
goal = 0
print("Now watch environment proceed step-by-step")
while not goal:
    print(env)
    print("Timestep: {}".format(time_step))
    print("Predators and prey coordinates: {}".format(env.get_positions()))
    print("Action chosen by first predator: {}".format(action[0]))
    print("Action chosen by second predator: {}".format(action[1]))
    print("Reward: {}".format(reward))
    

    reward, action = env.transition()
    goal = reward
    time_step += 1
    input("Press enter to visualize next state")












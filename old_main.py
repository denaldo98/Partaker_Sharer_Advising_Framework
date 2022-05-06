import numpy as np
import random
import environment
import utils


Action = int
Pair = tuple[int, int]


# define grid size
N = 4

# create environment by passing the grid size
env = environment.Environment(N)

# initialize q_tables (with zeros)
q_table1 = np.zeros([env.num_states, env.num_actions]) # for the 1st predator
q_table2 = np.zeros([env.num_states, env.num_actions]) # for the 2nd predator

# TRAINING PROCEDURE

# define Q-Learning PARAMETERS
alpha = 0.1 # decreasing
gamma = 0.9 # fixed
epsilon = 1
max_epsilon = 1
min_epsilon = 0.001
epsilon_decay_rate = 0.001

# num of TRAINING episodes
n_episodes = 2

# max steps of each episode
max_steps = 5000 

# communication parameters
b_ask1 = b_give1 = 3000
b_ask2 = b_give2 = 3000
v_p = 0.7

# number of visits of each state
n_visits1 = np.zeros(env.num_states)
n_visits2 = np.zeros(env.num_states)

# visits of each state-action pair
n_visits_sa1 = np.zeros([env.num_states, env.num_actions])
n_visits_sa2 = np.zeros([env.num_states, env.num_actions])

# Time to goal
time_goal = []



# TRAINING LOOP
for episode in range(n_episodes): # we perform n_episodes

    env.reset() # initialize environment state
    
    goal = 0 # encodes the reaching of a goal state (1 if goal state)
    
    performed_steps = 0

    # run episode until reaching goal state or max_steps
    while (not goal) and (performed_steps < max_steps):
        

        # choose action with epsilon-greedy approach
        action1 = epsilon_greedy(env.action_space, epsilon, q_table1, state1) # for agent 1
        action2 = epsilon_greedy(env.action_space, epsilon, q_table2, state2) # for agent 2

        # reduce epsilom
        epsilon = reduce_epsilon(episode + 1)

        # perform action
        next_state1, next_state2, reward, goal = env.step(action1, action2) # perform actions on the environment
        
        # APPLY Q-LEARNING UPDATE FORMULA
        q_table1[state1, action1] = q_learning_update(q_table1[state1, action1], np.max(q_table1[next_state1], reward)
        q_table2[state2, action2] = q_learning_update(q_table2[state2, action2], np.max(q_table2[next_state2], reward)

        state1, state2 = next_state1, next_state2 # update state

        performed_steps += 1
    
    # at the end of each episode we add to the list the time to goal
    time_goal.append(performed_steps) 

print("Training finished!")


def q_learning_update(old_value, next_max, reward):
    new_value = (1 - alpha) * old_value + alpha * (reward + gamma * next_max) # Q-learning update
    return new_value # update Q-table with new value





# auxiliary function for choosing the next action by means of epsilon-greedy 
def epsilon_greedy(action_space, epsilon, q_table, state):
    # generate random number
    rand = random.uniform(0, 1)
    if rand < epsilon:
        action = random.choice(action_space) # choose random action inside the action space
    else:
        action = np.argmax(q_table[state]) # best action in current state
    return action


def reduce_alpha(alpha):
    return 0


# AUXILIARY FUNCTIONS FOR THE COMMUNICATION MODEL
def prob_ask(state, n_visits):
    expl_degree = np.sqrt(n_visits[state])
    return pow((1 + v_p), -expl_degree)

def ask_q_values(state, n_visits, b_ask):
    if b_ask > 0:
        prob_ask = prob_ask(state, n_visits) # compute p_ask
        prob = random.uniform(0, 1)
        if prob < prob_ask:
            shared = share_q_value(state)









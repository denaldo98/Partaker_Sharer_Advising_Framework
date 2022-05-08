import random
import numpy as np

# set seed for reproducible experiments
#random.seed(1)

# Q-LEARNING PARAMETERS

gamma = 0.9 # fixed


# ACTIONS
ACTIONS = (0, 1, 2, 3, 4)
ACTION_TO_STRING = ("down", "up", "left", "right", "stay")
ACTION_TO_PAIR = ((1, 0), (-1, 0), (0, -1), (0, 1), (0, 0))


class Environment:
    
    def __init__(self, N):
        self.N = N
        self.size = (N, N) # grid dimensions

        # create predators/agents
        self.preds = [Agent() for _ in range(2)]

    # auxiliary function returning a random pair of coordinates
    def randpair(self):
        return (random.randrange(self.N), random.randrange(self.N))
    
    # Randomly set the coordinates
    def reset(self, epsilon, alpha):
        self.epsilon = epsilon
        self.alpha = alpha
        self.pred_locs = tuple([self.randpair() for _ in range(2)]) # each el is a tuple with the coordinates 
        self.prey_loc = tuple([self.randpair()]) # single tuple with prey coordinates

        # check bad initialization
        while (self.pred_locs[0] == self.prey_loc[0]) and (self.pred_locs[1] == self.prey_loc[0]):
            self.pred_locs = tuple([self.randpair() for _ in range(2)]) 
            self.prey_loc = tuple([self.randpair()]) 


    # perform a transition 
    def transition(self):
        # get the state index
        h = self.hash()

        # get predator's locations
        pred_locs = list(self.pred_locs)

        chosen_actions = np.zeros(2, dtype=int) # chosen actions
        
        # iterate over the predators
        for i, pred in enumerate(self.preds):
    
            action = pred.select_action(h[i], self.epsilon) # predator selects action in current state
            chosen_actions[i] = int(action)

            pred_locs[i] = move(pred_locs[i], action, self.size) # perform transition to next state

        # update state
        self.pred_locs = tuple(pred_locs)

        # REWARD
        reward = self.calculate_reward()

        # Update Q-values
        self.update_Q(reward)

        # return reward and chosen
        return reward, chosen_actions
    
    def calculate_reward(self):
        if (self.pred_locs[0] == self.prey_loc[0]) and (self.pred_locs[1] == self.prey_loc[0]): # check on goal condition
            reward = 1 # goal
        else:
            reward = 0
        return reward

    # Q-LEARNING update
    def update_Q(self, reward):
        
        index_new_state = self.hash()
        for i, pred in enumerate(self.preds):
            old_value = pred.Q[pred.prev_state][pred.prev_action]

            if index_new_state[i] not in pred.Q.keys(): # first time in new state--> add it to Q-table
                pred.Q[index_new_state[i]] = [0.001 for a in ACTIONS]

            next_max = np.max(pred.Q[index_new_state[i]])

            # Q-learning update
            pred.Q[pred.prev_state][pred.prev_action] = old_value + self.alpha * (reward + gamma * next_max - old_value )
            

    def hash(self):
        '''
        Environment state (i.e. relative positions) can be hashed for use in agent's Q-table
        '''
        # relative distance between 1st and 2nd predators
        distance_to_pred = (self.pred_locs[0][0] - self.pred_locs[1][0], self.pred_locs[0][1] - self.pred_locs[1][1])
        # relative distance between 1st pred and the prey
        distance_to_prey = (self.pred_locs[0][0] - self.prey_loc[0][0], self.pred_locs[0][1] - self.prey_loc[0][1])
        rel_pos1 = (distance_to_pred, distance_to_prey)

        # relative distance between 2nd and 1st preds
        distance_to_pred = (self.pred_locs[1][0] - self.pred_locs[0][0], self.pred_locs[1][1] - self.pred_locs[0][1])
        # relative distance between 2nd pred and the prey
        distance_to_prey = (self.pred_locs[1][0] - self.prey_loc[0][0], self.pred_locs[1][1] - self.prey_loc[0][1])
        rel_pos2 = (distance_to_pred, distance_to_prey)
        
        return [hash(rel_pos1), hash(rel_pos2)]
    

    def __repr__(self) -> str:
        # display environment as a grid
        grid = [[" "] * self.size[1] for _ in range(self.size[0])]

        # first agent represented with an 'X'
        grid[self.pred_locs[0][0]][self.pred_locs[0][1]] = "X"

        # second agent represented with a 'Y
        grid[self.pred_locs[1][0]][self.pred_locs[1][1]] = "Y"

        # prey represented as an 'O'
        grid[self.prey_loc[0][0]][self.prey_loc[0][1]] = "O"

        # check  goal condition
        if (self.pred_locs[0] == self.prey_loc[0]) and (self.pred_locs[1] == self.prey_loc[0]): # check on goal condition
            grid[self.prey_loc[0][0]][self.prey_loc[0][1]] = "XYO"
        
        # both predators in same cell
        elif self.pred_locs[0] == self.pred_locs[1]:
            grid[self.pred_locs[0][0]][self.pred_locs[0][1]] = "XY"
        
        # 1st predator and prey in same cell
        elif self.pred_locs[0] == self.prey_loc[0]:
            grid[self.pred_locs[0][0]][self.pred_locs[0][1]] = "XO"

        # 2nd predator and prey in same cell
        elif self.pred_locs[1] == self.prey_loc[0]:
            grid[self.pred_locs[1][0]][self.pred_locs[1][1]] = "YO"



        

        # ADD NEW SYMBOLS FOR PREDS IN THE SAME CELL, OR FOR PRED AND PREY IN SAME CELL, OR ALL IN THE SAME CELL

        return (
            ("_" * self.size[1] * 2 + "\n")
            + "\n".join("|" + " ".join(row) + "|" for row in grid)
            + ("\n" + " ̅" * self.size[1] * 2 + "\n")
        )
    

# IQL agent
class Agent:

    def __init__(self):
        # initialize Q-table
        self.Q = {}
 
    # select action according to epsilon greedy approach
    def select_action(self, h, epsilon):
        if h not in self.Q.keys():# first time in the state
            self.Q[h] = [0.001 for a in ACTIONS]

        rand = random.uniform(0, 1)
        
        # epsilon-greedy policy
        if rand < epsilon:
            action = random.choice(ACTIONS)
        else:
            action = np.argmax(self.Q[h]) 

        # save initial state and chosen action    
        self.prev_action = action
        self.prev_state = h
        
        return action
    
        
    

# perform TRANSITION on the grid
def move(start, action, size):
    directions = ACTION_TO_PAIR[action] # convert action into pair
    final_position = start[0] + directions[0], start[1] + directions[1] # coordinates of next state

    # in case of forbidden action, stay in the same position
    if not (0 <= final_position[0] < size[0] and 0 <= final_position[1] < size[1]):
        return start

    # return new coordinates
    return final_position

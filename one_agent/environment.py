import random
import numpy as np

# set seed for reproducible experiments
random.seed(247)

# Q-LEARNING PARAMETERS (define in the main)
gamma = 0.9 


# ACTIONS
ACTIONS = (0, 1, 2, 3, 4)
ACTION_TO_STRING = ("down", "up", "left", "right", "stay")
ACTION_TO_PAIR = ((1, 0), (-1, 0), (0, -1), (0, 1), (0, 0))

# ----------------------------- ENVIRONMENT ---------------------------#
class Environment:
    
    def __init__(self, N):

        # grid dimension
        self.size = N 

        # create predators/agents
        self.preds = [Agent() for _ in range(1)]

    # auxiliary function returning a random pair of coordinates
    def randpair(self):
        return (random.randrange(self.size), random.randrange(self.size))
    
    # Randomly set the coordinates
    def reset(self, epsilon, alpha):
        self.epsilon = epsilon
        self.alpha = alpha
        self.pred_locs = tuple([self.randpair() for _ in range(1)]) # each el is a tuple with the coordinates 
        self.prey_loc = tuple([self.randpair()]) # single tuple with prey coordinates

        # check bad initialization
        while self.pred_locs[0] == self.prey_loc[0]:
            self.pred_locs = tuple([self.randpair() for _ in range(1)]) 
            self.prey_loc = tuple([self.randpair()]) 


    # perform a transition 
    def transition(self):

        # get the state index
        h = self.hash()

        # get predator's locations
        pred_locs = list(self.pred_locs)

        chosen_actions = np.zeros(1, dtype=int) 
        
        # iterate over the predators (only 1 in this case)
        for i, pred in enumerate(self.preds):
    
            # action selection
            action = pred.select_action(h[i], self.epsilon) 
            chosen_actions[i] = int(action)

            # perform transition to next state
            pred_locs[i] = move(pred_locs[i], action, self.size) 

        # update state
        self.pred_locs = tuple(pred_locs)

        # REWARD
        # GOAL CONDITION: predator and prey in same grid cell
        if self.pred_locs[0] == self.prey_loc[0]:
            reward = 1 # goal 
        else:
            reward = 0

        # Update Q-values
        self.update_Q(reward)

        # return reward and chosen action
        return reward, chosen_actions, -1 # -1 for prey action (fixed in this case)
    

    # Q-LEARNING update
    def update_Q(self, reward):
        '''
        Apply Q-learning update formula for both agents
        '''
        # get index of newly reached state after preds and prey transitions
        index_new_state = self.hash()

        # iterate over agents (only 1 in this case)
        for i, pred in enumerate(self.preds):

            # Q-value in previous (state, action) pair
            old_value = pred.Q[pred.prev_state][pred.prev_action]
            
            # check new state and add it to Q-table
            if index_new_state[i] not in pred.Q.keys(): 
                pred.Q[index_new_state[i]] = [0 for a in ACTIONS]

            # max Q-value in next state
            next_max = np.max(pred.Q[index_new_state[i]])

            # Q-learning update
            pred.Q[pred.prev_state][pred.prev_action] = old_value + self.alpha * (reward + gamma * next_max - old_value )
            

    def hash(self):
        '''
        Environment state (i.e. relative positions) can be hashed for use in agent's Q-table
        '''
        # relative distance between predator and prey (x, y)
        rel_pos = (self.pred_locs[0][0] - self.prey_loc[0][0], self.pred_locs[0][1] - self.prey_loc[0][1]) 

        return [hash(rel_pos)]
    

    def __repr__(self) -> str:
        # display environment as a grid
        grid = [[" "] * self.size for _ in range(self.size)]

        for i, pdl in enumerate(self.pred_locs):
            if i == 0:
                grid[pdl[0]][pdl[1]] = "X" # first agent represented with an 'X'
            if i == 1:
                grid[pdl[0]][pdl[1]] = "Y" # second agent represented with a 'Y'
        for prl in self.prey_loc:
            grid[prl[0]][prl[1]] = "O"
        

        # ADD NEW SYMBOLS FOR PREDS IN THE SAME CELL, OR FOR PRED AND PREY IN SAME CELL, OR ALL IN THE SAME CELL

        return (
            ("_" * self.size * 2 + "\n")
            + "\n".join("|" + " ".join(row) + "|" for row in grid)
            + ("\n" + " ̅" * self.size * 2 + "\n")
        )
    

# IQL agent
class Agent:

    def __init__(self):

        # initialize Q-table
        self.Q = {}
 
    # select action according to epsilon greedy approach
    def select_action(self, h, epsilon):

        # check new state
        if h not in self.Q.keys():

            # initialize Q-table in new state
            self.Q[h] = [0 for a in ACTIONS]

        rand = random.uniform(0, 1)
        
        # epsilon-greedy policy
        if rand < epsilon:
            action = random.choice(ACTIONS)
        else:
            action = np.argmax(self.Q[h]) 

       # update old state and action (for Q-learning update)   
        self.prev_action = action
        self.prev_state = h
        
        return action
    
        
# perform state transition
def move(start, action, size):

    # convert action into pair
    directions = ACTION_TO_PAIR[action] 

    # compute new coordinates
    final_position = start[0] + directions[0], start[1] + directions[1] # coordinates of next state

    # in case of forbidden action, stay in the same position
    if not (0 <= final_position[0] < size and 0 <= final_position[1] < size):
        return start

    # return new coordinates
    return final_position

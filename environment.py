import random
import numpy as np
from dataclasses import dataclass, field

# grid word environment
Pair = tuple[int, int]
Action = int

epsilon = 0.6
alpha = 0.1 # decreasing
gamma = 0.9 # fixed


ACTIONS = (0, 1, 2, 3, 4)
ACTION_TO_STRING = ("up", "down", "left", "right", "stay")
ACTION_TO_PAIR = ((-1, 0), (1, 0), (0, -1), (0, 1), (0, 0))

class Environment:
    
    # constructor
    def __init__(self, N):
        self.N = N
        self.size = (N, N) # grid dimensions

        self.num_states = N * N * N * N * N * N # each state is <x1, y1, x2, y2>
        self.num_actions = 5 

        # create predators
        self.preds = [Agent(self.num_states, self.num_actions) for _ in range(2)]


    # auxiliary function returning a random pair
    def randpair(self):
        return (random.randrange(self.N), random.randrange(self.N))
    
    # Initialize randomly the state of the environment 
    def reset(self):
        self.pred_locs = tuple([self.randpair() for _ in range(2)]) # each el is a tuple with the coordinates 
        self.prey_loc = tuple([self.randpair()]) # single tuple with prey coordinates

    # perform a transition on the environment
    def transition(self):
        # get the state index
        h = self.state_index()

        # get predator's location
        pred_locs = list(self.pred_locs)

        # iterate over the predators
        for i, pred in enumerate(self.preds):
            # predator selects action in current state
            action = pred.choose(h)
            #perform movement
            pred_locs[i] = move(pred_locs[i], action, self.size)
    
        # update state
        self.pred_locs = tuple(pred_locs)

        # current reward
        if self.pred_locs[0] == self.prey_loc and self.pred_locs[1] == self.prey_loc: # check on goal condition
            reward = 1 # goal reward
        else:
            reward = 0

        # now, update Q-values
        for i, pred in enumerate(self.preds):
            old_value =  pred.Q[pred.prev_state, pred.prev_action]
            index_new_state= self.state_index()
            next_max = np.max(pred.Q[index_new_state])
            # Q-learning update
            pred.Q[pred.prev_state, pred.prev_action] = (1 - alpha) * old_value + alpha * (reward + gamma * next_max) 

            return reward

    def hash(self):
        # Environment state can be hashed for use in agent's Q-table
        return hash((self.pred_locs, self.prey_loc))
    
    def state_index(self):
        return self.pred_locs[0][0]*self.pred_locs[0][1]*self.pred_locs[1][0]*self.pred_locs[1][1]*self.prey_loc[0][0]*self.prey_loc[0][1]



    
    

    def __repr__(self) -> str:
        # display environment as a grid
        grid = [[" "] * self.size[1] for _ in range(self.size[0])]
        for i, pdl in enumerate(self.pred_locs):
            if i == 0:
                grid[pdl[0]][pdl[1]] = "X"
            if i == 1:
                grid[pdl[0]][pdl[1]] = "Y"
        for prl in self.prey_loc:
            grid[prl[0]][prl[1]] = "O"

        return (
            ("_" * self.size[1] * 2 + "\n")
            + "\n".join("|" + " ".join(row) + "|" for row in grid)
            + ("\n" + " ̅" * self.size[1] * 2 + "\n")
        )




# simple Q-learning agent
class Agent:
    def __init__(self, states, actions):
    
        # initialize Q-table
        self.Q = np.zeros([states, actions])  
        
        # COMMUNICATION AGENT
        # initialize budget
        #self.b_ask = self.b_give = 3000
        
        # number of visits of each state
        #self.n_visits = np.zeros(states)

        # visits of each state-action pair
        #self.n_visits_sa = np.zeros([states, actions])
    
    # select action according to epsilon greedy approach
    def choose(self, state):
        rand = random.uniform(0, 1)
        if rand < epsilon:
            action = random.choice(ACTIONS) # choose random action inside the action space
        else:
            action = np.argmax(self.Q[state]) # best action in current state
        self.prev_action = action
        self.prev_state = state
        return action
        
    

# auxiliary function that moves the agent on the grid to a new position(coordinates)
def move(start, action, size):
    dir = ACTION_TO_PAIR[action]
    result = start[0] + dir[0], start[1] + dir[1]

    # in case of forbidden action, stay in the same position
    if not (0 <= result[0] < size[0] and 0 <= result[1] < size[1]):
        return start
    # return final position
    return result

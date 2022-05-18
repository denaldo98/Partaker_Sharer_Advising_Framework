# COMMUNICATION WITH FIXED PREY
import random
import numpy as np
import math

# set seed for reproducible experiments
random.seed(247)

# Q-LEARNING PARAMETERS (defined in the main)
gamma = 0.9 

# COMMUNICATION PARAMETERS
vp = 0.7

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
        self.preds = [Agent() for _ in range(2)]

    # auxiliary function returning a random pair of coordinates
    def randpair(self):
        return (random.randrange(self.size), random.randrange(self.size))
    
    # Randomly set the coordinates
    def reset(self, epsilon, alpha):
        self.epsilon = epsilon
        self.alpha = alpha
        self.pred_locs = tuple([self.randpair() for _ in range(2)]) # each el is a tuple with the coordinates 
        self.prey_loc = tuple([self.randpair()]) # single tuple with prey coordinates

        # check bad initialization
        #while (self.pred_locs[0] == self.prey_loc[0]) and (self.pred_locs[1] == self.prey_loc[0]):
        #    self.pred_locs = tuple([self.randpair() for _ in range(2)]) 
        #    self.prey_loc = tuple([self.randpair()]) 

    # perform a transition 
    def transition(self):
        # get the state index
        h = self.hash()

        # action selection
        chosen_actions = self.choose_actions(h, self.epsilon) # list of chosen actions

        # get predatos' locations
        pred_locs = list(self.pred_locs)

        # perform transitions to next state
        for i in range(2):
            pred_locs[i] = move(pred_locs[i], chosen_actions[i], self.size) 
        
        # update state
        self.pred_locs = tuple(pred_locs)

        # REWARD
        reward = self.calculate_reward()

        # Update Q-values
        self.update_Q(reward)

        # return reward, chosen actions (no action for prey) and budgets of 1st pred
        return reward, chosen_actions, -1, self.preds[0].b_ask, self.preds[0].b_give

    # agents select action
    def choose_actions(self, h, epsilon):
        
        # chosen actions
        actions = np.zeros(2, dtype=int)

        # initialize and update dictionaries
        for i in range(2):

            # first time in current state
            if h[i] not in self.preds[i].Q.keys():

                # initialize Q-table entry
                self.preds[i].Q[h[i]] = [0 for a in ACTIONS]

            # check on (state, action) visits
            if h[i] not in self.preds[i].sa_visits.keys():

                # initialize (state, action) visits
                self.preds[i].sa_visits[h[i]] = [0 for a in ACTIONS]
            
            # check on state visits
            if h[i] not in self.preds[i].s_visits.keys():
                # initialize state visits
                self.preds[i].s_visits[h[i]] = 1
            # already visited state
            else:
                # update state visits
                self.preds[i].s_visits[h[i]] += 1


        # -------------------------------------------------------- PSAF ALGORITHM ----------------------------------------------------- #

        # 1st AGENT

        # check budget
        if self.preds[0].b_ask > 0:
            
            # calculate probability to ask for Q-values
            p_ask = pow((1 + vp), -math.sqrt(self.preds[0].s_visits[h[0]]))
            rand = random.uniform(0, 1)

            if rand < p_ask:

                # other agent provides its max Q-value
                shared = self.preds[1].provide_q_value(h[0], self.preds[0].sa_visits[h[0]])

                # Q-value received
                if shared is not None:

                    # reuduce budget
                    self.preds[0].b_ask -= 1

                    # update my Q table with the provided value
                    self.preds[0].Q[h[0]][shared[0]] = shared[1]

                    # execute best action
                    action1 = np.argmax(self.preds[0].Q[h[0]])

                    # update old state and action (for Q-learning update)
                    self.preds[0].prev_action = action1
                    self.preds[0].prev_state = h[0]
                
                # not provided value
                else:
                    action1 = self.preds[0].select_action(h[0], epsilon)
            # no communication
            else:
                action1 = self.preds[0].select_action(h[0], epsilon)
        # no budget
        else:
            action1 = self.preds[0].select_action(h[0], epsilon)


        # same reasoning for 2nd agent
        if self.preds[1].b_ask > 0:

            # calculate probability to ask for Q-values
            p_ask = pow((1 + vp), -math.sqrt(self.preds[1].s_visits[h[1]]))
            rand = random.uniform(0, 1)

            if rand < p_ask:

                # other agent provides its max Q-value
                shared = self.preds[0].provide_q_value(h[1], self.preds[1].sa_visits[h[1]])

                # Q-value received
                if shared is not None:

                    # reduce budget
                    self.preds[1].b_ask -= 1

                    # update Q table with the provided value
                    self.preds[1].Q[h[1]][shared[0]] = shared[1]

                    # execute best action
                    action2 = np.argmax(self.preds[1].Q[h[1]])

                    # update old state and action (for Q-learning update)
                    self.preds[1].prev_action = action2
                    self.preds[1].prev_state = h[1]
                
                # not provided value
                else:
                    action2 = self.preds[1].select_action(h[1], epsilon)
            # no communication
            else:
                action2 = self.preds[1].select_action(h[1], epsilon)
        # no budget
        else:
            action2 = self.preds[1].select_action(h[1], epsilon)
        
        # update list of chosen actions
        actions[0] = int(action1)
        actions[1] = int(action2)

        # update (state, action) visits
        self.preds[0].sa_visits[h[0]][action1] += 1
        self.preds[1].sa_visits[h[1]][action2] += 1

        # return chosen actions
        return actions
    
    def calculate_reward(self):
        '''
        Calculate reward for cooperative agents:
        --> 1 in case of goal state (predators and prey in same grid cell)
        --> 0 otherwise
        '''
        # check goal condition
        if (self.pred_locs[0] == self.prey_loc[0]) and (self.pred_locs[1] == self.prey_loc[0]):
            reward = 1 
        else:
            reward = 0

        return reward

    # Q-LEARNING update
    def update_Q(self, reward):
        '''
        Apply Q-learning update formula for both agents
        '''
        index_new_state = self.hash()

        # iterate over agents
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
    
    # set budget to zero when performing the simulation
    def set_budget(self):
        for pred in self.preds:
            pred.b_ask = 0
            pred.b_give = 0
    
    #  plot grid
    def __repr__(self) -> str:
        # display environment as a grid
        grid = [[" "] * self.size for _ in range(self.size)]

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

        return (
            ("_" * self.size * 2 + "\n")
            + "\n".join("|" + " ".join(row) + "|" for row in grid)
            + ("\n" + " ̅" * self.size * 2 + "\n")
        )


class Agent:

    def __init__(self):

        # initialize Q-table
        self.Q = {}
        
        # visits in each state
        self.s_visits = {}

        # visits in each state-action pair 
        self.sa_visits = {}

        # budgets
        self.b_ask = 3000
        self.b_give = 3000
 
    # select action according to epsilon greedy approach
    def select_action(self, h, epsilon):

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
    
    # PSAF: provide q-value to other agent
    def provide_q_value(self, partaker_state, partaker_conf):

        # check budget
        if self.b_give > 0:

            # check new state
            if partaker_state in self.Q.keys() and partaker_state in self.sa_visits.keys():
                
                best_action = int(np.argmax(self.Q[partaker_state]))

                # discrimination function
                discr = 1 - 1 /(np.max(self.Q[partaker_state]) - np.min(self.Q[partaker_state]) + 1)

                # calculate sharer confidence
                sharer_conf = self.sa_visits[partaker_state][best_action] * discr

                # compare with partaker confidence
                if sharer_conf > partaker_conf[best_action]:
                    # reduce budget
                    self.b_give -= 1

                    # return action and Q-value
                    return [best_action, self.Q[partaker_state][best_action]]
        return None



    

# perform state transition
def move(start, action, size):

    # convert action into pair
    directions = ACTION_TO_PAIR[action] 

    
    final_position = start[0] + directions[0], start[1] + directions[1]

    # in case of forbidden action, stay in the same position
    if not (0 <= final_position[0] < size[0] and 0 <= final_position[1] < size[1]):
        return start

    # return new coordinates
    return final_position

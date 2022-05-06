# TRIALS
import numpy as np
import random
import environment
import utils

# define grid size
N = 4

# create environment by passing the grid size
env = environment.Environment(N)


# TRAINING PROCEDURE

# TRAIN JUST 1 EPISODE
print("STARTING TRAINING")
train_steps = 1000

env.reset() # randomly initialize agents positioninh

for i in range(train_steps):
    if i % (train_steps // 20) == 0:
            print(f"Completed {i} / {train_steps} steps")
    env.transition()

print("Now watch environment proceed step-by-step")
while True:
    print(env)
    #print(env.reward_of())
    env.transition()
    input("Press enter")












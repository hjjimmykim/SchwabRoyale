import numpy as np
import pickle
from collections import deque
import time
import datetime
import copy

# # --Graphics--
import matplotlib.pyplot as plt
import matplotlib.animation as manimation

# # --Machine Learning--
import torch
from torch import autograd
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
# # -Optimizer
import torch.optim as optim

# # --Custom (Schwabbed) Code--
from Schwab_brain import Agent
import KS_sim_funcs as Sim



# # -Simulation Parameters
max_turn = 2000 # Max number of turns per episode
#record_turn = int(max_turn/100)  # Record turn every record_turn turns
n_ep = 100        # Number of training episodes

# # -Agent Parameters
alpha = 0.01    # Learning rate
beta = 0.1      # Exploration Parameter
gamma = 0.9     # Discount Factor

glee = 1	# Reward per opened door
sc = 0.7        # Baseline smoothing constant



# # --Maps--
# Remember to modify input_dim in Schwab_brain (sorry 4 hardcoding, but am lazy potato)
map_1p = np.array([
    [-2,-4,-2,-2,-2],
    [-2,-1,-1,-1,-2],
    [-2,-1,-1,-1,-2],
    [-3,-1,-1,-1,-2],
    [-2,-2,-2,-2,-2]])
'''
map_1p = np.array([
    [-2,-2,-2,-2,-2,-2,-2,-2,-2,-2,-2],
    [-2,-1,-1,-1,-1,-2,-1,-1,-1,-1,-2],
    [-2,-1,-1,-1,-1,-1,-1,-1,-1,-1,-2],
    [-2,-1,-1,-1,-1,-2,-1,-1,-1,-1,-2],
    [-2,-1,-1,-1,-1,-2,-1,-1,-1,-1,-4],
    [-2,-1,-1,-1,-1,-2,-1,-1,-1,-1,-2],
    [-2,-2,-2,-3,-2,-2,-2,-2,-2,-2,-2]])

map_2p = np.array([
    [-2,-2,-2,-2,-2,-2,-2,-2,-2,-2,-2],
    [-2,-1,-1,-1,-1,-2,-1,-1,-1,-1,-2],
    [-2,-1,-1,-1,-1,-1,-1,-1,-1,-1,-2],
    [-2,-1,-1,-1,-1,-2,-1,-1,-1,-1,-2],
    [-6,-1,-1,-1,-1,-2,-1,-1,-1,-1,-4],
    [-2,-1,-1,-1,-1,-2,-1,-1,-1,-1,-2],
    [-2,-2,-2,-3,-2,-2,-2,-2,-5,-2,-2]])
'''


# # --1 Player Simulation--
p1, map = Sim.initialize_1p(map_1p, np.array([3,3]), 5*5)
turn_list = []
# map_list = [map_1p]

for i_ep in range(n_ep):	# Loop through games
    # stats
    # Reset map and action/probability lists
    map = Sim.reset_1p(p1, map_1p, np.array([3,3]))
    prob_list = []
    act_list = []

    # print('Trial', i_ep, 'started.')
    # For keeping track of time
    t_start = time.time()
    t1 = time.time()

    for turn in range(max_turn):
        turn_reward = -0.1

        state = p1.observe(map)    # State formation
        state = torch.from_numpy(state).float() # It's really integer valued though

      # Action selection
        probs = p1.PolNet(state)
        action = np.random.choice(np.arange(4),p=np.squeeze(probs.detach().numpy()))
        action = torch.from_numpy(np.array(action))
        dir = Sim.get_dir(action)	# Convert to direction
        # Log action taken and it's probability
        prob_list.append(probs[action])


      # # Take the action
        # Candidate target location
        target_loc = p1.loc + dir

        # Determine result
        target_ind = map[target_loc[0]][target_loc[1]]    # Object at target location

        if target_ind == -1:				# If target location is empty
            map[p1.loc[0],p1.loc[1]] = -1		# Previous location becomes empty
            map[target_loc[0],target_loc[1]] = p1.id    # Target location becomes occupied
            p1.loc = target_loc				# Update location

        elif target_ind == -3:
            p1.has_key = True
            map[target_loc[0],target_loc[1]] = -2    # Remove key
            #print("Picked up the key on turn", turn)

        elif target_ind == -4 and p1.has_key:
            turn_reward = glee
            # print("Task completed!")
            break

        # Update the cumulative rewards
        p1.reward += turn_reward
        # Immediate reward
        turn_reward = torch.from_numpy(np.array(turn_reward))

        '''
        # Time-keeping
        if (turn+1) % record_turn == 0:
            t2 = time.time()
            print("Runtime for turns ", turn-record_turn+1, '-', turn, ': ', t2-t1)
            t1 = t2
        '''
        # Game ended without conclusion
        if turn == max_turn-1:
            print("Trial did not finish.")

#        np.append(map_list, [map], axis = 0)
  
    runtime = time.time()-t_start
    turn_list.append(turn)
    print("Trial", i_ep, "ended on turn", turn, "Runtime:", runtime, "-----------------------")
    if not p1.tenure:
        loss = 0
        for t in range(turn):
            turn_update = p1.REINFORCE(prob_list[t], p1.reward, p1.baseline)
            loss += turn_update
        p1.optimizer.zero_grad()
        loss.backward()
        p1.optimizer.step()
        # Baseline update
        p1.baseline = sc * p1.baseline + (1-sc) * p1.reward 



# # ----Save Runtime Data
savePath = 'Save/'
save_time = datetime.datetime.today().strftime('%Y-%m-%d_%H:%M')
fileName = save_time + '.pkl'
with open(savePath + fileName, 'wb') as f:
    pickle.dump(turn_list, f)



# # ----Runtime Plot----
trl = np.arange(n_ep)
plt.plot(trl, turn_list)
plt.show()

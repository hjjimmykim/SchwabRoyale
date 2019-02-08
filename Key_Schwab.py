import numpy as np
import pickle
from collections import deque
import time
import copy

# # --Graphics--
import matplotlib.pyplot as plt
# from IPython import display # For animation

# # --Machine Learning--
import torch
from torch import autograd
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
# # -Optimizer
import torch.optim as optim

# # --Custom (Schwabbed) Code--
from Schwab_brain import Memory, Agent
import KS_sim_funcs as Sim



# # -Simulation Parameters
max_turn = 10000 # Max number of turns per episode
record_turn = int(max_turn/100)  # Record turn every record_turn turns
n_ep = 10       # Number of training episodes

# # -Agent Parameters, Schwab_brain.py has values for input_, output_, and hidden_dimensions, and batch_ and memory_size
target_copy_freq = 10   # Update target network every tcf turns

alpha = 0.01    # Learning rate
beta = 0.1      # Exploration Parameter
gamma = 0.9     # Discount Factor

glee = 1	# Reward per opened door



# # --Maps--
map_1p = np.array([
    [-2,-2,-2,-2,-2,-2,-2,-2,-2,-2,-2],
    [-2,-1,-1,-1,-1,-2,-1,-1,-1,-1,-2],
    [-2,-1,-1,-1,-1,-1,-1,-1,-1,-1,-2],
    [-2,-1,-1,-1,-1,-2,-1,-1,-1,-1,-2],
    [-2,-1,-1,-1,-1,-2,-1,-1,-1,-1,-4],
    [-2,-1,-1,-1,-1,-2,-1,-1,-1,-1,-2],
    [-2,-2,-2,-3,-2,-2,-2,-2,-2,-2,-2]])
'''
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
p1, map = Sim.initialize_1p(map_1p, np.array([3,1]), glee)

for i_ep in range(n_ep):	# Loop through games
    # stats
    # Reset map and team scores
    map = Sim.reset_1p(p1, map, np.array([3,1]))

    print('Trial', i_ep, 'started.')
    # For keeping track of time
    t_start = time.time()
    t1 = time.time()

    for turn in range(max_turn):

        # Target q network update
        if turn % target_copy_freq:
            p1.DQN_target = copy.deepcopy(p1.DQN)

        state = p1.observe(map)    # State formation
        state = torch.from_numpy(state).float() # It's really integer valued though

        # Action selection
        q_values = p1.DQN(state).detach().numpy()
        p_values = np.exp(beta * q_values)/np.sum(np.exp(beta * q_values)) # Boltzmann probabilities
        action = np.random.choice(np.arange(4),p=np.squeeze(p_values))
        action = torch.from_numpy(np.array(action))
        dir = Sim.get_dir(action)	# Convert to direction

    # # Take the action
        # Rewards resulting from this move
        turn_reward = 0

        # Candidate target location
        target_loc = p1.loc + dir

        # Determine result
        target_ind = map[target_loc[0]][target_loc[1]]    # Object at target location

        if target_ind == -1:                                # If target location is empty
            map[p1.loc[0],p1.loc[1]] = -1               # Previous location becomes empty
            map[target_loc[0],target_loc[1]] = p1.id    # Target location becomes occupied
            p1.loc = target_loc                           # Update location

        elif target_ind == -3:
            p1.has_key = True
            map[target_loc[0],target_loc[1]] = p1.id    # Remove key
            print("Picked up the key on turn", turn)

        elif target_ind == -4 and p1.has_key:
            turn_reward += glee
            print("Task completed!")
            print("Trial", i_ep, "ended on turn", turn)#, "-----------------------")
            break
                                                                                                                                                                                                                                                                                                                                                                                                                                                           
        # Update the cumulative rewards
        p1.reward += turn_reward
        # Immediate reward
        turn_reward = torch.from_numpy(np.array(turn_reward))

        # New state
        state_new = p1.observe(map)
        state_new = torch.from_numpy(state_new).float()

        # Add experience to memory: [s,a,r,s']
        p1.memory.add([state, action, turn_reward, state_new])

        # Update Q-function
        if not p1.tenure:
            batch = p1.memory.sample()
            actual_batch_size = len(batch)

            # Separate out s,a,r,s' from the batch which is a list of lists
            states = torch.cat([item[0] for item in batch], 0)
            actions = np.array([item[1] for item in batch])
            rewards = np.array([item[2] for item in batch])
            next_states = torch.cat([item[3] for item in batch], 0)

            # Reshape (-1 automatically determines the appropriate length)
            states = np.reshape(states,[actual_batch_size, -1])
            actions = np.reshape(actions,[actual_batch_size, -1])
            rewards = np.reshape(rewards,[actual_batch_size, -1])
            next_states = np.reshape(next_states,[actual_batch_size, -1])

            # Q-value of the next state
            Q_current = p1.DQN(states).gather(1, torch.from_numpy(actions))
            Q_next = p1.DQN_target(next_states)
            Q_next_max = Q_next.max(dim=1,keepdim=True)[0].detach()

            target_Qs = torch.from_numpy(rewards).float() + gamma * Q_next_max
            target_Qs = target_Qs.float()

            loss = F.smooth_l1_loss(Q_current, target_Qs)

            p1.optimizer.zero_grad()
            loss.backward()
            p1.optimizer.step()

        # Time-keeping
        if (turn+1) % record_turn == 0:
            t2 = time.time()
            print("Runtime for turns ", turn-record_turn+1, '-', turn, ': ', t2-t1)
            t1 = t2

        # Game ended without conclusion
        if turn == max_turn-1:
            print("Trial did not finish.")
  
    t_finish = time.time()
    print("Trial", i_ep, "runtime:", t_finish-t_start, "-----------------------")
    #print("Game", str(i_ep), "ended on turn", turn, "-----------------------")

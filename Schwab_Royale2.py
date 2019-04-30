import numpy as np
import pickle
#from collections import deque
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
import Royale_Agents
#import KS_sim_funcs as Sim



# # -Simulation Parameters
# Agent parameters
K = 2           # Number of teams
N = 8           # Number of agents per team
hp_start = 8    # Starting hp
action_size = 4 # Four cardinal directions

# Simulation parameters
M = 32    # Width of map
max_turn = 5000 # Maximum number of turns per episode
record_turn = int(max_turn/100) # Record time every record_turn turns
n_ep = 1  # Number of training episodes (games)

# Reinforcement learning parameters
alpha = 0.01 # Learning rate
beta = 0.1   # Exploitation parameter
gamma = 0.9  # Discount factor
sigma = 0.5  # Selfishness factor
Omega = N    # Final survivor bonus (should it be for team and/or self?)
sc = 0.7     # Baseline smoothing constant



# # Schwifty Functions
# Check if the given location is within bounds
def check_valid_loc(loc):
    return (np.all(0 <= loc) and np.all(loc < M))

# Find the agent at a given location
def find_agent(loc):
    for ind, agnt in agent_dict.items(): # Loop through all agents
        if np.all(agnt.loc == loc): # Return the index of agent at the location
            return ind
    
# Translates action space to a direction
def get_dir(action):
    # NESW
    dir_list = [np.array([-1,0]),np.array([0,1]),np.array([1,0]),np.array([0,-1])]
    return dir_list[action]

# Initialize map, agents, and team scores
def initialize():
    agent_dict = {}                            # Dictionary of all agents
    map = -np.ones([M,M],dtype=np.int)         # M x M map
#    team_scores = np.zeros(K, dtype=np.int)    # Team scores

    for k in range(1, K+1):    # Loop through teams
        for n in range(N):       # Loop through agents per team
            id = N*(k-1) + n                                   # Agent id
            loc = np.array([int(M/K)*(k-1)+int(M**2/(K*N))*int(n/M), max(int(M/N),1)*(n%M)])        # Starting location
            '''Divides rows among teams, then fills up rows (if there are fewer agents per team than map width, spaces the agents evenly over one row), and spreads filled up rows over team-zone'''
  
            agent_dict[id] = Royale_Agents.Agent(id, hp_start, loc, k) # Create agent object and put it in the dictionary
            map[loc[0],loc[1]] = k                              # Record it on the map by its team number
#            agent_dict[id].memory.wipe()                        # Reset memory
      
    return map, agent_dict

# Reset map and team scores, preserve agents
def reset(agent_dict):
    map = -np.ones([M,M],dtype=np.int)         # M x M map

    for k in range(1, K+1):   # Loop through teams
        for n in range(N):      # Loop through agents per team
            agnt = agent_dict[N*(k-1) + n] # Take agent

            loc = np.array([int(M/K)*(k-1)+int(M**2/(K*N))*int(n/M), max(int(M/N),1)*(n%M)])       # Spawn location
            agnt.loc = loc     # Respawn
            agnt.hp = hp_start # Revive
            map[loc[0],loc[1]] = agnt.team    # Record on the map
            
    return map

# Return queue of surviving agents (the order in which they will act)
def action_queue(agent_dict):
    queue = []
    for ind, agnt in agent_dict.items(): # Loop through all agents
        if agnt.alive():                   # Only include the living
            queue.append(ind)

    np.random.shuffle(queue)               # Random shuffle  
    return queue



# # --Tracking--
tracking = True
record_freq = 100 #Record data every record_freq turns
tracking_list = []
tracking_list_smooth = []



# # --The Simulation--
map, agent_dict = initialize()
map_list = [map]

for i_ep in range(n_ep): # Loop through games
    # Reset map and team scores
    map = reset(agent_dict)

print('Game', i_ep, 'started.')
# For keeping track of time
t_start = time.time()
t1 = time.time()
for turn in range(max_turn):
turn_queue = action_queue(agent_dict) # Order of taking actions this turn

# End of game (last man standing)
if len(turn_queue) == 1:
print("David", turn_queue[0], "is the lone survivor.")
print("Team Scores:", team_scores)
print("Game", i_ep, "ended on turn", turn-1)#, "-----------------------")
break

# Target q network update
if turn % target_copy_freq:
for ind, agnt in agent_dict.items():
agnt.DQN_target = copy.deepcopy(agnt.DQN)

# Loop through alive agents
for j in turn_queue:
# Agent object
current_agent = agent_dict[j]

# If dead in queue, pass turn.
if not current_agent.alive():
continue;

state = current_agent.observe(map)    # State formation
state = torch.from_numpy(state).float() # It's really integer valued though

# Action selection
q_values = current_agent.DQN(state).detach().numpy()
p_values = np.exp(beta * q_values)/np.sum(np.exp(beta * q_values)) # Boltzmann probabilities
action = np.random.choice(np.arange(4),p=np.squeeze(p_values))
action = torch.from_numpy(np.array(action))

dir = get_dir(action) # Convert to direction

# Take the action
personal_point,team_point = current_agent.act(dir, turn, quiet=True)
#-----Toggle quiet to see kill feed-----

# New state
state_new = current_agent.observe(map)
state_new = torch.from_numpy(state_new).float()

# Immediate reward
reward = sigma * personal_point + (1-sigma) * team_point
reward = torch.from_numpy(np.array(reward))

# Add experience to memory: [s,a,r,s']
current_agent.memory.add([state, action, reward, state_new])

# Update Q-function
if not current_agent.tenure:
batch = current_agent.memory.sample(batch_size)
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
Q_current = current_agent.DQN(states).gather(1, torch.from_numpy(actions))
Q_next = current_agent.DQN_target(next_states)
Q_next_max = Q_next.max(dim=1,keepdim=True)[0].detach()

target_Qs = rewards + gamma * Q_next_max
target_Qs = target_Qs.float()

loss = F.smooth_l1_loss(Q_current, target_Qs)

current_agent.optimizer.zero_grad()
loss.backward()
current_agent.optimizer.step()

# Graphics -----------------------------------------------------------------
#print("Game " + str(i_ep) + ", turn " + str(turn))
#plot_map(map)
#time.sleep(1)

#Record team health
turn_health = [np.zeros(K)]
turn_hp_dict = hp_dict()
for agnt, health in turn_hp_dict.items():
team = agent_dict[agnt].team
turn_health[0][team-1] += health
ep_health = np.concatenate((ep_health,turn_health))

# Time-keeping
if (turn+1) % record_turn == 0:
t2 = time.time()
print("Runtime for turns ", turn-record_turn+1, '-', turn, ': ', t2-t1)
t1 = t2

# Game ended without conclusion
if turn == max_turn-1:
print("Game did not finish.")

stats_dict[i_ep] = ep_health

t_finish = time.time()
print("Game", i_ep, "runtime:", t_finish-t_start, "-----------------------")
#print("Game", str(i_ep), "ended on turn", turn, "-----------------------")



'''
# # ----Save Runtime Data
savePath = 'Save/'
save_time = datetime.datetime.today().strftime('%Y-%m-%d_%H:%M')
pickle_name = save_time + '.pkl'
with open(savePath + pickle_name, 'wb') as f:
    pickle.dump(tracking_list, f)


# # ----Animation----
ffmpegWriter = manimation.writers['ffmpeg']
metadata = dict(title = 'Key Schwab', artist='CJ', comment='Visualization')
writer = ffmpegWriter(fps=15, metadata=metadata)

fig = plt.figure()
anima_name = save_time + '.mp4'
with writer.saving(fig, anima_name, len(map_list)):
    for i in range(len(map_list)):
        plt.clf() # Clear figure
        plt.imshow(map_list[i], interpolation='none',aspect='equal')
        writer.grab_frame()

print('Done Animating')
'''

# # ----Runtime Plot----
track_t = np.arange(n_ep)
plt.plot(track_t, tracking_list)

for i in range(int(n_ep/record_freq)):
    tracking_list_smooth.append( sum(tracking_list[ i*record_freq : (i+1)*record_freq ])/record_freq )
track_t_smooth = record_freq * (np.arange(n_ep/record_freq) + 1)
plt.plot(track_t_smooth, tracking_list_smooth)

plt.show()

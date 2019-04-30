import numpy as np
from collections import deque
import copy

# # --Machine Learning--
import torch
from torch import autograd
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
# # -Optimizer
import torch.optim as optim

# # Agent
class Net(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__() # Initialize superclass

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        # Layers
        self.ff1 = nn.Linear(input_dim, hidden_dim)
#        self.ff2 = nn.Linear(hidden_dim, hidden_dim)
        self.ff3 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = F.relu(self.ff1(x)) # Input -> 1st hidden
#        x = F.relu(self.ff2(x)) # 1st hidden -> 2nd hidden
        x = F.softmax(self.ff3(x), dim=0)         # 2nd hidden -> output, deprecation along dim=0 (there should only be one)
        return x

class Agent:
    def __init__(self, id, hp, loc, k, tenure=False, baseline=0, in_dim=M*M + 1, hid_dim=10, out_dim=action_size):
        self.id = id    # Agent id (in agent_dict)
        self.hp = hp    # Hit points
        self.loc = loc  # Location (r,c coordinate [r,c])
        self.team = k   # Team index (k = 1, ..., K), since 0 is self!
        self.tenure = tenure       # If you don't have tenure yet, you gotta learn (i.e. determines whether the agent is learning)

        self.reward = 0   # Keep track of personal reward
        self.baseline = baseline

        # Create brain
        input_dim =  # Map cells + hp

        self.PolNet = Net(in_dim, hid_dim, out_dim)
        self.optimizer = optim.Adam(self.PolNet.parameters())

    # Checks living status
    def alive(self):
        return self.hp > 0

    # State formation
    def observe(self, map):
        state = copy.deepcopy(map)
        state[self.loc[0]][self.loc[1]] = 0            # Own location = 0 on map
        state = np.reshape(state, [1,-1]).squeeze()    # Convert to 1D array
        state = np.hstack([state, self.hp])            # Append hp
        return state
  
    # Taking an action based on chosen direction
    def act(self, dir, turn, quiet):
        # dir should be an np.array
        # turn = just for displaying kill message
        # quiet = print kill message or not

        # Rewards resulting from this move
        turn_point = 0
        # Candidate target location
        target_loc = self.loc + dir

        # Check if target location is within bounds (make sure the agent cannot move into itself)
        if check_valid_loc(target_loc):
            target_ind = map[target_loc[0]][target_loc[1]]    # Object at target location

            if target_ind == -1:                              # If target location is empty
                map[self.loc[0],self.loc[1]] = -1               # Previous location becomes empty
                map[target_loc[0],target_loc[1]] = self.team    # Target location becomes occupied
                self.loc = target_loc                           # Update location
            else:                                             # If target location is occupied
                # Agent at target location
                target_id = find_agent(target_loc)
                target_agent = agent_dict[target_id]

                if target_agent.team != self.team:
                    target_agent.hp -= 1    # Enemies are damaged
                else:
                    target_agent.hp += 1    # Allies are healed

                turn_point -= 1             # Cost of acting on another agent
                if not target_agent.alive():
                    # Display kill log
                    if not quiet:
                        print(kill_log(self.id, self.team, target_id, target_agent.team, turn))

                    target_agent.loc = [-1,-1]                    # Move corpse to the underworld

                    map[self.loc[0],self.loc[1]] = -1             # Previous location becomes empty
                    map[target_loc[0],target_loc[1]] = self.team  # Target location becomes occupied
                    self.loc = target_loc                         # Update location
  
        # Update the cumulative rewards
        self.reward += turn_point
    
        # Return immediate rewards
        return turn_point


class Aggressor(Agent):
    def __init__(self, id, hp, loc, k, tenure=False, baseline=0, in_dim=M*M + 1, hid_dim=10, out_dim=action_size):
        super().__init__()

    def act(self, dir, turn, quiet):
        # dir should be an np.array
        # turn = just for displaying kill message
        # quiet = print kill message or not

        # Rewards resulting from this move
        turn_point = 0
        # Candidate target location
        target_loc = self.loc + dir

        # Check if target location is within bounds (make sure the agent cannot move into itself)
        if check_valid_loc(target_loc):
            target_ind = map[target_loc[0]][target_loc[1]]    # Object at target location

            if target_ind == -1:                              # If target location is empty
                map[self.loc[0],self.loc[1]] = -1               # Previous location becomes empty
                map[target_loc[0],target_loc[1]] = self.team    # Target location becomes occupied
                self.loc = target_loc                           # Update location
            else:                                             # If target location is occupied
                # Agent at target location
                target_id = find_agent(target_loc)
                target_agent = agent_dict[target_id]

                if target_agent.team != self.team:
                    target_agent.hp -= 1    # Enemies are damaged
                    turn_point += 10        # Only care about attacking, heavily rewarded
                else:
                    target_agent.hp += 1    # Allies are healed

                if not target_agent.alive():
                    # Display kill log
                    if not quiet:
                        print(kill_log(self.id, self.team, target_id, target_agent.team, turn))

                    target_agent.loc = [-1,-1]                    # Move corpse to the underworld

                    map[self.loc[0],self.loc[1]] = -1             # Previous location becomes empty
                    map[target_loc[0],target_loc[1]] = self.team  # Target location becomes occupied
                    self.loc = target_loc                         # Update location
  
        # Update the cumulative rewards
        self.reward += turn_point
    
        # Return immediate rewards
        return turn_point

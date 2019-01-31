import numpy as np
from collections import deque

# # --Machine Learning--
import torch
from torch import autograd
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
# # -Optimizer
import torch.optim as optim



class Net(nn.Module):
    def __init__(self, input_dim, hidden_dim = 100, output_dim = 4, batch_size = 128):
        # Fully-connected feedforward network with 2 hidden layers
        # input-dim = 1-D reshaped map
        # hidden_dim = number of units per hidden layer
        # output_dim = number of outputs (i.e. actions)
        # Initialize superclass
        super().__init__() # Apparently we can use this newer notation. Old notation was: super(Net, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.batch_size = batch_size

        # Layers
        self.ff1 = nn.Linear(input_dim, hidden_dim)
        self.ff2 = nn.Linear(hidden_dim, hidden_dim)
        self.ff3 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x, batch_size=128):
        x = F.relu(self.ff1(x)) # Input -> 1st hidden
        x = F.relu(self.ff2(x)) # 1st hidden -> 2nd hidden
        x = self.ff3(x)         # 2nd hidden -> output

        return x


class Memory:
    def __init__(self, max_size):
        self.max = max_size
        self.buffer = deque(maxlen = max_size)

    # Add entry. FIFO.
    def add(self, experience):
        self.buffer.append(experience)

    # Sample batch_size number of entries, without replacement. Return as a list.
    def sample(self, batch_size):
        buffer_size = len(self.buffer)

        index = np.random.choice(np.arange(buffer_size),
                size = min(batch_size,buffer_size),
                replace = False)
        return [self.buffer[i] for i in index]

    # Own additions
    def is_full(self):
        return len(self.buffer) == self.max

    def wipe(self):
        self.buffer.clear() # NB, does not affect max length

    def length(self):
        return len(self.buffer)


class Agent:
      def __init__(self, id, loc, k, memory_size, tenure):
          self.id = id    # Agent id (in agent_dict)
          self.loc = loc  # Location (r,c coordinate [r,c])
          self.memory_size = memory_size # Experience replay maximum capacity
          self.tenure = tenure       # If you don't have tenure yet, you gotta learn (i.e. determines whether the agent is learning)

          self.personal_reward = 0   # Keep track of personal reward

          # Create brain
          input_dim = M*M # Map cells
          output_dim = action_size

          self.DQN = Net(input_dim, hidden_dim, output_dim, batch_size) # Personal neural network
          self.DQN_target = Net(input_dim, hidden_dim, output_dim, batch_size)         # Target network

          if torch.cuda.is_available() and use_cuda:

              self.DQN = self.DQN.cuda()
              self.DQN_target = self.DQN_target.cuda()

          self.optimizer = optim.Adam(self.DQN.parameters())

          self.memory = Memory(memory_size) # Experience replay


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
                                                                                                                                                                                                                  personal_point = 0
                                                                                                                                                                                                                      team_point = 0
                                                                                                                                                                                                                          
                                                                                                                                                                                                                              target_loc = self.loc + dir # Candidate target location
                                                                                                                                                                                                                                  
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
                                                                                                                                                                                                                                                                                                                                      '''
                                                                                                                                                                                                                                                                                                                                              if not target_agent.alive(): # Target agent is already dead; this should not run
                                                                                                                                                                                                                                                                                                                                                        print("David", self.id, "teabagged David", target_id)
                                                                                                                                                                                                                                                                                                                                                                '''
                                                                                                                                                                                                                                                                                                                                                                        target_agent.hp -= 1                            # Deal damage

                                                                                                                                                                                                                                                                                                                                                                                if not target_agent.alive():                    # If target agent has been killed
                                                                                                                                                                                                                                                                                                                                                                                              # Display kill log
                                                                                                                                                                                                                                                                                                                                                                                                        if not quiet:
                                                                                                                                                                                                                                                                                                                                                                                                                        print(kill_log(self.id, self.team, target_id, target_agent.team, turn))

                                                                                                                                                                                                                                                                                                                                                                                                                                  target_agent.loc = [-1,-1]                    # Move corpse to the underworld

                                                                                                                                                                                                                                                                                                                                                                                                                                            map[self.loc[0],self.loc[1]] = -1             # Previous location becomes empty
                                                                                                                                                                                                                                                                                                                                                                                                                                                      map[target_loc[0],target_loc[1]] = self.team  # Target location becomes occupied
                                                                                                                                                                                                                                                                                                                                                                                                                                                                self.loc = target_loc                         # Update location
                                                                                                                                                                                                                                                                                                                                                                                                                                                                          
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    personal_point += 1             # Gain a point for self
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              if target_agent.team != self.team:
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              team_point += 1               # Gain a point for the team, if the target was from another team
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    # Update the cumulative rewards
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        self.personal_reward += personal_point
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            team_scores[self.team-1] += team_point
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    # Return immediate rewards
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        return personal_point, team_point

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



# # ----Brain Parameters----
input_dim = 5*5#7*11    # Map cells
output_dim = 4      # Action space
hidden_dim = 100
memory_size = 100
batch_size = 16



class Net(nn.Module):
    def __init__(self):# , input_dim, hidden_dim = 100, output_dim = 4, batch_size = 16):
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

    def forward(self, x, batch_size = 16):
        x = F.relu(self.ff1(x)) # Input -> 1st hidden
        x = F.relu(self.ff2(x)) # 1st hidden -> 2nd hidden
        x = self.ff3(x)         # 2nd hidden -> output
        return x



class Memory:
    def __init__(self, max_size = memory_size):
        self.max = max_size
        self.buffer = deque(maxlen = max_size)

    # Add entry. FIFO.
    def add(self, experience):
        self.buffer.append(experience)

    # Sample batch_size number of entries, without replacement. Return as a list.
    def sample(self):
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
    def __init__(self, id, loc, glee, tenure):
        self.id = id    # Agent id (in agent_dict)
        self.loc = loc  # Location (r,c coordinate [r,c])
#        self.memory_size = memory_size # Experience replay maximum capacity
        self.tenure = tenure       # If you don't have tenure yet, you gotta learn (i.e. determines whether the agent is learning)

        self.has_key = False
	
        self.reward = 0     # Keep track of reward
        self.glee = glee    # Number of points gained from opening A door

        # Create brain
        self.DQN = Net() # Personal neural network
        self.DQN_target = Net()         # Target network

        self.optimizer = optim.Adam(self.DQN.parameters())

        self.memory = Memory(memory_size) # Experience replay


    # State formation
    def observe(self,map):
        state = copy.deepcopy(map)
        state[self.loc[0]][self.loc[1]] = 0            # Own location = 0 on map
        state = np.reshape(state, [1,-1]).squeeze()    # Convert to 1D array
        return state

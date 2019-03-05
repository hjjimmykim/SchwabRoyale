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



class DQN(nn.Module):
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
        x = self.ff3(x)         # 2nd hidden -> output
        return x



class Memory:
    def __init__(self, memory_size, batch_size):
        self.max = memory_size
        self.buffer = deque(maxlen = memory_size)
        self.batch_size = batch_size

    # Add entry. FIFO.
    def add(self, experience):
        self.buffer.append(experience)

    # Sample batch_size number of entries, without replacement. Return as a list.
    def sample(self):
        buffer_size = len(self.buffer)

        index = np.random.choice(np.arange(buffer_size),
                size = min(self.batch_size, buffer_size),
                replace = False)
        return [self.buffer[i] for i in index]

    # Own additions
    def is_full(self):
        return len(self.buffer) == self.max

    def wipe(self):
        self.buffer.clear() # NB, does not affect max length

    def length(self):
        return len(self.buffer)


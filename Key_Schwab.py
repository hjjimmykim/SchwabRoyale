import numpy as np
import pickle
from collections import deque

# # --Graphics--
import matplotlib.pyplot as plt
from IPython import display # For animation

# # --Machine Learning--
import torch
from torch import autograd
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
# # -Optimizer
import torch.optim as optim

# # --Custom (Schwabbed) Code--
import Schwab_brain.py as Brain
import KS_sim_funcs.py as Sim



# # -Simulation Parameters
max_turn = 10000 # Max number of turns per episode
record_turn = int(max_turn/100)  # Record turn every record_turn turns
n_ep = 10       # Number of training episodes

# # -Agent Parameters, Schwab_brain has preprogrammed defaults, given below
# #Neural Net input parameters: (input_dim, hidden_dim = 100, output_dim = 4, batch_size = 12)
# batch_size = 16     # Batch size
# hidden_dim = 100    # Hidden Layer size
memory_size = 100   # Number of experiences agent can recall
target_copy_freq = 10   # Update target network every tcf turns

alpha = 0.01    # Learning rate
beta = 0.1      # Exploration Parameter
gamma = 0.9     # Discount Factor



# # --Maps--
map_1p = np.array([-2,-2,-2,-2,-2,-2,-2,-2,-2,-2,-2],
    [-2,-1,-1,-1,-1,-2,-1,-1,-1,-1,-2],
    [-2,-1,-1,-1,-1,-1,-1,-1,-1,-1,-2],
    [-2,-1,-1,-1,-1,-2,-1,-1,-1,-1,-2],
    [-2,-1,-1,-1,-1,-2,-1,-1,-1,-1,-4],
    [-2,-1,-1,-1,-1,-2,-1,-1,-1,-1,-2],
    [-2,-2,-2,-3,-2,-2,-2,-2,-2,-2,-2])

map_2p = np.array([-2,-2,-2,-2,-2,-2,-2,-2,-2,-2,-2],
    [-2,-1,-1,-1,-1,-2,-1,-1,-1,-1,-2],
    [-2,-1,-1,-1,-1,-1,-1,-1,-1,-1,-2],
    [-2,-1,-1,-1,-1,-2,-1,-1,-1,-1,-2],
    [-6,-1,-1,-1,-1,-2,-1,-1,-1,-1,-4],
    [-2,-1,-1,-1,-1,-2,-1,-1,-1,-1,-2],
    [-2,-2,-2,-3,-2,-2,-2,-2,-5,-2,-2])



# # --Simulation--
p1, map = Sim.initialize_1p(map_1p, np.array([3,1]), 1, memory_size)

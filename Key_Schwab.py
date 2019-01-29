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


# # -Simulation Parameters
M = 11          # Width of map
max_turn = 1000 # Max number of turns per episode
record_turn = int(max_turn/50)  # Record turn every record_turn turns
n_ep = 10       # Number of training episodes

# # -Agent Parameters
batch_size = 10     # Batch size
memory_size = 100   # Number of experiences agent can recall
hidden_dim = 10     # Hidden Layer size

target_copy_freq = 10   # Update target network every tcf turns

alpha = 0.01    # Learning rate
beta = 0.1      # Exploration Parameter
gamma = 0.9     # Discount Factor

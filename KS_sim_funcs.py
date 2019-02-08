from Schwab_brain import Agent
import numpy as np
import copy



# Translates action space to a direction
def get_dir(action):	# NESW
    dir_list = [np.array([-1,0]),np.array([0,1]),np.array([1,0]),np.array([0,-1])]
    return dir_list[action]


# Return queue of surviving agents (the order in which they will act)
def action_queue():
    return np.random.shuffle(np.array([1,2]))\


# Initialize map, agents, and team scores
def initialize_1p(map_in, spawn_loc, glee):
    p1 = Agent(1, spawn_loc, glee, False)  # Create agent object and put it in the dictionary
    p1.memory.wipe()                                    # Reset memory
    map = copy.deepcopy(map_in)
    map[spawn_loc[0],spawn_loc[1]] = 1                  # Record on the map
    return p1, map


# Reset map and team scores, preserve agents
def reset_1p(player, map_in, spawn_loc):
    player.loc = spawn_loc              # Respawn
    player.has_key = False
    map = copy.deepcopy(map_in)
    map[spawn_loc[0],spawn_loc[1]] = 1  # Record on the map
    return map


# # --Visualization--
def plot_map(map):
  # Set background
    map_plot = copy.deepcopy(map)
    map_plot[map_plot == -1] = 8 # Color the background
  
  # Show map
    plt.imshow(map_plot, cmap='Set1',interpolation='none',aspect='equal')

  # Current axis
    ax = plt.gca()

  # Set up grid
    ax.set_xticks(np.arange(-0.5,M,1));
    ax.set_yticks(np.arange(-0.5,M,1));

    ax.set_xticklabels([]);
    ax.set_yticklabels([]);

  # Show grid
    plt.grid(color='w')
  
  # Display
    display.display(plt.gcf())
    display.clear_output(wait=True)

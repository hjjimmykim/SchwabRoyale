# Translates action space to a direction
def get_dir(action):	# NESW
    dir_list = [np.array([-1,0]),np.array([0,1]),np.array([1,0]),np.array([0,-1])]
    return dir_list[action]


# Return queue of surviving agents (the order in which they will act)
def action_queue():
    return np.random.shuffle(np.array([1,2]))\


# Initialize map, agents, and team scores
def initialize():
    agent_dict = {}                            # Dictionary of all agents
    map = -np.ones([M,M],dtype=np.int)         # M x M map
  
  for k in range(1, K+1):    # Loop through teams
    for n in range(N):       # Loop through agents per team
      id = N*(k-1) + n                                   # Agent id
      loc = np.array([int(M/K)*(k-1)+int(M**2/(K*N))*int(n/M), max(int(M/N),1)*(n%M)])        # Starting location
      '''Divides rows among teams, then fills up rows (if there are fewer agents per team than map width,
      spaces the agents evenly over one row), and spreads filled up rows over team-zone
      Previous code: loc = np.array([int(M/K)*(k-1),int(M/N)*n])'''
  
      agent_dict[id] = Agent(id, hp_start, loc, k, memory_size, False) # Create agent object and put it in the dictionary
      map[loc[0],loc[1]] = k                              # Record it on the map by its team number
      agent_dict[id].memory.wipe()                        # Reset memory
      
  return map, agent_dict, team_scores


# Reset map and team scores, preserve agents
def reset(agent_dict):
  map = -np.ones([M,M],dtype=np.int)         # M x M map
  team_scores = np.zeros(K, dtype=np.int)

  for k in range(1, K+1):   # Loop through teams
    for n in range(N):      # Loop through agents per team
      agnt = agent_dict[N*(k-1) + n] # Take agent
      
      loc = np.array([int(M/K)*(k-1)+int(M**2/(K*N))*int(n/M), max(int(M/N),1)*(n%M)])       # Spawn location
      agnt.loc = loc     # Respawn
      agnt.hp = hp_start # Revive
      map[loc[0],loc[1]] = agnt.team    # Record on the map
            
  return map, team_scores


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

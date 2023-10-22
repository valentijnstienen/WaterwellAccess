# Model settings
"""------------ SETTINGS ------------"""
CASENAME = "CAP_1000" # 'UNCAP_3000'; 'CAP_1000 #TODO
REMOVE_DOMINATED = False #TODO
PATH_TO_POP =  'Results/population_points_WD_xy_loc_CLUSTERED.csv'#'Subclustering/df_pop_clustered_NEW_3000.csv' #'Subclustering/df_pop_clustered_NEW_1000.csv'; 'Results/population_points_WD_xy_loc_CLUSTERED.csv' #TODO
CLUSTERS = [1]#range(1,4) #list(range(0,887))[::-1] # defines how large the cluster is (0: largest cluster, 886: smallest cluster) #TODO CHECK! # "All" is solving without clusters 
GRID_DIST = 250 # Grid potential water well locations
# Objective
CHOICE = 'standardFLP' #maxcover or standardFLP #TODO
OBJECTIVE = "max_cover" #max_cover or min_number_wells
if OBJECTIVE == "max_cover": # Maximization
    MAX_NUMBER_WELLS = range(21,5000)#[1,10,20,50,100,250,500,1000]#np.linspace(0,100,15)#[20] # How many water wells allowed to install
elif OBJECTIVE == "min_number_wells": # Minimization
    MIN_COVER = [100] #%
# Optimization settings
MIP_GAP = 0.0001
# Plotting
PLOT_INFO = True 
PLOT_PARETO = False
buf_opt = 500     # range of coverage for optimization (in meters)
current_locations = []
"""----------------------------------"""
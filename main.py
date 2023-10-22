import pandas as pd
import numpy as np
import geopandas as gpd
import math
import os
import time
import gurobipy as gp

from shapely.geometry import Point, Polygon, MultiPoint
from shapely.ops import unary_union
from shapely import wkt
from math import sqrt

from optimization_models_2 import optimize_locations
from visualize import visualize_situation
from load_data import load_data, load_newwps

# Model settings
from SETTINGS import *

df_pop, waterwells = load_data(path_to_pop = PATH_TO_POP, path_to_current_wells = 'Data (from QGIS)/waterwells_points_WD_xy.csv')
print(df_pop)

#######################################################################################
########################## Solve 100% for each cluster separately #####################
#######################################################################################
runtimes = pd.DataFrame(columns = ['Cluster', 'MaxNumberWaterwells', 'Runtime'])
optimal_locations = pd.DataFrame(columns = ['x', 'y', 'Cluster', 'Current'])
amount_variables = pd.DataFrame(columns = ['Cluster', 'nr_pop', 'nr_wps', 'vars_used'])
optimal_coverage = pd.DataFrame(columns = ['Cluster', 'Grid_dist', 'MaxNumberWaterwells', 'Households covered', 'Coverage (%)', 'NewAdd'])
for c in CLUSTERS:
    print("------------------------------------------")
    print("Optimizing cluster:", c)
    try: OLD_C = df_pop.loc[np.where(df_pop.cluster_order==c)[0][0], 'cluster_OLD']
    except: OLD_C = c
    
    # c = 'All'
    new_wps, df_pop_temp = load_newwps(c, df_pop, waterwells, GRID_DIST, buf_opt = buf_opt, REALCLUSTER=OLD_C, remove_dominated = REMOVE_DOMINATED, save = True, CASENAME = CASENAME, METHOD=3)
    current_locations = []#list(np.where(new_wps.Current>0)[0])
    
    # If you want the number of variables info for a specific cluster (e.g., 'All')
    # amv = sum([len(lst) for lst in df_pop_temp.cov_pop_index])
    # print([c, len(df_pop_temp), len(new_wps), amv])
    start = time.time()
    if OBJECTIVE == "min_number_wells": X,_,df_final = optimize_locations(df_pop_temp, new_wps, current_locations, c, GRID_DIST = GRID_DIST, CHOICE = CHOICE, OBJECTIVE = OBJECTIVE, MAX_NUMBER_WELLS=None, MIN_COVER = MIN_COVER, MIP_GAP = MIP_GAP, PLOT_INFO= PLOT_INFO, PLOT_PARETO=PLOT_PARETO, CASENAME=CASENAME, printall=False)
    elif OBJECTIVE == 'max_cover': X,_,df_final = optimize_locations(df_pop_temp, new_wps, current_locations, c, GRID_DIST = GRID_DIST, CHOICE = CHOICE, OBJECTIVE = OBJECTIVE, MAX_NUMBER_WELLS=MAX_NUMBER_WELLS, MIN_COVER = None, MIP_GAP = MIP_GAP, PLOT_INFO= PLOT_INFO, PLOT_PARETO=PLOT_PARETO, CASENAME=CASENAME, printall=False)
    runtime = time.time() - start

    new_opt_locations = new_wps.loc[np.where(X>0)[0], ['x', 'y', 'Current']]
    new_opt_locations.reset_index(drop = True, inplace = True)
    new_opt_locations['Cluster'] = c

    # Save locations
    optimal_locations = pd.concat([optimal_locations, new_opt_locations], axis=0)
    optimal_coverage = pd.concat([optimal_coverage, df_final], axis = 0)
    amv = sum([len(lst) for lst in df_pop_temp.cov_pop_index])
    amount_variables.loc[len(amount_variables)] = [c, len(df_pop_temp), len(new_wps), amv]
    runtimes.loc[len(runtimes)] = [c, len(df_final), runtime]
        
    if True:
        if not os.path.exists("ParetoFronts"): os.makedirs("ParetoFronts")
        optimal_locations.reset_index(drop=True, inplace = True)
        optimal_locations.to_csv('ParetoFronts/'+CASENAME+'_'+str(MIP_GAP)+'_optimal_locations.csv', sep = ";")
        optimal_coverage.reset_index(drop=True, inplace = True)
        optimal_coverage.to_csv('ParetoFronts/'+CASENAME+'_'+str(MIP_GAP)+'_optimal_coverage.csv', sep = ';')
        amount_variables.to_csv('ParetoFronts/'+CASENAME+'_'+str(MIP_GAP)+'_amount_variables.csv', sep = ";")
        runtimes.to_csv('ParetoFronts/'+CASENAME+'_'+str(MIP_GAP)+'_runtimes.csv', sep = ';')

optimal_locations.reset_index(drop=True, inplace = True)
optimal_locations.to_csv('ParetoFronts/'+CASENAME+'_'+str(MIP_GAP)+'_optimal_locations.csv', sep = ";")
optimal_coverage.reset_index(drop=True, inplace = True)
optimal_coverage.to_csv('ParetoFronts/'+CASENAME+'_'+str(MIP_GAP)+'_optimal_coverage.csv', sep = ';')
amount_variables.to_csv('ParetoFronts/'+CASENAME+'_'+str(MIP_GAP)+'_amount_variables.csv', sep = ";")
runtimes.to_csv('ParetoFronts/'+CASENAME+'_'+str(MIP_GAP)+'_runtimes.csv', sep = ';')

#######################################################################################
CLUSTER=3
PATH_TO_POPULATION = 'Preprocessed data/Population data/C_POP_'+str(CLUSTER)+'.csv'
PATH_TO_CURRENT_WATERWELLS = None#'Data (from QGIS)/waterwells_points_WD_xy.csv'
PATH_TO_POTENTIAL_WATERWELLS = 'Preprocessed data/PotentialWaterPoints/C'+str(CLUSTER)+'.csv'
PATH_TO_OPTIMAL_WATERWELLS = optimal_locations#'TEST.csv'
"""-------------------------------------------------------------"""
#visualize_situation(CLUSTER, PATH_TO_POPULATION, PATH_TO_CURRENT_WATERWELLS, PATH_TO_POTENTIAL_WATERWELLS, PATH_TO_OPTIMAL_WATERWELLS)
#######################################################################################
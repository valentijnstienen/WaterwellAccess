print("_________________________________________________________________________________________________________________________________________________________________")
print("_________________________________________________________________________________________________________________________________________________________________")
import pandas as pd
import numpy as np
import geopandas as gpd
import math
import os
import time
import networkx as nx
import gurobipy as gp
import matplotlib.pyplot as plt
import progress
import random

from shapely.geometry import Point, Polygon, MultiPoint, LineString
from shapely import wkt
from math import sqrt
from scipy.spatial import cKDTree
from tqdm import tqdm
from gurobipy import *
from pandarallel import pandarallel
from progress.bar import Bar
from sklearn.cluster import DBSCAN
from collections import Counter
from scipy.spatial.distance import cdist

from load_data import load_data, load_newwps
from optimization_models_2 import optimize_locations
from visualize import visualize_situation

"""---- SETTINGS ----"""
CLUSTERS = range(0,887)#range(0,887)
AMOUNT_SUBCLUSTERS = None #2^SOMETHING
MAX_POP = 1000
"""------------------"""
REMOVE_DOMINATED = True
GRID_DIST = 250 #m
MAX_DIST = 500 #m
SOLV_SUBS = False
if SOLV_SUBS: 
    CHOICE = 'maxcover' #maxcover or standardFLP
    MAX_NUMBER_WELLS = range(1,5000)
    MIP_GAP = 0.0001
    # Plotting
    PLOT_INFO = False 
    PLOT_PARETO = False
    # Comparison df (with the optimal solutions)
    FNAME_COMPARE = 'ParetoFronts/250_DOMINANCE_REMOVED_True_1e-05_optimal_coverage.csv' #'WITHOUT_DOMINANCE_REMOVED_1000_optimal_coverage.csv'##3000_optimal_coverage
current_locations = [] #needed for 
"""------------------"""

# Define casename for the base case (no subclustering)
CASENAME = str(GRID_DIST) + "_" + "DOMINANCE_REMOVED_" + str(REMOVE_DOMINATED) # 250_WITHOUT_DOMINANCE_REMOVED
MAX_POP_OLD = MAX_POP

# Load the population data and the current waterwell data
df_pop, waterwells = load_data(path_to_pop = 'Results/population_points_WD_xy_loc_CLUSTERED.csv', path_to_current_wells = 'Data (from QGIS)/waterwells_points_WD_xy.csv')
columns_needed = ['X', 'Y', 'VALUE', 'State_En', 'Loc_En', 'cluster','cluster_order', 'geometry']#list(df_pop.columns)

#TODO DONE
def solve_selection_problem(column_values, threshold):
    """
        Solve the coverage problem if we have multiple clusters that are
        separated by a heuristic that does not guarantee optimality.
    
    Parameters
    ----------
    column_values : each list indicates the amount of households that can be 
                covered with the [index] amount of wells in each separate cluster.
        list of lists
    
    threshold : Maximum number of facilities that are allowed to be open over 
                all the different clusters. 
        float
    
    Returns
    -------
    model.objVal : objective value of the optimization problem (combined
                coverage (households))
        float
    """
    num_columns = len(column_values)
    
    model = gp.Model("number_selection")
    
    # Binary variables indicating whether a number is selected from each column
    selected_vars = {}
    t = max(len(inner_list) for inner_list in column_values)
    for i in range(num_columns):
        for j in range(t):
            selected_vars[i, j] = model.addVar(vtype=gp.GRB.BINARY)
    
    model.setObjective(gp.quicksum(selected_vars[i, j] * column_values[i][j] for i in range(num_columns) for j in range(len(column_values[i]))), sense=gp.GRB.MAXIMIZE)
    
    # Constraint: Sum of selected indices must be below the threshold
    model.addConstr(gp.quicksum((j+1) * selected_vars[i, j] for j in range(t) for i in range(num_columns)) <= threshold)
    
    # Constraint: Maximum of one number must be selected from each column
    for i in range(num_columns):
        model.addConstr(gp.quicksum(selected_vars[i, j] for j in range(t)) <= 1)
    
    model.setParam('OutputFlag',0)
    model.Params.mipgap = 0.000001
    
    model.optimize()
    print('Objective:', model.objVal)
    
    return model.objVal

def solve_selection_model_2(a, total_pop):
    m = Model("water_well_allocation")

    # Solve until this optimality gap
    MIP_GAP = 0.00001

    # Data (e.g., number of clusters and wells)
    matrix = a.values
    num_clusters = len(matrix)
    max_wells = len(matrix[0])

    # Create variables
    x = {}
    for i in range(num_clusters):
        for j in range(max_wells):
            x[i, j] = m.addVar(vtype=GRB.BINARY, name=f"x_{i}_{j}")

    # Set objective
    m.setObjective(sum(x[i, j] * matrix[i][j] for i in range(num_clusters) for j in range(max_wells)), GRB.MAXIMIZE)

    # Each cluster can have at most one configuration
    for i in range(num_clusters):
        m.addConstr(sum(x[i, j] for j in range(max_wells)) <= 1, f"Cluster_{i}")

    # Total number of wells constraint
    h = m.addConstr(sum(x[i, j] * (j + 1) for i in range(num_clusters) for j in range(max_wells)) <= 0, "Total_Wells")

    # Set parameters
    m.setParam('OutputFlag', 0) # Suppress output during optimization
    m.Params.mipgap = MIP_GAP
    ##########################################################################

    ##########################################################################
    ########### Loop over all possible budgets to get a complete PF ##########
    ##########################################################################
    obj_val_array = []
    for threshold in range(1,10000):
        start = time.time()
        print("Optimizing for", threshold, 'waterwells.')
        m.remove(h)
        h = m.addConstr(sum(x[i, j] * (j + 1) for i in range(num_clusters) for j in range(max_wells)) <= threshold, "Total_Wells")
        m.optimize()
        obj_val = m.objVal
        rt = time.time()-start

        coverage = np.round(100*(obj_val/total_pop), 2)
        obj_val_array.append([threshold, obj_val, coverage, rt])#, new_add])#,list(Xvalues),list(Yvalues)])
        if True:
            ############################################ Print output ############################################
            print("    Objective value:", obj_val, "households.")
            print("    Coverage:", coverage, "%")
            print("    Opening", threshold, "water wells.")
            print("    Runtime", rt, "seconds.")
            print("_____________________")
            ######################################################################################################
        if threshold%20 == 0:
            df_final = pd.DataFrame(obj_val_array, columns = ['MaxNumberWaterwells', "Households covered", "Coverage (%)", 'Runtime'])#, "NewAdd"])
            df_final.to_csv('df_final_'+CASENAME+'.csv', sep = ";")
        if coverage > 100-(MIP_GAP*100): break
    df_final = pd.DataFrame(obj_val_array, columns = ['MaxNumberWaterwells', "Households covered", "Coverage (%)", 'Runtime'])#, "NewAdd"])
    df_final.to_csv('df_final_'+CASENAME+'.csv', sep = ";")
    return df_final

#TODO DONE
def determine_optimality_guarantee_FACILITIES(df_pop_temp, new_wps, k=2):
    """
        Compute how far a clustered solution is from the optimal value. 
    
    Parameters
    ----------
    df_pop_temp : population data including indication to which 
            cluster a household is assigned
        dataframe

    new_wps : potential waterwell locations for this whole cluster
        dataframe
    
    k : [DEVELOPMENT] amount of subclusters used.
    
    Returns
    -------
    bound_facilities : the optimality for the amount of facilities when using 
             these two subclusters.
        float
    """
    
    """--------------------- Find potential affected households --------------------"""
    # To save computation time, we first only find households that may be affected by 
    # the split of the cluster. To do this, we first find subsubclusters taht are 
    # then take the 
    VISUALIZE_POTENTIAL_AFFECTED_HH = False
    # Create the subclusters using DBSCAN
    maxlabelsOLD = 0
    for i in range(k):
        # Run DBSCAN
        X = df_pop_temp.loc[np.where(df_pop_temp.cluster==i)[0], ['X', 'Y']].to_numpy()#.reset_index(drop=True)
        dbscan = DBSCAN(eps=1000, min_samples=1)
        dbscan.fit(X)
        labels = dbscan.labels_
        df_pop_temp.loc[np.where(df_pop_temp.cluster==i)[0],'cluster_2'] = labels+maxlabelsOLD
        maxlabelsOLD = maxlabelsOLD+max(labels)+1
    # Create the polygons that form the bands that contain the potentially affected households
    if VISUALIZE_POTENTIAL_AFFECTED_HH: 
        fig, ax = plt.subplots()
    buffered_polygons_POS, buffered_polygons_NEG = [],[]
    for c in df_pop_temp.cluster_2.unique():
        ########################################################################
        ######################### Filter the households ########################
        ########################################################################
        points_cluster = list(zip(df_pop_temp.loc[df_pop_temp.cluster_2==c,'X'], df_pop_temp.loc[df_pop_temp.cluster_2==c,'Y']))
        ########################################################################
        
        ########################################################################
        ################# Create the outer boundary (positive) #################
        ########################################################################
        cv = gpd.GeoSeries([Point(x, y) for x, y in points_cluster]).buffer(500).unary_union
        try: 
            buffered_polygon_pos = cv.exterior
            polygon_x, polygon_y = buffered_polygon_pos.xy
            if VISUALIZE_POTENTIAL_AFFECTED_HH: ax.plot(polygon_x, polygon_y, color='purple')
            buffered_polygons_POS.append(Polygon(buffered_polygon_pos))
        except: 
            for poly in cv.geoms:
                buffered_polygon_pos = poly.exterior
                # Extract coordinates for plotting the buffered polygon
                buffered_polygon_x, buffered_polygon_y = buffered_polygon_pos.xy
                # Plot the buffered polygon
                if VISUALIZE_POTENTIAL_AFFECTED_HH: ax.plot(buffered_polygon_x, buffered_polygon_y, color='purple')
                buffered_polygons_POS.append(Polygon(buffered_polygon_pos))
        ########################################################################
                
        ########################################################################
        ################# Create the inner boundary (negative) #################
        ########################################################################
        copied_polygon = Polygon(cv.exterior)
        buffered_polygon_neg = copied_polygon.buffer(-1000)
        try: 
            buffered_polygon_neg = buffered_polygon_neg.exterior
            polygon_x, polygon_y = buffered_polygon_neg.xy
            if VISUALIZE_POTENTIAL_AFFECTED_HH: ax.plot(polygon_x, polygon_y, color='green')
            buffered_polygons_NEG.append(Polygon(buffered_polygon_neg))
        except: 
            for poly in buffered_polygon_neg.geoms:
                buffered_polygon_neg = poly.exterior
                # Extract coordinates for plotting the buffered polygon
                buffered_polygon_x, buffered_polygon_y = buffered_polygon_neg.xy
                # Plot the buffered polygon
                if VISUALIZE_POTENTIAL_AFFECTED_HH: ax.plot(buffered_polygon_x, buffered_polygon_y, color='green')
                buffered_polygons_NEG.append(Polygon(buffered_polygon_neg))
        ########################################################################
    # Filter the households to only the potentially affected households
    def within_potential_bands(point):
        for buf in buffered_polygons_NEG:
            if buf.contains(point):
                return False
        return True
    points_within_polygon = df_pop_temp.geometry.apply(within_potential_bands)
    df_pop_temp_FULL = df_pop_temp[points_within_polygon].reset_index(drop=True)
    # Visualize the bands
    if VISUALIZE_POTENTIAL_AFFECTED_HH:
        ax.plot(df_pop_temp.X, df_pop_temp.Y, 'o', color='pink', label='households')
        ax.plot(df_pop_temp_FULL.X, df_pop_temp_FULL.Y, 'o', color='red', label='households')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_title('Polygon, Generated Points, and Buffered Polygon')
        ax.legend()
        plt.show()
    """-----------------------------------------------------------------------------"""
 
    """ --------------------------- Create the graph -------------------------------"""
    VISUALIZE_GRAPH = False
    # Having a graph, we can easily find the households that may be affected by the 
    # split of the cluster. 
    def find_covering_pop(geo_point, MAX_POINT, df_pop_temp):
        geo = geo_point.buffer(1000)
        a = df_pop_temp[df_pop_temp.geometry.within(geo)].copy()
        
        a['distance'] = a.geometry.distance(geo_point)
        a['angle'] = np.degrees(np.arctan2(a.geometry.y - geo_point.y, a.geometry.x - geo_point.x))+180
        a.sort_values(by=['distance', 'angle'], inplace=True, ascending = False)
        
        a = list(a.index)
        if MAX_POINT== 'all': return a#[1::]
        elif len(a)>MAX_POINT: return a[0:MAX_POINT]#return [a[i] for i in numbers[0:10]]# a[0:10]
        else: return a[0::]
    df_pop_temp_FULL['cov_pop_index_self_FULL'] = df_pop_temp_FULL.geometry.parallel_apply(find_covering_pop, args=('all', df_pop_temp_FULL))
    edges_FULL = []
    with Bar('Processing', max=len(df_pop_temp_FULL)) as bar:
        for i in df_pop_temp_FULL.index:
            for l in df_pop_temp_FULL.loc[i, 'cov_pop_index_self_FULL']:
            
                if ((l,i) in edges_FULL) | (i == l): continue
                
                else: edges_FULL.append((i, l))
            bar.next()
    #print("All edges:", len(edges_FULL))
    # Create the graph
    G_FULL = nx.Graph()
    G_FULL.add_nodes_from(range(0,len(df_pop_temp_FULL)))
    G_FULL.add_edges_from(edges_FULL)
    node_colors = df_pop_temp_FULL.color
    ################ Visualize the graph ################
    if VISUALIZE_GRAPH:
        pos = {node: (x, y) for node, x, y in zip(G_FULL.nodes, df_pop_temp_FULL.X, df_pop_temp_FULL.Y)}
        nx.draw(G_FULL, pos,  with_labels=False, node_size=50, edge_color='grey', node_color=node_colors)
        plt.show()
    #####################################################
    """-----------------------------------------------------------------------------"""
    
    """---------------------- Determine the real cut value --------------------------"""
    # Determine the real cut value, but also find all the cut edges. These are found 
    # to determine which people could be affected by the split.
    def extract_edges_with_different_colors(edges):
        edges_with_different_colors = []
        for e in edges:
            color_from = node_colors[e[0]]
            color_to = node_colors[e[1]]
            if color_from != color_to:
                edges_with_different_colors.append(e)
        return edges_with_different_colors
    cut_edges = extract_edges_with_different_colors(edges_FULL)
    # print('Bound 1:', len(cut_edges))
    """-----------------------------------------------------------------------------"""
    
    FACILITIES_BOUND = True
    if FACILITIES_BOUND:
        """--------------------- Give some optimality guarantee ------------------------""" 
        # Find all people that can be covered by a removed well
        affected_households_ind = list(set(item for sublist in cut_edges for item in sublist))
        affected_households = df_pop_temp_FULL.iloc[affected_households_ind, :]
    
        # Now, we could eliminate one well if we can combine two wells into one. Therefore, 
        # a guarntee is if we find the maximum amount of wells on one side, this is in the best
        # case what can be removed.

        ########## Solve for all orange households ##########
        affected_households_orange = affected_households[affected_households.color == 'orange'].reset_index(drop=True)
        if len(affected_households_orange)==0: return 0
        potential_waterwells_orange_ind = list(set(item for sublist in list(affected_households_orange.cov_pop_index) for item in sublist))
        potential_waterwells_orange = new_wps.iloc[potential_waterwells_orange_ind, :].copy()
        potential_waterwells_orange.reset_index(drop=True, inplace = True)
        def find_covering_pop(geo):
            a = list(potential_waterwells_orange[potential_waterwells_orange.geometry.within(geo)].index)
            return a
        affected_households_orange['cov_pop_index'] = affected_households_orange.geometry.buffer(buf_opt).parallel_apply(find_covering_pop)
        # print(affected_households_orange)
        def find_covering_pop(geo):
            a = list(affected_households_orange[affected_households_orange.geometry.within(geo)].index)
            return a
        # print(potential_waterwells_orange)
        potential_waterwells_orange['cov_pop_index'] = potential_waterwells_orange.geometry.buffer(buf_opt).parallel_apply(find_covering_pop)
        X, _ = optimize_locations(affected_households_orange, potential_waterwells_orange, current_locations, 'TEST', GRID_DIST = 250, CHOICE = 'maxcover', OBJECTIVE = 'min_number_wells', MAX_NUMBER_WELLS=None, MIN_COVER = [100], MIP_GAP = 0.001, PLOT_INFO= False, PLOT_PARETO=False, CASENAME='CHEESE', printall=False)
        optimal_locations_orange = potential_waterwells_orange.iloc[np.where(X>0)[0],:].reset_index(drop = True)
        #####################################################
    
        ########## Solve for all purple households ##########
        affected_households_purple = affected_households[affected_households.color == 'purple'].reset_index(drop=True)
        if len(affected_households_purple)==0: return 0
        potential_waterwells_purple_ind = list(set(item for sublist in list(affected_households_purple.cov_pop_index) for item in sublist))
        potential_waterwells_purple = new_wps.iloc[potential_waterwells_purple_ind, :].copy()
        potential_waterwells_purple.reset_index(drop=True, inplace = True)
        def find_covering_pop(geo):
            a = list(potential_waterwells_purple[potential_waterwells_purple.geometry.within(geo)].index)
            return a
        # print(affected_households_purple)
        affected_households_purple['cov_pop_index'] = affected_households_purple.geometry.buffer(buf_opt).parallel_apply(find_covering_pop)
        #print(affected_households_purple) 
        def find_covering_pop(geo):
            a = list(affected_households_purple[affected_households_purple.geometry.within(geo)].index)
            return a
        potential_waterwells_purple['cov_pop_index'] = potential_waterwells_purple.geometry.buffer(buf_opt).parallel_apply(find_covering_pop)
        # print(potential_waterwells_purple)
        X, _ = optimize_locations(affected_households_purple, potential_waterwells_purple, current_locations, 'TEST', GRID_DIST = 250, CHOICE = 'maxcover', OBJECTIVE = 'min_number_wells', MAX_NUMBER_WELLS=None, MIN_COVER = [100], MIP_GAP = 0.001, PLOT_INFO= False, PLOT_PARETO=False, CASENAME='CHEESE', printall=False)
        optimal_locations_purple = potential_waterwells_purple.iloc[np.where(X>0)[0],:].reset_index(drop = True)
        #####################################################
    
        ############ Visualize the two solutions ############
        VISUALIZE_SOLUTIONS = False
        if VISUALIZE_SOLUTIONS:
            optimal_locations = pd.concat([optimal_locations_orange, optimal_locations_purple], axis=0)#.reset_index(drop = True)
            affected_households = pd.concat([affected_households_orange, affected_households_purple], axis=0)#.reset_index(drop = True)
            potential_waterwells = pd.concat([potential_waterwells_orange, potential_waterwells_purple], axis=0)#,reset_index(drop=True)
            visualize_situation(c, affected_households, None, potential_waterwells, optimal_locations)
        #####################################################
    
        # The bound on the amount of facilities
        bound_facilities = max(len(optimal_locations_orange), len(optimal_locations_purple))
        """-----------------------------------------------------------------------------"""
        return bound_facilities

#TODO DONE   
def determine_optimality_guarantee_COVERAGE(df_pop_temp, new_wps, k=2):
    """
        Compute how far a clustered solution is from the optimal value. 
    
    Parameters
    ----------
    df_pop_temp : population data including indication to which 
            cluster a household is assigned
        dataframe

    new_wps : potential waterwell locations for this whole cluster
        dataframe
    
    k : [DEVELOPMENT] amount of subclusters used.
    
    Returns
    -------
    bound_facilities : the optimality for the amount of facilities when using 
             these two subclusters.
        float
    """
    
    """--------------------- Find potential affected households --------------------"""
    # To save computation time, we first only find households that may be affected by 
    # the split of the cluster. To do this, we first find subsubclusters taht are 
    # then take the 
    VISUALIZE_POTENTIAL_AFFECTED_HH = False
    # Create the subclusters using DBSCAN
    maxlabelsOLD = 0
    for i in range(k):
        # Run DBSCAN
        X = df_pop_temp.loc[np.where(df_pop_temp.cluster==i)[0], ['X', 'Y']].to_numpy()#.reset_index(drop=True)
        dbscan = DBSCAN(eps=1000, min_samples=1)
        dbscan.fit(X)
        labels = dbscan.labels_
        df_pop_temp.loc[np.where(df_pop_temp.cluster==i)[0],'cluster_2'] = labels+maxlabelsOLD
        maxlabelsOLD = maxlabelsOLD+max(labels)+1
    # Create the polygons that form the bands that contain the potentially affected households
    if VISUALIZE_POTENTIAL_AFFECTED_HH: 
        fig, ax = plt.subplots()
    buffered_polygons_POS, buffered_polygons_NEG = [],[]
    
    # print(df_pop_temp)
    for c in df_pop_temp.cluster_2.unique():
        ########################################################################
        ######################### Filter the households ########################
        ########################################################################
        points_cluster = list(zip(df_pop_temp.loc[df_pop_temp.cluster_2==c,'X'], df_pop_temp.loc[df_pop_temp.cluster_2==c,'Y']))
        ########################################################################
        #print(points_cluster)
        ########################################################################
        ################# Create the outer boundary (positive) #################
        ########################################################################
        cv = gpd.GeoSeries([Point(x, y) for x, y in points_cluster]).buffer(500).unary_union
       # print(cv)
        try: 
            buffered_polygon_pos = cv.exterior
            polygon_x, polygon_y = buffered_polygon_pos.xy
            if VISUALIZE_POTENTIAL_AFFECTED_HH: ax.plot(polygon_x, polygon_y, color='purple')
            buffered_polygons_POS.append(Polygon(buffered_polygon_pos))
        except: 
            for poly in cv.geoms:
                buffered_polygon_pos = poly.exterior
                # Extract coordinates for plotting the buffered polygon
                buffered_polygon_x, buffered_polygon_y = buffered_polygon_pos.xy
                # Plot the buffered polygon
                if VISUALIZE_POTENTIAL_AFFECTED_HH: ax.plot(buffered_polygon_x, buffered_polygon_y, color='purple')
                buffered_polygons_POS.append(Polygon(buffered_polygon_pos))
        ########################################################################
                
        ########################################################################
        ################# Create the inner boundary (negative) #################
        ########################################################################
        copied_polygon = Polygon(cv.exterior)
        buffered_polygon_neg = copied_polygon.buffer(-1000)
        try: 
            buffered_polygon_neg = buffered_polygon_neg.exterior
            polygon_x, polygon_y = buffered_polygon_neg.xy
            if VISUALIZE_POTENTIAL_AFFECTED_HH: ax.plot(polygon_x, polygon_y, color='green')
            buffered_polygons_NEG.append(Polygon(buffered_polygon_neg))
        except: 
            for poly in buffered_polygon_neg.geoms:
                buffered_polygon_neg = poly.exterior
                # Extract coordinates for plotting the buffered polygon
                buffered_polygon_x, buffered_polygon_y = buffered_polygon_neg.xy
                # Plot the buffered polygon
                if VISUALIZE_POTENTIAL_AFFECTED_HH: ax.plot(buffered_polygon_x, buffered_polygon_y, color='green')
                buffered_polygons_NEG.append(Polygon(buffered_polygon_neg))
        ########################################################################
    # Filter the households to only the potentially affected households
    def within_potential_bands(point):
        for buf in buffered_polygons_NEG:
            if buf.contains(point):
                return False
        return True
    points_within_polygon = df_pop_temp.geometry.apply(within_potential_bands)
    df_pop_temp_FULL = df_pop_temp[points_within_polygon].reset_index(drop=True)
    # Visualize the bands
    if VISUALIZE_POTENTIAL_AFFECTED_HH:
        ax.plot(df_pop_temp.X, df_pop_temp.Y, 'o', color='pink', label='households')
        ax.plot(df_pop_temp_FULL.X, df_pop_temp_FULL.Y, 'o', color='red', label='households')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_title('Polygon, Generated Points, and Buffered Polygon')
        ax.legend()
        plt.show()
    """-----------------------------------------------------------------------------"""
 
    """ --------------------------- Create the graph -------------------------------"""
    VISUALIZE_GRAPH = False
    # Having a graph, we can easily find the households that may be affected by the 
    # split of the cluster. 
    def find_covering_pop(geo_point, MAX_POINT, df_pop_temp):
        geo = geo_point.buffer(1000)
        a = df_pop_temp[df_pop_temp.geometry.within(geo)].copy()
        
        a['distance'] = a.geometry.distance(geo_point)
        a['angle'] = np.degrees(np.arctan2(a.geometry.y - geo_point.y, a.geometry.x - geo_point.x))+180
        a.sort_values(by=['distance', 'angle'], inplace=True, ascending = False)
        
        a = list(a.index)
        if MAX_POINT== 'all': return a#[1::]
        elif len(a)>MAX_POINT: return a[0:MAX_POINT]#return [a[i] for i in numbers[0:10]]# a[0:10]
        else: return a[0::]
    df_pop_temp_FULL['cov_pop_index_self_FULL'] = df_pop_temp_FULL.geometry.parallel_apply(find_covering_pop, args=('all', df_pop_temp_FULL))
    edges_FULL = []
    with Bar('Processing', max=len(df_pop_temp_FULL)) as bar:
        for i in df_pop_temp_FULL.index:
            for l in df_pop_temp_FULL.loc[i, 'cov_pop_index_self_FULL']:
            
                if ((l,i) in edges_FULL) | (i == l): continue
                
                else: edges_FULL.append((i, l))
            bar.next()
    #print("All edges:", len(edges_FULL))
    # Create the graph
    G_FULL = nx.Graph()
    G_FULL.add_nodes_from(range(0,len(df_pop_temp_FULL)))
    G_FULL.add_edges_from(edges_FULL)
    
    # Define a list of colors (you can customize this list)
    colors = ['orange', 'purple', 'green', 'red', 'yellow', 'purple', 'pink', 'brown', 'gray', 'cyan']
    # Assign colors to rows based on the 'Numbers' column
    df_pop_temp_FULL['color'] = df_pop_temp_FULL['subcluster'].apply(lambda x: colors[x % len(colors)])
    node_colors = df_pop_temp_FULL.color
    ################ Visualize the graph ################
    if VISUALIZE_GRAPH:
        pos = {node: (x, y) for node, x, y in zip(G_FULL.nodes, df_pop_temp_FULL.X, df_pop_temp_FULL.Y)}
        nx.draw(G_FULL, pos,  with_labels=False, node_size=50, edge_color='grey', node_color=node_colors)
        plt.show()
    #####################################################
    """-----------------------------------------------------------------------------"""
    
    """---------------------- Determine the real cut value --------------------------"""
    # Determine the real cut value, but also find all the cut edges. These are found 
    # to determine which people could be affected by the split.
    def extract_edges_with_different_colors(edges):
        edges_with_different_colors = []
        for e in edges:
            color_from = node_colors[e[0]]
            color_to = node_colors[e[1]]
            if color_from != color_to:
                edges_with_different_colors.append(e)
        return edges_with_different_colors
    cut_edges = extract_edges_with_different_colors(edges_FULL)
    # print('Bound 1:', len(cut_edges))
    """-----------------------------------------------------------------------------"""
    
    HOUSEHOLDS_BOUND = True
    if HOUSEHOLDS_BOUND:
        """--------------------- Give some optimality guarantee ------------------------""" 
        # Find all potential well locations that can serve multiple clusters
        affected_households_ind = list(set(item for sublist in cut_edges for item in sublist))
        affected_households = df_pop_temp_FULL.iloc[affected_households_ind, :].reset_index(drop=True)
        affected_households_per_color_ind = []
        for col in affected_households.color.unique():
            affected_households_temp_col = affected_households[affected_households.color == col].copy().reset_index(drop=True)
            potential_waterwells_col_ind = list(set(item for sublist in list(affected_households_temp_col.cov_pop_index) for item in sublist))
            affected_households_per_color_ind.append(potential_waterwells_col_ind)
        element_counts = Counter(element for sublist in affected_households_per_color_ind for element in sublist)
        overlap_ind = [element for element, count in element_counts.items() if count >= 2]
        
        # Prepare data that is relevant for this particular region (where potential waterwells can cover at least 2 subclusters)
        potential_waterwells = new_wps.iloc[overlap_ind, :].copy()
        potential_waterwells.reset_index(drop=True, inplace = True)
        def find_covering_pop(geo):
            a = list(potential_waterwells[potential_waterwells.geometry.within(geo)].index)
            return a
        affected_households['cov_pop_index'] = affected_households.geometry.buffer(MAX_DIST).parallel_apply(find_covering_pop)
        def find_covering_pop(geo):
            a = list(affected_households[affected_households.geometry.within(geo)].index)
            return a
        potential_waterwells['cov_pop_index'] = potential_waterwells.geometry.buffer(MAX_DIST).parallel_apply(find_covering_pop)
    
        # Visualize the starting situation
        #visualize_situation(c, affected_households , None, potential_waterwells, None)
        
        def max_coverage_clusters(df_pop, new_wps, current_locations, CLUSTER, GRID_DIST, CHOICE, OBJECTIVE, MAX_NUMBER_WELLS, MIN_COVER, MIP_GAP, PLOT_INFO, PLOT_PARETO, CASENAME, printall=True):
            def getvariables(X, Y):
                Xvalues = np.zeros(len(new_wps))
                for i in range(len(new_wps)):
                    Xvalues[i]=X[i].x
                Yvalues = np.zeros((len(df_pop),len(new_wps)))
                for i in range(len(df_pop)):
                    for j in range(len(new_wps)):
                        try: Yvalues[i,j]=Y[i,j].x
                        except: Yvalues[i,j] = Y[i,j]
                return(Xvalues, Yvalues)
            ################################################################################################
            ########################## Model settings that are used more often #############################
            ################################################################################################
            # pop_arr = df_pop.VALUE.to_numpy()
            E = df_pop.cov_pop_index.to_dict()
            ################################################################################################
    
            M = Model()
            
            # Decision variables
            print("Create the variables:")
            s = time.time()
            X = M.addVars(len(new_wps), vtype=GRB.BINARY, name='x')
            bar = Bar('Processing', max=len(df_pop))
            Y = {}
            for i in range(len(df_pop)):
                for j in range(len(new_wps)):
                    if j not in E[i]: Y[i,j] = 0
                    else: 
                        Y[i, j] = M.addVar(lb = 0, ub = 1)#vtype=GRB.BINARY)
                bar.next()
            bar.finish()
            
            t = M.addVars(len(new_wps))
            print(time.time()-s, "seconds passed.")
            
            # Currently open facilities
            c = M.addConstrs((X[i] == 1 for i in current_locations))
            # Limit number of facilities located
            h = M.addConstr(X.sum() <= MAX_NUMBER_WELLS[0]+len(current_locations))

            # Make sure that y[i,f]=0 if there is no well at f)
            s = time.time()
            bar = Bar('Processing', max=len(new_wps))
            for f in range(len(new_wps)):
                M.addConstr(quicksum(Y[i,f] for i in range(len(df_pop))) <= len(new_wps) * X[f])
                bar.next()
            bar.finish()
            print(time.time()-s, "seconds passed.")
            
            # print('Making sure that i can only be assigned to one f')
            s = time.time()
            bar = Bar('Processing', max=len(df_pop))
            for i in range(len(df_pop)):
                M.addConstr(quicksum(Y[i,f] for f in range(len(new_wps))) <= 1)
                bar.next()
            bar.finish()
            print(time.time()-s, "seconds passed.")
            
            # For each selected waterwell, make sure that the t (the objective) <= the worst case scenario. 
            # Create [hh_values], for each subcluster, set the value of that cluster to 0
            hh_values = []
            for col in df_pop.color.unique():                
                g = [val if satisfy_condition else 0 for val, satisfy_condition in zip(df_pop['VALUE'], df_pop['color'] != col)]
                hh_values.append(g)
            for f in range(len(new_wps)):
                for z in range(len(hh_values)):
                    M.addConstr(quicksum([hh_values[z][i]*Y[i,f] for i in range(len(df_pop))])>=t[f])
            
            # Define the objective function
            #obj = quicksum(pop_arr[i]*Y[i,f] for i in range(len(df_pop)) for f in range(len(new_wps)))
            #M.setObjective(obj, GRB.MAXIMIZE)
            M.setObjective(t.sum(), GRB.MAXIMIZE)

            # Set parameters 
            M.setParam('OutputFlag', False) # Suppress output during optimization
            M.Params.mipgap = MIP_GAP   #0.01
            
            if OBJECTIVE == "max_cover":
                cur_loc_start = len(current_locations)
                obj_val_array = []
                obj_val_OLD=1000000
                for max_ww in MAX_NUMBER_WELLS:
                    start = time.time()
                    print("Optimizing for", max_ww, 'waterwells.')
                    M.remove(h)
                    h = M.addConstr(X.sum() <= max_ww + cur_loc_start)
                    M.optimize()
                    obj_val = M.objVal
                    Xvalues, Yvalues = getvariables(X, Y)
                    coverage = np.round(100*(obj_val/sum(df_pop.VALUE)), 2)

                    print(time.time()-start, "seconds passed!")
                    obj_val_array.append([CLUSTER, GRID_DIST, max_ww, obj_val, coverage])#, new_add])#,list(Xvalues),list(Yvalues)])
                    if PLOT_INFO:
                        ############################################ Print output ############################################
                        print("    Objective value:", obj_val, "households.")
                        print("    Coverage:", coverage, "%")
                        print("    Opening", max_ww, "water wells.")
                        print("_____________________")
                        ######################################################################################################
                    if round(obj_val_OLD, 1) == round(obj_val,1): break
                    obj_val_OLD = obj_val
                    
                # if not os.path.exists("Subclustering/Bounds"): os.makedirs("Subclustering/Bounds")
                df_final = pd.DataFrame(obj_val_array, columns = ['Cluster', 'Grid_dist', 'MaxNumberWaterwells', "Households covered", "Coverage (%)"])#, "NewAdd"])
                #df_final.to_csv("Subclustering/Bounds/"+CASENAME+"_"+str(MAX_POP)+'.csv', sep = ";")
                #print(df_final)
                if PLOT_PARETO:
                    fig = plt.Figure()
                    plt.plot(df_final['MaxNumberWaterwells'], df_final['Coverage (%)'])
                    plt.xlabel('Maximum number of wells to be placed (budget)')
                    plt.ylabel('Maximum coverage achieved (%)')
                    plt.show()
        
            return Xvalues, obj_val, df_final
        
        # Solve the optimization problem
        TESTBOUND = [len(overlap_ind)]#[len(overlap_ind)-6]#range(100,len(overlap_ind))
        X, bound_coverage, df_final = max_coverage_clusters(affected_households, potential_waterwells, current_locations, affected_households.cluster_order[0], GRID_DIST = 250, CHOICE = 'maxcover', OBJECTIVE = 'max_cover', MAX_NUMBER_WELLS=TESTBOUND, MIN_COVER = None, MIP_GAP = 0.00001, PLOT_INFO= True, PLOT_PARETO=False, CASENAME='bound_coverage', printall=False)

        # If you want to visualize each solution separately
        # objs = []
        # for NUM in TESTBOUND:
        #     X, obj_val = max_coverage_clusters(affected_households, potential_waterwells, current_locations, 'B_COVERAGE', GRID_DIST = 250, CHOICE = 'maxcover', OBJECTIVE = 'max_cover', MAX_NUMBER_WELLS=[NUM], MIN_COVER = None, MIP_GAP = 0.001, PLOT_INFO= True, PLOT_PARETO=False, CASENAME='bound_coverage', printall=False)
        #     optimal_locations = potential_waterwells.iloc[np.where(X>0)[0],:].reset_index(drop = True)
        #     objs.append(obj_val)
        #     #visualize_situation(c, affected_households , None, None, optimal_locations)
        #     # Visualize solution per opened well
        #     # s = 0
        #     # for i in range(0,NUM):
        #     #     unique_elements = optimal_locations.cov_pop_index[i]
        #     #     a = affected_households.iloc[unique_elements,:].copy().reset_index(drop=True)
        #     #     # You could obtain an upper bound for the objective (this is when you always need to choose the same color)
        #     #     h = sum(a.loc[a.color=='purple','VALUE'])
        #     #     k = sum(a.loc[a.color=='orange','VALUE'])
        #     #     s += min(h,k)
        #     #     visualize_situation(c, a, None, None, optimal_locations.iloc[i:(i+1),:].reset_index(drop=True))
        # Save the data
        # bound_coverage = pd.DataFrame(objs)
        # bound_coverage.to_csv('bound_coverage.csv',sep = ";")
    
        return df_final#bound_coverage

#TODO DONE
def subcluster(df, new_wps, current_locations):
    """
        Create subclusters of population data.
    
    Parameters
    ----------
    df : population data for a specific cluster. 
        dataframe

    new_wps : potential waterwell locations for this whole cluster
        dataframe
    
    current_locations : list of indices that indicate which wells are 
            already in operation (out fo the new_wps dataframe)
        list
    
    Returns
    -------
    df_cluster_0 : population data filtered to the new subcluster 0
        dataframe
    
    df_cluster_1 : population data filtered to the new subcluster 1
        dataframe
    
    bound : the optimality for the amount of facilities when using 
             these two subclusters.
        float
    """
    df_pop_temp = gpd.GeoDataFrame(df, geometry='geometry', crs='32634')
    df_pop_temp_LATLON = df_pop_temp.to_crs(crs='EPSG:4326')
    # Amount of clusters
    k = 2 # (>2 is only in test phase)
    METHOD = 2
    """----------- Use the most simple instant heuristic sorting -----------"""
    if METHOD==1:
        ###########################################################################
        ####################### Find the two extreme points #######################
        ###########################################################################
        points = list(zip(df_pop_temp.X, df_pop_temp.Y))
        cv = MultiPoint([Point(x, y) for x, y in points]).convex_hull
        from scipy.spatial.distance import cdist
        hull_points = np.array(cv.exterior.coords)
        # Calculate pairwise distances
        distance_matrix = cdist(hull_points, hull_points)
        max_distance_indices = np.unravel_index(np.argmax(distance_matrix), distance_matrix.shape)
        # Retrieve the points with the maximum distance
        point1_index, point2_index = max_distance_indices
        point1 = hull_points[point1_index]
        point2 = hull_points[point2_index]
        # Print the points with the maximum distance
        print("Point 1:", point1, ', index:', point1_index)
        print("Point 2:", point2, ', index:', point2_index)
        initial_points = np.array(points)
        point1_initial_index = np.where(np.all(initial_points == point1, axis=1))[0][0]
        point2_initial_index = np.where(np.all(initial_points == point2, axis=1))[0][0]
        # Print the indices of the corresponding points
        print("Indices of Point 1 in the initial list:", point1_initial_index)
        print("Indices of Point 2 in the initial list:", point2_initial_index)
        print(df_pop_temp_LATLON.iloc[[point1_initial_index, point2_initial_index], :])
        ###########################################################################
        
        ###########################################################################
        ############################# FIND DISTANCES ##############################
        ###########################################################################
        # Add the clostest 50% of the points to the orange cluster
        def test1(point):
            return point.distance(Point(point1))
        def test2(point):
            return point.distance(Point(point2))
        df_pop_temp['dist_to_one'] = df_pop_temp.geometry.parallel_apply(test1)
        df_pop_temp['dist_to_two'] = df_pop_temp.geometry.parallel_apply(test2)
        df_pop_temp = df_pop_temp.sort_values(by= ['dist_to_one'], inplace = False).reset_index(drop=True)
        df_pop_temp['color'] = 'purple'
        df_pop_temp.loc[0:np.ceil(len(df_pop_temp)/2), 'color'] = 'orange'
        df_pop_temp['cluster'] = [0 if i == 'orange' else 1 for i in df_pop_temp.color]
        print(df_pop_temp)
        ###########################################################################
        
        ###########################################################################
        ################################# VISUALIZE ###############################
        ###########################################################################
        #visualize_situation(CLUSTER=0, PATH_TO_POPULATION = df_pop_temp, PATH_TO_CURRENT_WATERWELLS=None, PATH_TO_POTENTIAL_WATERWELLS=None, PATH_TO_OPTIMAL_WATERWELLS=None)
        ###########################################################################
    """---------------------------------------------------------------------"""
    """--------------- Use alternating adding worst to other ---------------"""
    if METHOD==2:
        PLOT_IO = False
        ###########################################################################
        ####################### Find the two extreme points #######################
        ###########################################################################
        points = list(zip(df_pop_temp.X, df_pop_temp.Y))
        cv = MultiPoint([Point(x, y) for x, y in points]).convex_hull
        from scipy.spatial.distance import cdist
        hull_points = np.array(cv.exterior.coords)
        distance_matrix = cdist(hull_points, hull_points) # Calculate pairwise distances
        
        # Function to select the extreme points
        def maximum_min_distance_gurobi(points, k):
            """
              Find the set of k points that maximizes the minimum pairwise distance between 
              these k points. Use points of a convex hull
            
            Parameters
            ----------
            points : distance matrix for all considered points
                ndarray
    
            k : Amount of points you want to find
                float
    
            Returns
            -------
            model.objVal : objective value of the optimization problem (the maximized, minium distance))
                float
            
            selected_points : indices of all points that are selected
                list
            
            """
            n = len(points)
            model = Model("MaximumMinDistance")

            # Binary variable indicating whether a point is selected
            x = model.addVars(n, vtype=GRB.BINARY, name="x")
            # Variable representing the objective function minimum distance between the selected extreme points
            t = model.addVar(name="t")

            # Auxiliary variables introduces to linearize the model
            z = model.addVars(n, n, vtype=GRB.BINARY, name="z")  # Additional binary variable
              
            # Original nonlinear constraint
            # M = 100000000
            # for i in range(n):
            #    for j in range(i + 1, n):
            #        model.addConstr(t <= x[i]*x[j]*points[i][j]) + (1-x[i]*x[j])*M
            
            # Linearized version (x[i]*x[j] = z[i,j])
            M = 100000000
            for i in range(n):
                for j in range(i + 1, n):
                    model.addConstr(z[i, j] >= x[i] + x[j] - 1)
                    model.addConstr(z[i, j] <= x[i])
                    model.addConstr(z[i, j] <= x[j])
                    model.addConstr(t <= z[i, j] * points[i][j] + (1 - z[i, j]) * M)
                    
            # Maximize the minimum distance
            model.setObjective(t, GRB.MAXIMIZE)

            # Total number of selected points must be equal to k
            model.addConstr(quicksum(x[i] for i in range(n)) == k)
            
            model.setParam('OutputFlag',0) # Suppress output during optimization
            model.Params.mipgap = 0.000001   #0.01
            
            # Solve the model
            model.optimize()

            selected_points = [i for i in range(n) if x[i].x > 0.5]
            return model.objVal, selected_points
        obj, selected_points = maximum_min_distance_gurobi(distance_matrix, k=k)
        
        # find the index of all the extreme points in the set of initial points (not the index in the convex hull)
        point_initial_index = []
        initial_points = np.array(points)
        for i in range(k):
            p = hull_points[selected_points[i]]
            pii = np.where(np.all(initial_points == p, axis=1))[0][0]
            point_initial_index.append(pii)
        ###########################################################################
        
        ###########################################################################
        ######################### Initialize the clusters #########################
        ###########################################################################
        coords = list(zip(*df_pop_temp[['X', 'Y']].T.values))
        N, N_indices = [],[] # N will be a list that contains k lists that contain the points of that belong to these clusters
        # Add the extreme points (and their indices) to the respective clusters.
        for i in range(k):
            N.append([coords[point_initial_index[i]]])
            N_indices.append([point_initial_index[i]])
        # Define a set of remaining points (and their indices) that need to be assigned to all the clusters. 
        P = [item for item in coords if item not in [item for sublist in N[0:k] for item in sublist]]
        P_indices = [item for item in range(len(coords)) if item not in [item for sublist in N_indices[0:k] for item in sublist]]
        ###########################################################################
        
        ###########################################################################
        ############################ Fill the clusters ############################
        ###########################################################################
        # Turn on interactive mode
        if PLOT_IO:
            plt.ion()
            fig, ax = plt.subplots()
        colors = ['purple', 'orange','green','brown','yellow','cyan']
        turn = 0
        for i, _ in enumerate(tqdm(range(len(P_indices)))):
            # Find the points in the cluster that is playing (turn)
            Nt = N[turn]
            tree = cKDTree(np.array(Nt))
            # Compute the distances to all the remaining points in P
            distances, _ = tree.query(np.array(P), k=1)
            
            # Now we want to subtract the distances from the other cluster to the points 
            # in P. In this way, we include that we prefer points that are close to the 
            # cluster to be added. 
            # Find the points in the cluster that is NOT playing (NOT turn)
            turn_to = abs(turn-1)
            Nt_2 = N[turn_to]
            tree_2 = cKDTree(np.array(Nt_2))
            # Compute the distances to all the remaining points in P
            distances_2, _ = tree_2.query(np.array(P), k=1)
            # Subtract the distances
            distances -= distances_2

            # So the point we want to add to the other cluster is the
            # point that has the maxmium distance (after subtraction)
            furthest_point = tuple(P[np.argmax(distances)])
            furthest_index = P_indices[np.argmax(distances)]
            
            # Stuff when k > 2
            # psbl = list(range(k))
            # psbl.remove(turn)
            # dists = []
            # for l in psbl:
            #     Nt = N[l]
            #     tree = cKDTree(np.array(Nt))
            #     distances, indices = tree.query(furthest_point, k=1)
            #     dists.append(np.min(distances))
            # turn_to = psbl[np.argmin(dists)]
    
            # Add the point to the turn_to cluster
            N[turn_to].append(furthest_point)
            N_indices[turn_to].append(furthest_index)  # Append the closest inde
            
            # Remove the point from the remaining points
            P.remove(furthest_point)
            P_indices.remove(furthest_index)
            
            # Turn to the other cluster
            # if turn < k: turn+=1
            # else: turn=0
            turn = turn_to
            
            # Update the graph every 10 points
            if PLOT_IO and (i % 10 == 0):
                ax.clear()  # Clear previous plot
                count = 0
                for i in N[0:k]:
                    ax.scatter(*np.array(i).T, color=colors[count])
                    count+=1

                ax.scatter(*np.array(P).T, color='b')
                ax.scatter(*furthest_point, color='yellow')

                ax.axis('equal')
                plt.draw()  # Draw updated plot
                plt.pause(0.001)  # Pause for half a second
        # Turn off interactive mode
        if PLOT_IO: plt.ioff()  

        # Give each cluster a separate color, also in the data
        df_pop_temp['color'] = 'purple'
        df_pop_temp['cluster'] = 0
        count = 1
        for i in N_indices[1::]:
            df_pop_temp.loc[i, 'color'] = colors[count]
            df_pop_temp.loc[i, 'cluster'] = count
            count+=1
        ###########################################################################
    """---------------------------------------------------------------------"""
    """------------- Create graph if using graph based method --------------"""
    if METHOD in [3]:
        def find_covering_pop(geo_point, MAX_POINT, df_pop_temp):
            geo = geo_point.buffer(1000)
            a = df_pop_temp[df_pop_temp.geometry.within(geo)].copy()
        
            a['distance'] = a.geometry.distance(geo_point)
            a['angle'] = np.degrees(np.arctan2(a.geometry.y - geo_point.y, a.geometry.x - geo_point.x))+180
            a.sort_values(by=['distance', 'angle'], inplace=True, ascending = False)
            a = list(a.index)
            if MAX_POINT== 'all': return a#[1::]
            elif len(a)>MAX_POINT: return a[0:MAX_POINT]#return [a[i] for i in numbers[0:10]]# a[0:10]
            else: return a[0::]
        df_pop_temp['cov_pop_index_self'] = df_pop_temp.geometry.parallel_apply(find_covering_pop, args=(10, df_pop_temp))
        # print(df_pop_temp)
        edges = []
        with Bar('Processing', max=len(df_pop_temp)) as bar:
            for i in df_pop_temp.index:
                for l in df_pop_temp.loc[i, 'cov_pop_index_self']:
                    if ((l,i) in edges) | (i == l): continue
                    # if l >= i: continue
                    else: edges.append((i, l))
                bar.next()     
        print("Pruned edges:", len(edges))
        # Create an empty graph
        G = nx.Graph()
        G.add_nodes_from(range(0,len(df_pop_temp)))
        G.add_edges_from(edges)
    """----------------- Use the ILP formulation (EXACT) -------------------""" 
    if METHOD==3:
        c = {(i,j) : 1 for (i,j) in G.edges}

        # Create model object
        m = gp.Model()

        # Create variable for each edge, indicating whether it is cut
        y = m.addVars(G.edges, lb = 0, ub=1)#, vtype=gp.GRB.BINARY)

        # Create variable for each node, indicating whether it is on s-side of cut
        z = m.addVars(len(G.nodes), vtype=gp.GRB.BINARY)

        # Ensure approximately similar-sized sub-clusters
        m.addConstr(z.sum() >= 0.4*len(z))
        m.addConstr(z.sum() <= 0.6*len(z))

        # Objective function: minimize (weight of) cut
        m.setObjective( gp.quicksum( c[e]*y[e] for e in G.edges ), gp.GRB.MINIMIZE )

        # Constraints: edge (i,j) is cut if (i is with s) and (j is not).
        m.addConstrs( y[i,j] >= z[i] - z[j] for i,j in G.edges )
        m.addConstrs( y[i,j] >= z[j] - z[i] for i,j in G.edges )
        m.setParam('OutputFlag',0) # Suppress output during optimization

        # Solve
        m.optimize()

        # Get the cut value (Note this cut value is from the smaller pruned graph) NOT REPRESENTATIVE!!!
        cut_value = m.objVal
        # print("Cut Value (Pruned):", cut_value)
        S = [ i for i in G.nodes if z[i].x > 0.5 ] # s-side of cut
        node_colors = [ "purple" if i in S else "orange" for i in G.nodes ]
        df_pop_temp['color'] = node_colors
        df_pop_temp['cluster'] = [ 0 if i in S else 1 for i in G.nodes]
        
        ###########################################################################
        ################################# VISUALIZE ###############################
        ###########################################################################
        # visualize_situation(CLUSTER=c, PATH_TO_POPULATION = df_pop_temp, PATH_TO_CURRENT_WATERWELLS=None, PATH_TO_POTENTIAL_WATERWELLS=None, PATH_TO_OPTIMAL_WATERWELLS=None)
        ###########################################################################
    """---------------------------------------------------------------------"""
    
    # Determine the optimality guarantuee for the amount of facilites (wells)
    bound_facilities = 0#determine_optimality_guarantee_FACILITIES(df_pop_temp, new_wps, k=k)
    # print("BOUND (facilities):", bound_facilities)    
    df_cluster_0 = df_pop_temp.loc[df_pop_temp.cluster == 0, :].reset_index(drop=True)
    df_cluster_1 = df_pop_temp.loc[df_pop_temp.cluster == 1, :].reset_index(drop=True)
    return df_cluster_0, df_cluster_1, bound_facilities

#TODO DONE
def split_dataframe(df, new_wps, current_locations, threshold, current_id=1):
    """
        Recursive function to keep splitting a dataframe until all 
        subclusters are having <= [threshold] households.
    
    Parameters
    ----------
    df : population data for a specific cluster. 
        dataframe

    new_wps : potential waterwell locations for this whole cluster
        dataframe
    
    current_locations : list of indices that indicate which wells are 
            already in operation (out fo the new_wps dataframe)
        list
    
    threshold : maximum amount of households per (sub)cluster
        float
    
    current_id : [DO NOT ADJUST!] used for numbering the subclusters
        string
    
    Returns
    -------
    result : population data that contains now a column that indicates to which
            subcluster a household is assigned.
        dataframe
    
    total_boundadd : the optimality bound when using when solving for 100% 
        float 
    """
    # Do not split if not too much households
    if len(df) <= threshold:
        df['subcluster'] = current_id
        return df, 0
    # Split the dataframe into two
    left_half, right_half, boundadd_facilities = subcluster(df, new_wps, current_locations)
    # Assign cluster IDs for each half
    left_half['subcluster'] = current_id
    right_half['subcluster'] = current_id + 1
    # Recursive calls for each half
    left_half, left_boundadd_facilities = split_dataframe(left_half, new_wps, current_locations, threshold, current_id=current_id * 2)
    right_half, right_boundadd_facilities = split_dataframe(right_half, new_wps, current_locations, threshold, current_id=current_id * 2 + 1)
    # Sum the total bounds
    total_boundadd_facilities = boundadd_facilities + left_boundadd_facilities + right_boundadd_facilities
    # Concatenate the results
    result = pd.concat([left_half, right_half], ignore_index=True)
    return result, total_boundadd_facilities

bound_coverage_df, df_pop_clustered_NEW, new_clust = pd.DataFrame(), pd.DataFrame(), 0
runtimes = []
for c in CLUSTERS:
    ######################### Load data for the specific cluster #########################
    start = time.time()
    print("Cluster: ", c)
    new_wps, df_pop_temp = load_newwps(c, df_pop, waterwells, GRID_DIST, buf_opt = MAX_DIST, remove_dominated = REMOVE_DOMINATED, save = True, CASENAME = CASENAME)
    if AMOUNT_SUBCLUSTERS is not None:
        MAX_POP = (len(df_pop_temp)/(AMOUNT_SUBCLUSTERS))+math.log2(AMOUNT_SUBCLUSTERS) # Last addition
        if (MAX_POP_OLD != MAX_POP) & (MAX_POP_OLD is not None):
            print(MAX_POP)
            print("MAXPOP and AMOUNT_SPLITS do not align! Please adjust...")
            stop     
    # continue # If you first want to time creating the information
    time_well_generation = time.time()-start
    ######################################################################################
    
    ######################################################################################
    # Print progress
    start = time.time()
    total_pop = sum(df_pop_temp.VALUE)
    print("Subclustering cluster:", c, "Size:",(len(df_pop_temp), len(new_wps)), 'Population:', total_pop)
    # Subcluster the population data
    df_pop_temp['subcluster'] = 0
    df_pop_temp_subclustered, bound_facilities = split_dataframe(df_pop_temp, new_wps, current_locations, MAX_POP)
    df_pop_temp_subclustered['subcluster'] = df_pop_temp_subclustered['subcluster']-min(df_pop_temp_subclustered['subcluster'])
    time_SC = time.time()-start
    
    
    # If no subclusters were created, skip finding the upper bound, and adjusting the df_pop_temp
    if max(df_pop_temp_subclustered['subcluster'])>0:
        start = time.time()
        bound_coverage = determine_optimality_guarantee_COVERAGE(df_pop_temp_subclustered, new_wps, k=2)
        df_pop_temp_subclustered = df_pop_temp_subclustered.loc[:,columns_needed+['subcluster']]
        time_UB = time.time() - start

        bound_coverage['Coverage (%)'] = 100*bound_coverage['Households covered']/total_pop

        # Show some information
        print("------------------")
        print("Bound (coverage):", bound_coverage.loc[0,'Coverage (%)'])
        #print("Bound (coverage):", 100*bound_coverage.loc[0,'Households covered']/total_pop)
        print("Bound (facilities):", None) # bound_facilities
        print("------------------")
    
        # Do the subclusters again from largest to smallest (add a unique cluster number to all clusters)
        df_pop_temp_subclustered.cluster = df_pop_temp_subclustered.subcluster
        df_pop_2, _ = load_data(path_to_pop = df_pop_temp_subclustered, path_to_current_wells = 'Data (from QGIS)/waterwells_points_WD_xy.csv')
        df_pop_2['cluster_OLD'] = c
        df_pop_2.cluster = new_clust
        for sc in df_pop_2.subcluster.unique():
            df_pop_temp = df_pop_2[df_pop_2.subcluster == sc].copy()
            df_pop_temp.cluster = new_clust
            df_pop_clustered_NEW = pd.concat([df_pop_clustered_NEW, df_pop_temp], axis=0)
            new_clust+=1
        
        # Save all the bounds
        bound_coverage_df = pd.concat([bound_coverage_df, bound_coverage], axis=0)
    else:
        time_UB = 0
        df_pop_temp_subclustered = df_pop_temp_subclustered.loc[:,columns_needed+['subcluster']]

        # Do the subclusters again from largest to smallest (add a unique cluster number to all clusters)
        df_pop_temp_subclustered.cluster = df_pop_temp_subclustered.subcluster
        df_pop_2, _ = load_data(path_to_pop = df_pop_temp_subclustered, path_to_current_wells = 'Data (from QGIS)/waterwells_points_WD_xy.csv')
        df_pop_2['cluster_OLD'] = c
        df_pop_2.cluster = new_clust
        for sc in df_pop_2.subcluster.unique():
            df_pop_temp = df_pop_2[df_pop_2.subcluster == sc].copy()
            df_pop_temp.cluster = new_clust
            df_pop_clustered_NEW = pd.concat([df_pop_clustered_NEW, df_pop_temp], axis=0)
            new_clust+=1

    runtimes += [[c, max(df_pop_temp_subclustered.subcluster)+1, time_well_generation, time_SC, time_UB]]
    ############################################################################################################
    ########### Actually solve the subproblems of the clusters and combine with the optimal solution ###########
    ############################################################################################################
    if SOLV_SUBS:
        # Create a pareto front for all subclusters
        optimal_coverage_subclusters = pd.DataFrame(columns = ['Cluster', 'Grid_dist', 'MaxNumberWaterwells', 'Households covered', 'Coverage (%)', 'NewAdd', 'Subcluster'])
        for h in df_pop_temp_subclustered.subcluster.unique():
            print("------------------------------------------")
            print("Optimizing cluster:", h)
            new_wps, df_pop_temp = load_newwps(h, df_pop_2, waterwells, GRID_DIST, buf_opt = MAX_DIST, REALCLUSTER=c, remove_dominated = REMOVE_DOMINATED, save = False, CASENAME = CASENAME+"_SUBCLUSTER", METHOD = 3)
            X,_,df_final = optimize_locations(df_pop_temp, new_wps, current_locations, c, GRID_DIST = GRID_DIST, CHOICE = CHOICE, OBJECTIVE = 'max_cover', MAX_NUMBER_WELLS=MAX_NUMBER_WELLS, MIN_COVER = None, MIP_GAP = MIP_GAP, PLOT_INFO= PLOT_INFO, PLOT_PARETO=PLOT_PARETO, CASENAME=CASENAME+str(h), printall=False)
            df_final['Subcluster'] = h
            optimal_coverage_subclusters = pd.concat([optimal_coverage_subclusters, df_final], axis = 0)
        optimal_coverage_subclusters.reset_index(drop = True, inplace = True)
        optimal_coverage_subclusters = optimal_coverage_subclusters.drop_duplicates(subset =['Cluster','Households covered'], keep = 'first').reset_index(drop=True)
        a = optimal_coverage_subclusters.set_index(['Subcluster', 'MaxNumberWaterwells']).unstack()['Households covered']
        a = a.apply(lambda x: x.fillna(x.max()), axis=1) # fill missing values with the max of the row
        
        # Solve the resource allocation problem.
        start = time.time()
        df_test = solve_selection_model_2(a, sum(df_pop_temp_subclustered.VALUE))
        print(df_test)
        one = time.time()-start
        
        # Load the optimal values to which we want to compare the combined solution
        df = pd.read_csv(FNAME_COMPARE, sep=';') 
        df = df.loc[np.where(df.Cluster == c)[0], 'Households covered']
        print(df)
        
        # Compare both solutions
        lens = min(len(df_test), len(df))
        df_test = df_test.iloc[0:lens, :] # Might need more when not optimal (not interesting for the comparison)
        df_test['optimal'] = list(df)[0:lens]
        df_test['diff'] = df_test.optimal - df_test['Households covered']
        df_test['diff_cov'] = (df_test['diff']/total_pop)*100
        
        # Save a df with the comparisons
        if not os.path.exists("Subclustering"): os.makedirs("Subclustering")
        df_test.to_csv('Subclustering/CombinedClusterSolutions_'+str(c)+'_'+str(MAX_POP)+'_'+str(MIP_GAP)+'_NEW.csv', sep = ";")

if not os.path.exists("Subclustering"): os.makedirs("Subclustering")
pd.DataFrame(runtimes, columns = ['Cluster', 'NumberSC', 'Well_generation', 'SC', 'UB']).to_csv('Subclustering/runtimes_'+str(MAX_POP)+'.csv', sep = ";")
bound_coverage_df.to_csv('Subclustering/bound_coverage_df_'+str(MAX_POP)+'.csv', sep = ";")
df_pop_clustered_NEW.to_csv('Subclustering/df_pop_clustered_NEW_'+str(MAX_POP)+'.csv', sep = ";")
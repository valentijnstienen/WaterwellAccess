import pandas as pd
import geopandas as gpd
import numpy as np
import os

from shapely.geometry import Point, Polygon, MultiPoint
from pandarallel import pandarallel
from math import sqrt
from shapely import wkt
from visualize import visualize_situation
from tqdm import tqdm             
                
# from SETTINGS import *

# Initialize pandarallel with 8 workers
pandarallel.initialize(progress_bar=False, nb_workers = 20)

#TODO DONE
def load_data(path_to_pop, path_to_current_wells):
    """
    Load the clustered population data and the set of current waterwells. Order them, 
    from the largest (most amount of households) to the smallest and geometrize the 
    geometry column. 
    
    Parameters
    ----------
    path_to_pop : path to the clustered population data (came from QGIS)
        string
    path_to_current_wells : path to the csv with the locations of the current waterwells 
        string
    
    Returns
    -------
    df_pop : population dataframe
    waterwells : current waterwell dataframe
    """
    ################################## Population data ####################################
    try: df_pop = pd.read_table(path_to_pop, sep = ";", index_col =0)
    except: df_pop = path_to_pop
    # df_pop.cluster = 8
    df_pop = df_pop.sort_values(by = ['cluster'], inplace = False).reset_index(drop = True)
    print("There are:", len(df_pop.cluster.unique()), "clusters (not ordered).")
    # Order the dataframe such that the largest cluster is computed first
    order_dict = {}
    for i, cluster_num in enumerate(df_pop['cluster'].value_counts().index):
        order_dict[cluster_num] = i
    df_pop['cluster_order'] = df_pop['cluster'].map(order_dict)
    df_pop = df_pop.sort_values(by = ['cluster_order'], inplace = False).reset_index(drop = True)
    # Preprocess other columns
    df_pop['geometry'] = df_pop.apply(lambda r: Point(r.X, r.Y), axis = 1)
    df_pop = gpd.GeoDataFrame(df_pop, geometry = 'geometry', crs = '32634')
    ################################## Current locations ##################################
    waterwells = pd.read_table(path_to_current_wells, sep = ",")
    waterwells['geometry'] = waterwells.apply(lambda r: Point(r.X, r.Y), axis = 1)
    waterwells = gpd.GeoDataFrame(waterwells, geometry = 'geometry', crs = '32634')
    #######################################################################################
    return df_pop, waterwells

#TODO None
def load_newwps(CLUSTER, df_pop, waterwells, GRID_DIST, buf_opt, ADDON='', REALCLUSTER=None, remove_dominated = True, save=True, CASENAME = None, METHOD = 1):
    """
        Load the dataframes that are used for the optimization processes. First, create
        a dataframe with potential waterwell locations. Then also augment the population 
        dataframe with info about their vicinity to wells (needed in the optimizaiton 
        models).
    
    Parameters
    ----------
    CLUSTER : Indicates for which cluster_order [CLUSTER] (from the df_pop), we want 
                to determine the potential waterwell locations. If you want to solve
                for the entire dataset (no clusters), you can specify 'All'. You can 
                also combine multiple clusters by using a list, e.g., [0,2,3].
        float/string/list

    df_pop : dataframe with the clustered population data. 
        dataframe
    
    waterwells : dataframe with the current waterwells, used to indicate which potential 
            waterwells are already open due to their distance from an existing waterwell.
        dataframe
    
    GRID_DIST : distance between potential waterwell locations
        float
    
    ADDON : [optional] You can specify an additional string to the filename to which 
                you save the population and potential waterwell df.
        string
    
    REALCLUSTER : 
    
    save : Indicates whether you want to save or just return the dfs (e.g., in development)
        boolean
    
    Returns
    -------
    new_wps : dataframe with all potential waterwells that should be considered. 
        dataframe
    df_pop_temp : extended dataframe with all the relevant households
        dataframe
    """
    if REALCLUSTER is None: REALCLUSTER=CLUSTER
    try: 
        # Load data if already ran before (and saved)
        def geom_converter(wkt_str):
            return wkt.loads(wkt_str)
        def list_converter(l):
            return eval(l)
        #try:
        new_wps = pd.read_table('Preprocessed data_'+CASENAME+'/PotentialWaterPoints/C'+str(CLUSTER)+ADDON+'.csv', sep=";", index_col=0, converters={'geometry':geom_converter, 'cov_pop_index':list_converter})
        new_wps = gpd.GeoDataFrame(new_wps, geometry='geometry', crs='32634')
        df_pop_temp = pd.read_table('Preprocessed data_'+CASENAME+'/Population data/C_POP_'+str(CLUSTER)+ADDON+'.csv', sep=";", index_col=0, converters={'geometry':geom_converter, 'cov_pop_index':list_converter})
        #except:
        #    BASE_CASENAME = str(GRID_DIST) + "_" + "DOMINANCE_REMOVED_" + str(remove_dominated)
        #    new_wps = pd.read_table('Preprocessed data_'+BASE_CASENAME+'/PotentialWaterPoints/C'+str(CLUSTER)+ADDON+'.csv', sep=";", index_col=0, converters={'geometry':geom_converter, 'cov_pop_index':list_converter})
        #    new_wps = gpd.GeoDataFrame(new_wps, geometry='geometry', crs='32634')
        #    df_pop_temp = pd.read_table('Preprocessed data_'+BASE_CASENAME+'/Population data/C_POP_'+str(CLUSTER)+ADDON+'.csv', sep=";", index_col=0, converters={'geometry':geom_converter, 'cov_pop_index':list_converter})  
        ##########################################################################################
        ############## If you want to visualize the potential water well locations ###############
        ##########################################################################################
        # visualize_situation(CLUSTER, PATH_TO_POPULATION=df_pop_temp, PATH_TO_CURRENT_WATERWELLS=None, PATH_TO_POTENTIAL_WATERWELLS=new_wps, PATH_TO_OPTIMAL_WATERWELLS=None)
        ##########################################################################################
    except:
    
        if CLUSTER != "All": df_pop_temp = df_pop[df_pop.cluster_order==CLUSTER].reset_index(drop=True) #0 -> 887
        else: df_pop_temp = df_pop
        
        ###########################################################################################
        ##################### Generate the potential new water well locaitons #####################
        ###########################################################################################
        # METHOD 1: create a new grid # METHOD 2,3: OLD STUFF
        if METHOD == 1:
            def generate_grid_in_polygon(spacing, polygon, crs_ = 32634):
                ''' 
                    This Function generates evenly spaced points within the given 
                    GeoDataFrame. The parameter 'spacing' defines the distance between 
                    the points in coordinate units. 
                '''
                # Square around the country with the min, max polygon bounds
                minx, miny, maxx, maxy = polygon.bounds
                # Now generate the entire grid
                x_coords = list(np.arange(np.floor(minx), int(np.ceil(maxx)), spacing))
                y_coords = list(np.arange(np.floor(miny), int(np.ceil(maxy)), spacing))
                grid = [Point(x) for x in zip(np.meshgrid(x_coords, y_coords)[0].flatten(), np.meshgrid(x_coords, y_coords)[1].flatten())]
                grid_df = gpd.GeoDataFrame(geometry = grid, crs=crs_)
                # Only use points that are within the specified polygon
                extracted_grid = gpd.clip(grid_df, polygon)
                extracted_grid = extracted_grid.reset_index(drop=True)
                extracted_grid['x'] = extracted_grid.geometry.apply(lambda g: g.x)
                extracted_grid['y'] = extracted_grid.geometry.apply(lambda g: g.y)
                return (extracted_grid)    
            def df_grid_per_clust(ci, df_pop_temp, crs_ = 32634):
                # Filter to the speciifc cluster
                df_filtered = df_pop_temp[df_pop_temp.cluster_order==ci].reset_index(drop=True)
                nr_pop = len(df_filtered)
                # If there is just one household in the cluster, there is just one potential location
                if nr_pop == 1 :
                    cntrd = df_filtered.loc[0, 'geometry']
                    a = gpd.GeoDataFrame({'geometry':[cntrd], 'x':[cntrd.x], 'y':[cntrd.y]})
                    a = a.set_crs(epsg=crs_)
                    return a
                # If there are just two households in the cluster, there is just one potential location
                elif nr_pop == 2 :
                    cntrd = df_filtered.geometry.unary_union.centroid
                    a = gpd.GeoDataFrame({'geometry':[cntrd], 'x': [cntrd.x], 'y': [cntrd.y]})
                    a = a.set_crs(epsg=crs_)
                    return a
                # If there are more household, we create a grid of pontial water well locations
                else:
                    a = df_filtered.geometry.buffer(buf_opt).unary_union
                    return generate_grid_in_polygon(GRID_DIST, a)
            # Loop through all clusters
            new_wps = gpd.GeoDataFrame()
            for ci in df_pop_temp.cluster_order.unique():
                print("Processing cluster:", ci)
                t = df_grid_per_clust(ci, df_pop_temp)
                new_wps = pd.concat([new_wps, t])
            new_wps.reset_index(drop=True, inplace=True)
            """-----------------------------------------------------------------------"""
        elif METHOD == 2:
            """------------- Alternative initialization -------------"""
            #Here we can choose to use another (existing) set of waterwells...
            def geom_converter(wkt_str):
                return wkt.loads(wkt_str)  
            new_wps = gpd.GeoDataFrame()
            for ci in df_pop_temp.cluster_order.unique():
                print("Processing cluster:", ci)
                new_wps_ADD = pd.read_table('Preprocessed data/PotentialWaterPoints/new_wps_'+str(ci)+'_250_SMALL.csv', sep=";", index_col = [0,1], converters={'geometry':geom_converter}).reset_index(drop = True)
                new_wps_ADD = gpd.GeoDataFrame(new_wps_ADD, geometry = 'geometry', crs=crs_)
                new_wps = pd.concat([new_wps, new_wps_ADD])
            new_wps.reset_index(drop=True, inplace=True)
            """------------------------------------------------------"""
        elif METHOD == 3:
            BASE_CASENAME = str(GRID_DIST) + "_" + "DOMINANCE_REMOVED_" + str(remove_dominated)
            # Load data (NEW POTENTIAL WATER LOCATIONS)
            def geom_converter(wkt_str):
                return wkt.loads(wkt_str)
            def list_converter(l):
                return eval(l)
            new_wps = pd.read_table('Preprocessed data_'+BASE_CASENAME+'/PotentialWaterPoints/C'+str(REALCLUSTER)+'.csv', sep=";", index_col=0, converters={'geometry':geom_converter, 'cov_pop_index':list_converter})
            new_wps = gpd.GeoDataFrame(new_wps, geometry='geometry', crs='32634')
        ###########################################################################################
        
        ###########################################################################################
        ############### Add information about which households are in its vicinity ################
        ###########################################################################################
        def find_covering_pop(geo):
            return list(df_pop_temp[df_pop_temp.geometry.within(geo)].index)
        new_wps['cov_pop_index'] = new_wps.geometry.buffer(buf_opt).parallel_apply(find_covering_pop)
        # Remove wells that have no household in its vicintiy (only useful when solving subclusters)
        new_wps = new_wps[new_wps['cov_pop_index'].apply(lambda x: len(x) > 0)].reset_index(drop=True)
        #visualize_situation(CLUSTER, PATH_TO_POPULATION=df_pop_temp, PATH_TO_CURRENT_WATERWELLS=None, PATH_TO_POTENTIAL_WATERWELLS=new_wps, PATH_TO_OPTIMAL_WATERWELLS=None)
        ##########################################################################################
        
        ##########################################################################################
        ######### Remove the dominated potential waterwell locations from the dataframe ##########
        ##########################################################################################
        if remove_dominated: 
            new_wps['lists_set'] = new_wps['cov_pop_index'].apply(set)
            new_wps = new_wps.drop_duplicates(subset=['lists_set'])
            rows_to_remove = set()
            for i, row1 in tqdm(new_wps.iterrows(), total=len(new_wps), desc="Checking dominance"):
                for j, row2 in new_wps.iterrows():
                    if i != j and row1['lists_set'].issubset(row2['lists_set']):
                        if row1['lists_set']!=row2['lists_set']: rows_to_remove.add(i)
                        break
            new_wps = new_wps.drop(rows_to_remove).reset_index(drop=True)
            new_wps = new_wps.drop(columns=['lists_set'])
        #visualize_situation(CLUSTER, PATH_TO_POPULATION=df_pop_temp, PATH_TO_CURRENT_WATERWELLS=None, PATH_TO_POTENTIAL_WATERWELLS=new_wps, PATH_TO_OPTIMAL_WATERWELLS=None)
        ##########################################################################################

        ##########################################################################################
        ############# Find all the current water wells (wuthin 250m of a water well) #############
        ##########################################################################################
        new_wps['Current'] = 0
        d = np.array(new_wps.geometry.parallel_apply(lambda p1: waterwells.geometry.apply(lambda p2: p1.distance(p2))))
        current_locations = np.where(np.any(d < (buf_opt/2), axis=1))[0]
        new_wps.loc[current_locations, 'Current'] = 1
        ##########################################################################################
        
        ##########################################################################################
        ######### Add information about which wells are in the vicinity of households ############
        ##########################################################################################
        # Add the cov info to the POPULATION DATA. This is used in the optimiziation models
        def find_covering_wells(geo):
            return list(new_wps[new_wps.geometry.within(geo)].index)
        df_pop_temp['cov_pop_index'] = df_pop_temp.geometry.buffer(buf_opt).parallel_apply(find_covering_wells)
        ######################################## Save data ########################################
        
        ##########################################################################################
        ################################# Save the dataframes ####################################
        ##########################################################################################
        if save: 
            if not os.path.exists("Preprocessed data_"+CASENAME+"/PotentialWaterPoints"): os.makedirs("Preprocessed data_"+CASENAME+"/PotentialWaterPoints/")
            new_wps.to_csv('Preprocessed data_'+CASENAME+'/PotentialWaterPoints/C'+str(CLUSTER)+ADDON+'.csv', sep = ";")
            if not os.path.exists("Preprocessed data_"+CASENAME+"/Population data"): os.makedirs("Preprocessed data_"+CASENAME+"/Population data/")
            df_pop_temp.to_csv('Preprocessed data_'+CASENAME+'/Population data/C_POP_'+str(CLUSTER)+ADDON+'.csv', sep = ";")
        ###########################################################################################
        
    return new_wps, df_pop_temp
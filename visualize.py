import dash
from dash import html
from dash import dcc
import plotly.graph_objects as go
import pyproj
from shapely.ops import transform
with open('./Licenses/mapbox_accesstoken.txt') as f: mapbox_accesstoken = f.readlines()[0]

from shapely.ops import unary_union
import pandas as pd
import geopandas as gpd
from shapely import wkt
from shapely.geometry import Point, Polygon, MultiPoint

# Model settings
from SETTINGS import *

def visualize_situation(CLUSTER, PATH_TO_POPULATION, PATH_TO_CURRENT_WATERWELLS, PATH_TO_POTENTIAL_WATERWELLS, PATH_TO_OPTIMAL_WATERWELLS):

    transformer = pyproj.Transformer.from_crs('32634', 'EPSG:4326', always_xy=True)

    """--------------------- Population points ---------------------"""
    try: 
        def geom_converter(wkt_str):
            return wkt.loads(wkt_str)
        def list_converter(l):
            return eval(l)
        df_pop = pd.read_table(PATH_TO_POPULATION, sep=";", index_col=0, converters={'geometry':geom_converter, 'cov_pop_index':list_converter})
    except: 
        df_pop = PATH_TO_POPULATION
    df_pop = gpd.GeoDataFrame(df_pop, geometry = 'geometry', crs = '32634')
    boundary = transform(transformer.transform, unary_union(list(df_pop.geometry.buffer(500))))
    df_pop_LATLON = df_pop.to_crs(crs='EPSG:4326')
    lon_coords = df_pop_LATLON['geometry'].apply(lambda p: p.x)
    lat_coords = df_pop_LATLON['geometry'].apply(lambda p: p.y)
    hover_info_pop = df_pop_LATLON.VALUE
    try: colors = df_pop_LATLON['color']
    except: colors = 'orange'
    """------------------------------------------------------------""" 

    """-------------- Potential water well locations --------------"""
    if PATH_TO_POTENTIAL_WATERWELLS is not None:
        try: 
            potential_optimal_waterwell_locations = pd.read_table(PATH_TO_POTENTIAL_WATERWELLS, sep = ";", index_col = 0, converters={'geometry':geom_converter})
        except: 
            potential_optimal_waterwell_locations = PATH_TO_POTENTIAL_WATERWELLS
        potential_optimal_waterwell_locations['geometry'] = potential_optimal_waterwell_locations.apply(lambda r: Point(r.x, r.y), axis = 1)
        potential_optimal_waterwell_locations = gpd.GeoDataFrame(potential_optimal_waterwell_locations, geometry = 'geometry', crs = '32634')
        potential_optimal_waterwell_locations_LATLON = potential_optimal_waterwell_locations.to_crs(crs='EPSG:4326')
        lon_coords_wps = potential_optimal_waterwell_locations_LATLON['geometry'].apply(lambda p: p.x)
        lat_coords_wps = potential_optimal_waterwell_locations_LATLON['geometry'].apply(lambda p: p.y)
        hover_info_ww = "-"
    """------------------------------------------------------------""" 

    """------------- Current water well locations (RAW) -----------"""
    if PATH_TO_CURRENT_WATERWELLS is not None:
        ############################## RAW ###############################
        current_waterwells = pd.read_table(PATH_TO_CURRENT_WATERWELLS, sep = ",")
        current_waterwells['geometry'] = current_waterwells.apply(lambda r: Point(r.X, r.Y), axis = 1)
        current_waterwells = gpd.GeoDataFrame(current_waterwells, geometry = 'geometry', crs = '32634')
        current_waterwells_LATLON = current_waterwells.to_crs(crs='EPSG:4326')
        lon_coords_wps_current = current_waterwells_LATLON['geometry'].apply(lambda p: p.x)
        lat_coords_wps_current = current_waterwells_LATLON['geometry'].apply(lambda p: p.y)
        hover_info_ww_current = "-"
        ########################### COMPUTED #############################
        potential_optimal_waterwell_locations_LATLON_CURRENT = potential_optimal_waterwell_locations_LATLON.loc[potential_optimal_waterwell_locations_LATLON.Current == 1,:]
        lon_coords_wps_current_computed = potential_optimal_waterwell_locations_LATLON_CURRENT['geometry'].apply(lambda p: p.x)
        lat_coords_wps_current_computed = potential_optimal_waterwell_locations_LATLON_CURRENT['geometry'].apply(lambda p: p.y)
        hover_info_ww_current_computed = "-"
    """------------------------------------------------------------"""

    """-------------------- Optimal water wells -------------------"""
    if PATH_TO_OPTIMAL_WATERWELLS is not None:
        try: 
            optimal_waterwells = pd.read_table(PATH_TO_OPTIMAL_WATERWELLS, sep = ";", index_col=0)
        except: 
            optimal_waterwells = PATH_TO_OPTIMAL_WATERWELLS
        #optimal_waterwells = optimal_waterwells[optimal_waterwells.Cluster == CLUSTER].reset_index(drop = True)
        optimal_waterwells['geometry'] = optimal_waterwells.apply(lambda r: Point(r.x, r.y), axis = 1)
        optimal_waterwells = gpd.GeoDataFrame(optimal_waterwells, geometry = 'geometry', crs = '32634')
        optimal_waterwells['covers'] = optimal_waterwells.geometry.buffer(500).apply(lambda x: transform(transformer.transform, x))
        optimal_waterwells = optimal_waterwells.to_crs(crs='EPSG:4326')
        lon_coords_wps_OPTIMAL = optimal_waterwells['geometry'].apply(lambda p: p.x)
        lat_coords_wps_OPTIMAL = optimal_waterwells['geometry'].apply(lambda p: p.y)
        hover_info_ww_OPTIMAL = '-'        
    """------------------------------------------------------------"""

    ######################################################################################################
    ################################### CREATE THE VISUALIZATION #########################################
    ######################################################################################################
    # Create the figure
    fig = go.Figure()

    ######################### Draw basemap (Satellite) ########################
    fig.update_layout(mapbox1 = dict(center = dict(lat=lat_coords.mean(), lon=lon_coords.mean()), accesstoken = mapbox_accesstoken, zoom = 13), margin = dict(t=10, b=0, l=10, r=10),showlegend=False, mapbox_style="satellite")
    ###########################################################################

    ############# Draw the boundary of the cluster (unary union) ##############
    if CLUSTER != 'All':
        try: 
            x, y = boundary.exterior.coords.xy
            fig.add_trace(go.Scattermapbox(mode='lines', lat=y.tolist(), lon=x.tolist(), visible = True,  marker = {'size' : 20, 'color': 'pink', 'allowoverlap': True}))
        except: print("Plotting the boundary is not possible.")
    ###########################################################################
    
    """------------------------ Household locations ------------------------"""
    try: fig.add_trace(go.Scattermapbox(mode='markers', lat=lat_coords, lon=lon_coords, visible = True, text = hover_info_pop, marker = {'size' : 10, 'opacity': 1, 'color': colors, 'allowoverlap': True}))
    except: fig.add_trace(go.Scattermapbox(mode='markers', lat=lat_coords, lon=lon_coords, visible = True, text = hover_info_pop, marker = {'size' : 10, 'opacity': 1, 'color': 'red', 'allowoverlap': True}))
    """---------------------------------------------------------------------"""

    """---------------------------- Water wells ----------------------------"""
    ################# Draw the current water wells (COMPUTED) #################
    if PATH_TO_CURRENT_WATERWELLS is not None: fig.add_trace(go.Scattermapbox(mode='markers', lat=lat_coords_wps_current_computed, lon=lon_coords_wps_current_computed, visible = True, text = hover_info_ww_current_computed, marker = {'size' : 10, 'opacity': 1, 'color': 'blue', 'allowoverlap': True}))
    ###########################################################################
    ################ Draw the considered potential water wells ################
    if PATH_TO_POTENTIAL_WATERWELLS is not None: fig.add_trace(go.Scattermapbox(mode='markers', lat=lat_coords_wps, lon=lon_coords_wps, visible = True, text = hover_info_ww, marker = {'size' : 5, 'opacity': 1, 'color': 'yellow', 'allowoverlap': True}))
    ###########################################################################
    ################### Draw the current water wells (RAW) ####################
    if PATH_TO_CURRENT_WATERWELLS is not None: fig.add_trace(go.Scattermapbox(mode='markers', lat=lat_coords_wps_current, lon=lon_coords_wps_current, visible = True, text = hover_info_ww_current, marker = {'size' : 10, 'opacity': 1, 'color': 'navy', 'allowoverlap': True}))
    ###########################################################################
    """---------------------------------------------------------------------"""

    """------------------------ Optimal water wells ------------------------"""
    ###################### Draw the optimal water wells #######################
    if PATH_TO_OPTIMAL_WATERWELLS is not None: 
        fig.add_trace(go.Scattermapbox(mode='markers', lat=lat_coords_wps_OPTIMAL, lon=lon_coords_wps_OPTIMAL, visible = True, text = hover_info_ww_OPTIMAL, marker = {'size' : 11, 'opacity': 1, 'color': '#02EEF8', 'allowoverlap': True}))
        ########## Draw a cover-circle around each water well #################
        if False: #COVERS_OPTIMAL_WELLS: 
            for polygon in optimal_waterwells['covers']:
                x, y = polygon.exterior.coords.xy
            fig.add_trace(go.Scattermapbox(mode='lines', lat=y.tolist(), lon=x.tolist(), visible = True, fill='toself', marker = {'size' : 20, 'color': 'green','allowoverlap': True}))
    ##########################################################################
    """---------------------------------------------------------------------"""

    # Launch app
    app = dash.Dash(__name__)
    app.layout = html.Div([html.Div(id = 'fig2', children=[dcc.Graph(id='fig',figure=fig, style={"height" : "95vh"})], style={"height" : "80vh"})], className = "container" )
    #if __name__ == '__main__':
    app.run_server(debug=False)
    ######################################################################################################
    ######################################################################################################

if __name__ == "__main__":
    """-------------------------------------------------------------"""
    CLUSTER = 0
    PATH_TO_POPULATION = 'Preprocessed data (FAST)/Population data/C_POP_'+str(CLUSTER)+'_SPARSE.csv'
    PATH_TO_CURRENT_WATERWELLS = None#'Data (from QGIS)/waterwells_points_WD_xy.csv'
    PATH_TO_POTENTIAL_WATERWELLS = 'Preprocessed data (FAST)/PotentialWaterPoints/C'+str(CLUSTER)+'_SPARSE.csv'
    PATH_TO_OPTIMAL_WATERWELLS = 'optimal_locations_GREENFIELD_01.csv'
    if PATH_TO_OPTIMAL_WATERWELLS is not None: 
        COVERS_OPTIMAL_WELLS = False
    """-------------------------------------------------------------"""
    visualize_situation(CLUSTER, PATH_TO_POPULATION, PATH_TO_CURRENT_WATERWELLS, PATH_TO_POTENTIAL_WATERWELLS, PATH_TO_OPTIMAL_WATERWELLS)
    
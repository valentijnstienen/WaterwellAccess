import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import geopandas
import os

from sklearn.cluster import DBSCAN    

def create_shapefile(gdf, filename): 
    a = gdf.X
    attributes = ['X','Y', 'VALUE', 'cluster']
    for attr in attributes:
        try: gdf[attr] = [str(l) for l in gdf[attr]]
        except: a = 1
    gdf.to_file(filename)


df_pop = pd.read_table('Data (from QGIS)/population_points_WD_xy_loc.csv', sep = ",")
print(df_pop)
X = df_pop[['X', 'Y']].to_numpy()

# Run DBSCAN
dbscan = DBSCAN(eps=1000, min_samples=1)
dbscan.fit(X)

labels = dbscan.labels_
df_pop['cluster'] = labels

# Number of clusters in labels, ignoring noise if present.
n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
n_noise_ = list(labels).count(-1)
print(f"Estimated number of clusters: {n_clusters_}")
print(f"Estimated number of noise: {n_noise_}")

print(df_pop)
df_pop_grouped = df_pop.groupby(['cluster']).agg({'cluster': 'count', 'VALUE': 'sum'}).reset_index(drop=True)
df_pop_grouped.sort_values(by = ['cluster'], inplace = True)
print(df_pop_grouped)


# Interesting statistics 
print("----------------------------")
print("Descriptive stats:")
print("Clusters sizes (ordered):",list(df_pop_grouped.cluster))
print(sum(list(df_pop_grouped.cluster)))
print("Number of 1-clusters:", sum(df_pop_grouped.cluster == 1)) # Number of 1-clusters
print("Size of clusters:", "min:",min(df_pop_grouped.cluster), ", mean:",np.mean(df_pop_grouped.cluster), ", median:",np.median(df_pop_grouped.cluster), ", max:", max(df_pop_grouped.cluster))
print("Number of people per cluster:", "min:",min(df_pop_grouped.VALUE), ", mean:",np.mean(df_pop_grouped.VALUE), ", max:", max(df_pop_grouped.VALUE)) # Number of 1-clusters
print("----------------------------")


# Save results
if not os.path.exists("Results"): os.makedirs("Results")
df_pop.to_csv("Results/population_points_WD_xy_loc_CLUSTERED.csv", sep = ";") # Save as csv file
gdf_pop = geopandas.GeoDataFrame(df_pop, geometry=geopandas.points_from_xy(df_pop.X, df_pop.Y)) # Save as shapefile
create_shapefile(gdf_pop, "Results/df_pop_clustered_1000_NEW.shp")

# Visualize the results (BETTER TO PLOT IN QGIS!!!)
plt.title(f"Estimated number of clusters: {n_clusters_}")
plt.scatter(X[:,0], X[:,1], c=labels, cmap='Paired')
plt.show()


import pandas as pd
import time
import numpy as np
import os
import matplotlib.pyplot as plt

from gurobipy import Model, GRB
from load_data import load_data

# load the data
df_pop, waterwells = load_data(path_to_pop = 'Results/population_points_WD_xy_loc_CLUSTERED.csv', path_to_current_wells = 'Data (from QGIS)/waterwells_points_WD_xy.csv')

#
# total_pop = sum(df_pop.VALUE)
#
# df_uncapacitated = pd.read_table('df_final_1e-07_3000_optimal_coverage.csv', sep = ";", index_col = 0)
# df_uncapacitated['diff'] = df_uncapacitated['Households covered'].diff()#/total_pop
# print(df_uncapacitated)
#
#
#
#
# stop
# df_uncapacitated.to_csv('uncap_3000.csv', sep =";")
# df_capacitated = pd.read_table('df_final_1e-07_WITHOUT_DOMINANCE_REMOVED_1000_optimal_coverage.csv', sep = ";", index_col = 0)
# df_capacitated['diff'] = df_capacitated['Households covered'].diff()#/total_pop
# print(df_capacitated)
# df_capacitated.to_csv('cap_1000.csv', sep =";")
#
# stop
# plt.figure()
# # Create a line plot with separate lines for Y1 and Y2
# plt.plot(df_uncapacitated['Coverage (%)'], label='Uncapacitated')
# plt.plot(df_capacitated['Coverage (%)'], label='Capacitated')
# # Add labels and a legend
# plt.ylabel('Coverage (%)')
# plt.legend()
# # Show the plot
# plt.show()
#
#
#
# plt.figure()
# # Create a line plot with separate lines for Y1 and Y2
# plt.plot(df_uncapacitated['diff'], label='Uncapacitated')
# plt.plot(df_capacitated['diff'], label='Capacitated')
# # Add labels and a legend
# plt.ylabel('Coverage diff (%)')
# plt.legend()
# # Show the plot
# plt.show()
#
#
#
# stop



CASENAME = 'CAP_1000_0.001'

##########################################################################
#################### load and format the Pareto fronts ###################
##########################################################################
fname = 'ParetoFronts/'+CASENAME+'_optimal_coverage.csv' #'UNCAP_3000_optimal_coverage.csv'; 'CAP_1000_optimal_coverage.csv'
df = pd.read_csv(fname, sep=';')#WITHOUT_DOMINANCE_REMOVED_optimal_coverage
df = df.drop_duplicates(subset =['Cluster','Households covered'], keep = 'first').reset_index(drop=True)
a = df.set_index(['Cluster', 'MaxNumberWaterwells']).unstack()['Households covered']
a = a.apply(lambda x: x.fillna(x.max()), axis=1) # fill missing values with the max of the row
##########################################################################

##########################################################################
############ Plot a figure with example plots of paretofronts ############
##########################################################################
PLOT_FIGURE = False
if PLOT_FIGURE:
    a_plot = a.iloc[0:10, 0:200]
    a_plot = a_plot.round(0).reset_index(drop=True)
    print(a_plot)
    # Create a function to plot all line charts in the same figure
    def plot_all_line_charts(df):
        plt.figure(figsize=(12, 6))  # Create a single figure
    
        for index, row in a_plot.iterrows():
            # Filter out NaN values and convert the row to a list
            values = row.dropna().tolist()
        
            # Create a range of x-values based on the number of non-NaN values
            x = range(1, len(values) + 1)
        
            # Plot the line chart for the current row
            plt.plot(x, values, label=f'Cluster {index}')  # Add a label for each line

        # Customize the chart appearance and labels
        plt.title("Line Charts for Each Cluster")
        plt.xlabel("X-axis label")
        plt.ylabel("Y-axis label")
        plt.legend()  # Display legends for each line
    # Assuming your DataFrame is named 'df'
    plot_all_line_charts(a)
    plt.show()  # Display the single figure with all line charts
    a_plot.T.to_csv('paretofronts.csv', sep = ";")
##########################################################################

##########################################################################
########################### Create a new model ###########################
##########################################################################
print("_____________________")
print("Activating licence...")
os.environ['GRB_LICENSE_FILE'] = 'Licenses/gurobi.lic'
print("_____________________")
def solve_selection_model_2(a, total_pop):
    m = Model("water_well_allocation")

    # Solve until this optimality gap
    MIP_GAP = 0.0000001

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
        mip_gap = m.MIPGap
        coverage = np.round(100*(obj_val/total_pop), 2)
        
        obj_val_array.append([threshold, obj_val, coverage, rt, mip_gap])#, new_add])#,list(Xvalues),list(Yvalues)])
            
        if True:
            ############################################ Print output ############################################
            print("    Objective value:", obj_val, "households (", str(mip_gap)+" )." )
            print("    Coverage:", coverage, "%")
            print("    Opening", threshold, "water wells.")
            print("    Runtime", rt, "seconds.")
            print("_____________________")
            ######################################################################################################
        if threshold%20 == 0:
            df_final = pd.DataFrame(obj_val_array, columns = ['MaxNumberWaterwells', "Households covered", "Coverage (%)", 'Runtime', 'MIPGap'])#, "NewAdd"])
            if not os.path.exists("TotalCoverageSolutions"): os.makedirs("TotalCoverageSolutions")
            df_final.to_csv('TotalCoverageSolutions/df_final_'+CASENAME+'.csv', sep = ";")
        if coverage > 100-(MIP_GAP*100): break
    df_final = pd.DataFrame(obj_val_array, columns = ['MaxNumberWaterwells', "Households covered", "Coverage (%)", 'Runtime', 'MIPGap'])#, "NewAdd"])
    df_final.to_csv('TotalCoverageSolutions/df_final_'+CASENAME+'.csv', sep = ";")
    return df_final
    
df_final = solve_selection_model_2(a, sum(df_pop.VALUE))
print(df_final)
##########################################################################


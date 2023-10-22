import geopandas as gpd
from gurobipy import *
import itertools
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import time
from gurobipy import *
from math import *
from pandarallel import pandarallel
from shapely.geometry import Point, LineString, MultiPolygon, Polygon
from shapely import wkt
from progress.bar import Bar
print("_____________________")
print("Activating licence...")
os.environ['GRB_LICENSE_FILE'] = 'Licenses/gurobi.lic'
print("_____________________")
def optimize_locations(df_pop, new_wps, current_locations, CLUSTER, GRID_DIST, CHOICE, OBJECTIVE, MAX_NUMBER_WELLS, MIN_COVER, MIP_GAP, PLOT_INFO, PLOT_PARETO, CASENAME, printall=True):
    ################################################################################################
    ########################## Model settings that are used more often #############################
    ################################################################################################
    pop_arr = df_pop.VALUE.to_numpy()
    E = df_pop.cov_pop_index.to_dict()
    ################################################################################################
    
    GREEDY = False

    ################################################################################################
    ################################################################################################
    def getvariables(X, Z):
        Xvalues = np.zeros(len(new_wps))
        for i in range(len(new_wps)):
            Xvalues[i]=X[i].x
        Zvalues = np.zeros(len(df_pop))
        for i in range(len(df_pop)):
            Zvalues[i]=Z[i].x
        return(Xvalues, Zvalues)
    if CHOICE == 'maxcover':
        M = Model()
        start = time.time()
        # Decision variables
        X = M.addVars(len(new_wps), vtype=GRB.BINARY) # Whether a facility is opened at each location
        Z = M.addVars(len(df_pop), vtype=GRB.BINARY) # If demand node (population point) is served
        if OBJECTIVE == "min_number_wells": NUMBER_WELLS = M.addVar(vtype=GRB.INTEGER)
        
        # Make sure that every region has a coverage of at least 50%
        # if OBJECTIVE == "max_cover":
        #     for l in df_pop.Loc_En.unique():
        #         pop_loc = df_pop[df_pop.Loc_En == l]
        #         total_pop =  sum(pop_loc.VALUE)
        #         s =M.addConstr(quicksum(Z[j]*pop_loc.VALUE[j] for j in list(pop_loc.index))/total_pop >= 0.5)
            
        # Currently open facilities
        c = M.addConstrs((X[i] == 1 for i in current_locations))

        # Limit number of facilities located
        if OBJECTIVE == "max_cover": s = M.addConstr(X.sum() <= MAX_NUMBER_WELLS[0]+len(current_locations))
        elif OBJECTIVE == "min_number_wells": M.addConstr(X.sum() <= NUMBER_WELLS)

        # Minimum coverage we want to achieve    
        if OBJECTIVE == "min_number_wells": s = M.addConstr(LinExpr((df_pop.VALUE/sum(df_pop.VALUE)), Z.select('*')) >= MIN_COVER[0]/100)

        # Limit number of waterwells a household is connected to, let a household only connect to an opened facility
        M.addConstrs((Z[i] <= (quicksum(X[j] for j in E[i]))) for i in range(len(df_pop)))

        # Define the objective function
        if OBJECTIVE == "max_cover":
            obj = LinExpr(df_pop.VALUE, Z.select('*')) #gp.LinExpr(100*(df.VALUE/sum(df.VALUE)), Z.select('*'))
            M.setObjective(obj, GRB.MAXIMIZE)
        elif OBJECTIVE == "min_number_wells": 
            obj = NUMBER_WELLS
            M.setObjective(obj, GRB.MINIMIZE)

        # Set parameters 
        M.setParam('OutputFlag',printall) # Suppress output during optimization
        M.Params.mipgap = MIP_GAP   #0.01

        # Print runtime
        if printall: print(time.time()-start, "seconds passed! (model set-up)")
        
        # Iterate over multiple optimization problems
        if OBJECTIVE == "max_cover":
            cur_loc_start = len(current_locations)
            obj_val_array = []
            for max_ww in MAX_NUMBER_WELLS:
                start = time.time()
                print("Optimizing for", max_ww, 'waterwells.')
                M.remove(s)
                s = M.addConstr(X.sum() <= max_ww + cur_loc_start)
                if GREEDY:
                    M.remove(c)
                    c = M.addConstrs((X[i] == 1 for i in current_locations))
                M.optimize()
                obj_val = M.objVal
                Xvalues, Zvalues = getvariables(X, Z)
                coverage = np.round(100*(obj_val/sum(df_pop.VALUE)), 2)
                rt = time.time()-start
                
                if GREEDY:
                    current_locations_NEW = np.where(Xvalues>0)[0]
                    new_add = list(set(current_locations_NEW) ^ set(current_locations))[0]
                    current_locations = current_locations_NEW
                else: new_add = None

                
                obj_val_array.append([CLUSTER, GRID_DIST, max_ww, obj_val, coverage, new_add])#,list(Xvalues),list(Yvalues)])
                if PLOT_INFO:
                    ############################################ Print output ############################################
                    print("    Objective value:", obj_val, "households.")
                    print("    Coverage:", coverage, "%")
                    print("    Opening", max_ww, "water wells.")
                    print("    Runtime", rt, "seconds.")
                    print("_____________________")
                    ######################################################################################################
                
                if max_ww%20 == 0: 
                    df_final = pd.DataFrame(obj_val_array, columns = ['Cluster', 'Grid_dist', 'MaxNumberWaterwells', "Households covered", "Coverage (%)", "NewAdd"])
                    df_final.to_csv('df_final_'+CASENAME+'.csv', sep = ";")
                if coverage > 100-(MIP_GAP*100): break #return X
            
            df_final = pd.DataFrame(obj_val_array, columns = ['Cluster', 'Grid_dist', 'MaxNumberWaterwells', "Households covered", "Coverage (%)", "NewAdd"])
            df_final.to_csv('df_final_'+CASENAME+'.csv', sep = ";")
             
            if PLOT_PARETO:
                import matplotlib.pyplot as plt
                fig = plt.Figure()
                plt.plot(df_final['MaxNumberWaterwells'], df_final['Coverage (%)'])
                plt.xlabel('Maximum number of wells to be placed (budget)')
                plt.ylabel('Maximum coverage achieved (%)')
                plt.show()
        elif OBJECTIVE == "min_number_wells":
            obj_val_array = []
            for min_cov in MIN_COVER:
                #start = time.time()
                if printall: print("Optimizing for a minimal coverage of", min_cov, '%.')
                M.remove(s)
                s = M.addConstr(LinExpr((df_pop.VALUE/sum(df_pop.VALUE)), Z.select('*')) >= min_cov/100)
        
                M.optimize()
                obj_val = M.objVal
                Xvalues, Zvalues = getvariables(X, Z)
                actual_coverage = np.round((sum(df_pop.VALUE[list(np.where(Zvalues>0)[0])])/sum(df_pop.VALUE))*100, 2)
                obj_val_array.append([CLUSTER, GRID_DIST, min_cov, actual_coverage, obj_val])
            
                if PLOT_INFO:
                    ############################################ Print output ############################################
                    print("    Objective value:", int(obj_val), "wells.")
                    print("    Coverage:", actual_coverage, "%")
                    print("_____________________")
                    ######################################################################################################
            df_final = pd.DataFrame(obj_val_array, columns = ['Cluster', 'Grid_dist', 'MinCover', "Coverage (%)", "Amount of wells needed"])
            if PLOT_PARETO:
                import matplotlib.pyplot as plt
                fig = plt.Figure()
                plt.plot(df_final["MinCover"], df_final["Amount of wells needed"])
                plt.xlabel('Minimum cover (%)')
                plt.ylabel('Amount of wells needed')
                plt.show()
        
        # Print the final dataframe
        if printall: print(df_final)
        if printall: print(time.time()-start, "seconds passed.")
        return Xvalues, obj_val, df_final
    ################################################################################################
    ################################################################################################

    ################################################################################################
    ################################################################################################
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
    if CHOICE == 'standardFLP':
        start = time.time()
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
        if OBJECTIVE == "min_number_wells": NUMBER_WELLS = M.addVar()#vtype=GRB.INTEGER)
        if printall: print(time.time()-s, "seconds passed.")
        
        # Currently open facilities
        c = M.addConstrs((X[i] == 1 for i in current_locations))
        
        # Limit number of facilities located
        if OBJECTIVE == "max_cover": h = M.addConstr(X.sum() <= MAX_NUMBER_WELLS[0]+len(current_locations))
        elif OBJECTIVE == "min_number_wells": M.addConstr(X.sum() <= NUMBER_WELLS)

        # Minimum coverage we want to achieve    
        if OBJECTIVE == "min_number_wells": h = M.addConstr(quicksum(pop_arr[i]*Y[i,f] for i in range(len(df_pop)) for f in range(len(new_wps)))/sum(df_pop.VALUE) >= MIN_COVER[0]/100)

        # Make sure that y[i,f]=0 if there is no well at f)
        s = time.time()
        bar = Bar('Processing', max=len(new_wps))
        for f in range(len(new_wps)):
            M.addConstr(quicksum(Y[i,f] for i in range(len(df_pop))) <= len(df_pop) * X[f])
            bar.next()
        bar.finish()
        if printall: print(time.time()-s, "seconds passed.")
        
        # Capacity constraint
        s = time.time()
        bar = Bar('Processing', max=len(new_wps))
        for f in range(len(new_wps)):
            M.addConstr(quicksum(Y[i,f]*pop_arr[i] for i in range(len(df_pop))) <= 600)
            bar.next()
        bar.finish()
        if printall: print(time.time()-s, "seconds passed.")

        # print('Making sure that i can only be assigned to one f')
        s = time.time()
        bar = Bar('Processing', max=len(df_pop))
        for i in range(len(df_pop)):
            M.addConstr(quicksum(Y[i,f] for f in range(len(new_wps))) <= 1)
            bar.next()
        bar.finish()
        if printall: print(time.time()-s, "seconds passed.")

        # Define the objective function
        if OBJECTIVE == "max_cover":
            obj = quicksum(pop_arr[i]*Y[i,f] for i in range(len(df_pop)) for f in range(len(new_wps)))
            M.setObjective(obj, GRB.MAXIMIZE)
        elif OBJECTIVE == "min_number_wells": 
            obj = NUMBER_WELLS
            M.setObjective(obj, GRB.MINIMIZE)

        # Set parameters 
        M.setParam('OutputFlag', True) # Suppress output during optimization
        M.Params.mipgap = MIP_GAP   #0.01

        # Print runtime
        if printall: print(time.time()-start, "seconds passed! (model set-up)")

            # Iterate over multiple optimization problems
        if OBJECTIVE == "max_cover":
            cur_loc_start = len(current_locations)
            obj_val_array = []
            for max_ww in MAX_NUMBER_WELLS:
                start = time.time()
                print("Optimizing for", max_ww, 'waterwells.')
                M.remove(h)
                h = M.addConstr(X.sum() <= max_ww + cur_loc_start)
                M.optimize()
                obj_val = M.objVal
                Xvalues, Yvalues = getvariables(X, Y)
                coverage = np.round(100*(obj_val/sum(df_pop.VALUE)), 2)
                rt = time.time()-start
                
                obj_val_array.append([CLUSTER, GRID_DIST, max_ww, obj_val, coverage])#, new_add])#,list(Xvalues),list(Yvalues)])
                if PLOT_INFO:
                    ############################################ Print output ############################################
                    print("    Objective value:", obj_val, "households.")
                    print("    Coverage:", coverage, "%")
                    print("    Opening", max_ww, "water wells.")
                    print("    Runtime", rt, "seconds.")
                    print("_____________________")
                    ######################################################################################################
                
                if max_ww%20 == 0: 
                    df_final = pd.DataFrame(obj_val_array, columns = ['Cluster', 'Grid_dist', 'MaxNumberWaterwells', "Households covered", "Coverage (%)"])#, "NewAdd"])
                    df_final.to_csv('df_final_'+CASENAME+'_FLP_TEST.csv', sep = ";")
                if coverage > 100-(MIP_GAP*100): break
            
            df_final = pd.DataFrame(obj_val_array, columns = ['Cluster', 'Grid_dist', 'MaxNumberWaterwells', "Households covered", "Coverage (%)"])#, "NewAdd"])
            df_final.to_csv('df_final_'+CASENAME+'_FLP_TEST.csv', sep = ";")

            if PLOT_PARETO:
                import matplotlib.pyplot as plt
                fig = plt.Figure()
                plt.plot(df_final['MaxNumberWaterwells'], df_final['Coverage (%)'])
                plt.xlabel('Maximum number of wells to be placed (budget)')
                plt.ylabel('Maximum coverage achieved (%)')
                plt.show()
        elif OBJECTIVE == "min_number_wells":
            obj_val_array = []
            for min_cov in MIN_COVER:
                start = time.time()
                print("Optimizing for a minimal coverage of", min_cov, '%.')
                M.remove(h)
                h = M.addConstr(quicksum(pop_arr[i]*Y[i,f] for i in range(len(df_pop)) for f in range(len(new_wps)))/sum(df_pop.VALUE) >= min_cov/100)
        
                M.optimize()
                obj_val = M.objVal
                Xvalues, Yvalues = getvariables(X, Y)
                actual_coverage = np.round((sum(pop_arr[i]*Yvalues[i,f] for i in range(len(df_pop)) for f in range(len(new_wps)))/sum(df_pop.VALUE))*100, 2)

                print(time.time()-start, "seconds passed!")
                obj_val_array.append([CLUSTER, GRID_DIST, min_cov, actual_coverage, obj_val])#,list(Xvalues),list(Yvalues)])
                if PLOT_INFO:
                    ############################################ Print output ############################################
                    print("    Objective value:", int(obj_val), "wells.")
                    print("    Coverage:", actual_coverage, "%")
                    print("_____________________")
                    ######################################################################################################
            df_final = pd.DataFrame(obj_val_array, columns = ['Cluster', 'Grid_dist', 'MinCover', "Coverage (%)", "Amount of wells needed"])
            if PLOT_PARETO:
                import matplotlib.pyplot as plt
                fig = plt.Figure()
                plt.plot(df_final["MinCover"], df_final["Amount of wells needed"])
                plt.xlabel('Minimum cover (%)')
                plt.ylabel('Amount of wells needed')
                plt.show()
        
        return Xvalues, obj_val, df_final
    ################################################################################################
    ################################################################################################


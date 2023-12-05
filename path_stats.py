import numpy as np
import pandas as pd
import math
import skgstat as skg
from scipy.stats import wasserstein_distance
from skimage.metrics import structural_similarity

import sgs_plts
import sgs_preprocess


types = {"grid": "Grid", "griddiag": "Diagonal Grid", "horiz": "Horizontal",
         "vert": "Vertical", "negdiag": "Negative Diagonal", "posdiag": "Positive Diagonal"}
locations = {"GL": "Greenland", "PIG": "PIG"}


if __name__ == '__main__':

    print("\nStarting Topographic Map Comparison\n")

    loc_list = ["GL", "PIG"]
    path_list = ["vert", "horiz", "grid", "posdiag", "negdiag", "griddiag"]
    sim = ["1", "2"]

    stat_df = pd.DataFrame(columns = ['Path', 'Loc', 'Sim #', 'MSE', 'Wass Dist', 'SSIM'])
    
    

    for loc in loc_list:

        print(f"Location: {locations[loc]}\n")

        if loc == "GL":
            x = 'X'; y = 'Y'; z = 'Bed'
        else:
            x = 'x'; y = 'y'; z = 'bedrock_altitude (m)'

        true_path_file = 'Data/Real/' +  locations[loc] + '_data.csv'
        true_path = pd.read_csv(true_path_file)
        true_path,_,_ =sgs_preprocess.grid_data(true_path, np.min(true_path[x]), np.max(true_path[x]),
                                  np.min(true_path[y]), np.max(true_path[y]), 500, x, y, z)

        for path in path_list: 

            print(f"\t Path: {types[path]}")

            for i in sim:

                print(f"\t\t Simulation: {i}")

                if i == '1':
                    alt = '2'
                else:
                    alt = '1'

                sim_file = "Output/Simulated/" + loc + "_" + path + "_sim" + i + ".csv"
                other_file = "Output/Simulated/" + loc + "_" + path + "_sim" + i + ".csv"
                GT_file = "Output/Real/" + locations[loc] + "_GT_" + i + ".csv"
                diffmap_file = "Output/" + loc + "_" + path + "_diff" + i + ".png"
                alt_path = 'Data/Simulated/' +  locations[loc] + '/' + loc + '_' + path + '_' + i + '.csv'

                sim_df = pd.read_csv(sim_file)
                other_df = pd.read_csv(other_file)
                GT_df = pd.read_csv(GT_file)
                alt_path = pd.read_csv(alt_path)

                combined = pd.concat([true_path, alt_path], axis=0)
                overlap = len(combined[[x,y]]) - len(combined[[x,y]].drop_duplicates())

                # Obtain comparison metrics
                res = np.abs(sim_df[z].to_numpy() - GT_df[z].to_numpy())
                sse = np.sum(res**2)
                mse = (sse / len(sim_df))
                mean = np.mean(res)
                max = np.max(res)

                # Fancy image comparison metrics 
                wass_dist = wasserstein_distance(sim_df[z].to_numpy(), GT_df[z].to_numpy())
                range = np.max(sim_df[z]) - np.min(sim_df[z])
                ssim = structural_similarity(sim_df[z].to_numpy(), GT_df[z].to_numpy(), data_range=range)

                # Obtain difference map
                # sgs_plts.diff_graph(sim_df, (sim_df[z] - GT_df[z]), 500, x, y, diffmap_file, path, other_df[z])

                # Add row to dataframe
                stat_df.loc[len(stat_df.index)] = [path, loc, i, mse, wass_dist, ssim]
                
                print(f"\t\t Completed!\n")
                
    
    stat_df.to_csv('Output/real.csv')


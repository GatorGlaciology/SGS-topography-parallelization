import sgs_preprocess
import sgs_alg
import sgs_plts

import numpy as np
import pandas as pd
from pathlib import Path
import sys
import os


if __name__ == '__main__':

    # retrieve user parameters
    file_name, x, y, z, xmin, xmax, ymin, ymax, res, num_realizations = sgs_preprocess.menu()
    
    # read data from input file
    df_bed = pd.read_csv(file_name)
    
    # grid data
    df_data, grid_matrix, df_nan = sgs_preprocess.grid_data(df_bed, xmin, xmax, ymin, ymax, res, x, y, z)
    
    # normal score transformation of bed elevation
    df_data.loc[:,'Norm_Bed'], nst_trans = sgs_preprocess.nscore(df_data, z)
    
    # adaptive clustering
    max_pts = 100           # maximum number of points in each cluster
    min_len = 50000         # minimum side length of squares
    df_data, i = sgs_preprocess.adaptive_partitioning(df_data, xmin, xmax, ymin, ymax, max_pts, min_len)
    
    # get number of processes to use
    processes = int(os.cpu_count())
    
    # get variograms for each cluster in parallel
    max_lag = 30000         # maximum lag distance
    n_lags = 100            # number of bins
    gamma = sgs_preprocess.get_variograms(df_data, n_lags, max_lag, processes)
    
    for i in range(num_realizations):

        print(f'-----------------------------------------')
        print(f'\tStarting Realization #{i+1}\n')
    
        # shuffle df of points to simulate (random path)
        df_nan = sgs_preprocess.shuffle_pred_grid(df_nan)
    
        # get kriging weights in parallel
        max_num_nn = 50     # maximum number of nearest neighbors
        rad = 30000         # search radius
        kr_dictionary = sgs_alg.kriging_weights(df_data, df_nan, gamma, rad, max_num_nn, res, processes, x, y, 'Norm_Bed', 'cluster')
    
        # sequential gausian simulation
        data_xyzk, pred_xyzk = sgs_alg.sgs_pred_Z(kr_dictionary, df_data, df_nan, gamma, x, y, 'Norm_Bed', 'cluster')
    
        # concatenate data frames
        df_sim = sgs_alg.concat(data_xyzk, pred_xyzk)

        #reverse normal score transformation
        tmp = df_sim['Norm_Bed'].values.reshape(-1,1)
        df_sim[z] = nst_trans.inverse_transform(tmp)

        # save dataframe to csv
        filepath = Path(f'Output/sim_{i+1}.csv')
        filepath.parent.mkdir(parents=True, exist_ok=True)
        df_sim.to_csv(filepath, index=False)
        
        # output graph
        sgs_plts.plt_graph(df_sim, df_bed, res, x, y, z, i)
        
    sys.exit()

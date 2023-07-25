import sgs_preprocess
import sgs_alg
import sgs_plts
import sgs_serial as gs

from sklearn.preprocessing import QuantileTransformer
import skgstat as skg

import numpy as np
import pandas as pd
from pathlib import Path
import sys
import os
import time
import csv

def run_test_p(res, p):

    file_name = 'Data/PIG_data.csv'
    x = 'x'
    y = 'y'
    z = 'bedrock_altitude (m)'
    xmin = None
    xmax = None
    ymin = None
    ymax = None
    num_realizations = 1
    
    start = time.time()

    # read data from input file
    df_bed = pd.read_csv(file_name)
    
    # grid data
    df_data, grid_matrix, df_nan = sgs_preprocess.grid_data(df_bed, xmin, xmax, ymin, ymax, res, x, y, z)
    
    missing_pts = len(df_nan)
    
    # normal score transformation of bed elevation
    df_data.loc[:,'Norm_Bed'], nst_trans = sgs_preprocess.nscore(df_data, z)
    
    # adaptive clustering
    max_pts = 100           # maximum number of points in each cluster
    min_len = 50000         # minimum side length of squares
    df_data, i = sgs_preprocess.adaptive_partitioning(df_data, xmin, xmax, ymin, ymax, max_pts, min_len)
    
    processes = p
    
    start_sgs = time.time()
    
    # get variograms for each cluster in parallel
    max_lag = 30000         # maximum lag distance
    n_lags = 100            # number of bins
    gamma = sgs_preprocess.get_variograms(df_data, n_lags, max_lag, processes)
    
    
    for i in range(num_realizations):
    
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
        
    
    end = time.time()

    return (end - start), (end - start_sgs), missing_pts
    
def run_test_s(res):
    
    df_bed = pd.read_csv('Data/PIG_data.csv')
    
    start = time.time()
    
    # grid data to resolution and remove coordinates with NaNs
    df_grid, grid_matrix, rows, cols = gs.Gridding.grid_data(df_bed, 'x', 'y', 'bedrock_altitude (m)', res)
    df_grid = df_grid[df_grid["Z"].isnull() == False]
    df_grid = df_grid.rename(columns = {"Z": "bedrock_altitude (m)"})
    
    # normal score transformation
    data = df_grid['bedrock_altitude (m)'].values.reshape(-1,1)
    nst_trans = QuantileTransformer(n_quantiles=500, output_distribution="normal").fit(data)
    df_grid['Norm_Bed'] = nst_trans.transform(data)
    
    # max_points is the most important parameter
    max_points = 800
    min_length = 25000
    max_iter = None

    # initialze parms for full dataset
    xmin = df_grid.X.min(); xmax = df_grid.X.max()
    ymin = df_grid.Y.min(); ymax = df_grid.Y.max()

    i = 0

    # initialize cluster column with NaNs to have zero-indexed
    df_grid['K'] = np.full(df_grid.shape[0], np.nan)

    # begin adaptive partioning
    df_grid, i = gs.adaptive_partitioning(df_grid, xmin, xmax, ymin, ymax, i, max_points, min_length, max_iter)
    
    start_sgs = time.time()
    
    # experimental variogram parameters
    maxlag = 30_000
    n_lags = 70
    
    clusters, counts = np.unique(df_grid.K, return_counts=True)

    variograms = []

    for k in clusters:
        tmp = df_grid[df_grid.K == k]
        coords = tmp[['X', 'Y']].values
        values = tmp['Norm_Bed']
        variograms.append(skg.Variogram(coords, values, bin_func='even', n_lags=n_lags, maxlag=maxlag, normalize=False))
                    

    Pred_grid_xy = gs.Gridding.prediction_grid(xmin, xmax, ymin, ymax, res)
    
    # make a dataframe with variogram parameters
    azimuth = 0
    nug = 0 # nugget effect

    # define variograms for each cluster and store parameters
    # Azimuth, nugget, major range, minor range, sill
    varlist = [[azimuth,
                nug,
                var.parameters[0],
                var.parameters[0],
                var.parameters[1]] for var in variograms]

    df_gamma = pd.DataFrame({'Variogram': varlist})
    
    # simulate

    k = 100               # number of neighboring data points used to estimate a given point
    rad = 50000           # 50 km search radius

    sgs = gs.Interpolation.cluster_sgs(Pred_grid_xy, df_grid, 'x', 'y', 'Norm_Bed', 'K', k, df_gamma, rad)
    
    end = time.time()
    
    return (end - start), (end - start_sgs)
    


if __name__ == '__main__':

    header = ['res', 'num missing pts', 'tot time', 'sgs time', 'type']
    res = [500,500] # List of test resolutions
    
    with open('Test/almostDoneA.csv', 'w', encoding='UTF8', newline='') as f:
        
        writer = csv.writer(f)
        
        writer.writerow(header)
        
        for i in range(2):
        
            t, sgs_t, missing_pts = run_test_p(res[i], 8)
            data = [res[i], missing_pts, t, sgs_t, 'Parallel']
            writer.writerow(data)
            
            t, sgs_t = run_test_p(res[i], 1)
            data = [res[i], missing_pts, t, sgs_t, 'Serisal']
            writer.writerow(data)
            
    sys.exit()

import multiprocessing as mp
import numpy as np
import pandas as pd
import numpy.linalg as linalg
from sklearn.metrics import pairwise_distances
import math
import itertools
import time
import random
    
    
###################################

#   Parallelization Functions

###################################
    
    
def kriging_weights(df_data, df_nan, gamma, rad, max_num_nn, res, processes, xx, yy, zz, cluster):
    """
    Prepares data, calls function to be executed in parallel, and returns kriging weights
    Inputs:
        df_data - DataFrame with observed bed elevation values
        df_nan - DataFrame of points to simulate
        gamma - dictionary of variogram parameters for each cluster
        max_num_nn - maximum number of nearest neighbors
        rad - radius for nearest neighbor search
        res - grid cell resolution
        process - number of processes
        xx - column name for x coordinates of input data frame
        yy - column name for y coordinates of input data frame
        zz - column for z values (or data variable) of input data frame
        cluster - column name for cluster assigned to location
    Outputs:
        kr_dictionary - dictionary with kriging weights and data for SGS
    """
    print(f'Parallel simulation with {processes} processes...')
    
    df_data = df_data.rename(columns = {xx: "X", yy: "Y", zz: "Z", cluster: "K"})
    df_nan = df_nan.rename(columns = {xx: "X", yy: "Y", zz: "Z", cluster: "K"})
    
    # dataframe of conditional and simulated data
    all_xyk = pd.concat([df_data, df_nan])
    all_xyk = all_xyk[['X','Y','K']]
    offset = len(df_data)
    
    # create iterable parameter list
    i = [i for i in range(len(df_nan))]
    args = zip(i, itertools.cycle([all_xyk]), itertools.cycle([gamma]), itertools.cycle([rad]), itertools.cycle([max_num_nn]), itertools.cycle([offset]))
    
    pool = mp.Pool(processes)
    start = time.time()
    kr_dictionary = {}
    out = pool.starmap(parallel_calc, args, chunksize=200)
    
    for i, (idx, weights, covariance_array, nearest_indices, cluster_num) in enumerate(out, start=1):
        
        # aggregate output into a dictionary to look up data by index
        kr_dictionary[idx] = (weights, covariance_array, nearest_indices, cluster_num)
    
    print(f'\t{round((time.time()-start), 2)} seconds to complete \n')

    return kr_dictionary


def parallel_calc(i, all_xyk, gamma, rad, max_num_nn, offset):
    """
    Function that is executed in parallel
    Inputs:
        i - ordered index of current point to simulate from df_nan
        all_xyk - dataframe of conditional and simulated data
        gamma - dictionary of variogram parameters for each cluster
        rad - radius for nearest neighbor search
        max_num_nn - maximum number of nearest neighbors
        offset - size of observational DataFrame, used for indexing
    Outputs:
        pred_xy_index - unordered index of point to simulate (unordered due to shuffing step)
        k_weights - kriging weights for bed elevation values
        covariance_array - covariance between data and unknown
        nearest_indicies - indicies of nearest neigbors
        cluster_num - cluster value assigned to current point
    """
    # offset in all_xyk of location to simulate
    curr_offset = offset + i
    
    pred_xy_index = all_xyk.iloc[curr_offset].name
    pred_xy = all_xyk.iloc[curr_offset,0:2]

    # filter from all data in grid to include only observed and previously simulated locations
    all_pts_before = all_xyk.iloc[:curr_offset]

    nearest, cluster_num, nearest_indices = NNS_cluster(all_pts_before, rad, max_num_nn, pred_xy)
    vario = gamma[cluster_num]
    k_weights, covariance_array = kriging(vario, nearest, pred_xy)

    return pred_xy_index, k_weights, covariance_array, nearest_indices, cluster_num


##############################

#   Get Nearest Neighbors

##############################


def NNS_cluster(all_pts_before, rad, max_num_nn, loc):
    """
    Nearest neighbor octant search
    Inputs:
        radius - search radius
        max_num_nn - maximum number of points to search for
        loc - coordinates for grid cell of interest
        all_pts_before - all observed data and simulated locations before current
    Outputs:
        near - nearest neighbor locations
        K - cluster value assigned to current point
        nearest_indices - indicies of nearest neigbors
    """
    locx = loc.iloc[0]
    locy = loc.iloc[1]
    data = all_pts_before.copy()
    centered_array = center(data['X'].values, data['Y'].values, locx, locy)
    data["angles"] = np.arctan2(centered_array[0], centered_array[1])
    data["dist"] = np.linalg.norm(centered_array, axis=0)
    
    # scales search radius if too small to find nearest neigbors
    rad = min_rad(data, rad)

    data = data[data.dist < rad]
    data = data.sort_values('dist', ascending = True)
    
    # assign simulated data to random cluster in radius
    rand_K = data[~np.isnan(data['K'])]['K'].values
    K = random.choice(rand_K)
    
    bins = [-math.pi, -3*math.pi/4, -math.pi/2, -math.pi/4, 0, math.pi/4, math.pi/2, 3*math.pi/4, math.pi]
    data["Oct"] = pd.cut(data.angles, bins = bins, labels = list(range(8)))
    
    # number of points to look for in each octant, if not fully divisible by 8, round down
    oct_count = max_num_nn // 8
    smallest = np.ones(shape=(max_num_nn, 2)) * np.nan
    nearest_indices = []
    
    for i in range(8):
    
        octant = data[data.Oct == i].iloc[:oct_count][['X','Y']]
        
        for j, row in enumerate(octant.itertuples()):
        
            smallest[i*oct_count+j,:] = [row.X, row.Y]
            nearest_indices.append(row.Index)
    
    near = smallest[~np.isnan(smallest)].reshape(-1,2)
    
    return near, K, nearest_indices
    
    
def center(arrayx, arrayy, centerx, centery):
    """
    Shift data points so that grid cell of interest is at the origin
    Inputs:
        arrayx - x coordinates of data
        arrayy - y coordinates of data
        centerx - x coordinate of grid cell of interest
        centery - y coordinate of grid cell of interest
    Outputs:
        centered_array - array of coordinates that are shifted with respect to grid cell of interest
    """
    centerx = arrayx - centerx
    centery = arrayy - centery
    centered_array = np.array([centerx, centery])
    
    return centered_array


def min_rad(data, rad):
    """
    Find minimum radius to aquire minimum number of NN and cluster value
    Inputs:
        data - all observed data and simulated locations before current
        rad - begining search radius
    Outputs:
        rad - search radius that satisfies simulation conditions
    """
    tmp = data[data.dist < rad]
    
    # at least 8 NN point and one observed bed elevation location for cluster assignment
    while len(tmp) < 8 or len(tmp[~np.isnan(tmp['K'])]) < 1:
        rad += 10000
        tmp = data[data.dist < rad]
        
    return rad
    

########################

#   Simple Kringing

########################


def kriging(vario, nearest_pts, pred_xy):
    """
    Simple Kriging where vairogram parameters are unique to each cluster
    Inputs:
        vario - variogram parameter list for clusted assigned to current location
        nearest_pts - nearest neighbor locations
        pred_xy - current location the kriging weights are being calculated for
    Outputs:
        k_weights - kriging weights
        covariance_array - covariance between data and unknown
    """
    numpoints = len(nearest_pts)
    xy_val = nearest_pts[:,:2]
    
    # unpack variogram parameters
    azimuth = vario[0]
    major_range = vario[2]
    minor_range = vario[3]
    rotation_matrix = make_rotation_matrix(azimuth, major_range, minor_range)
    
    # covariance between data
    covariance_matrix = np.zeros(shape=((numpoints, numpoints)))
    covariance_matrix = make_covariance_matrix(xy_val, vario, rotation_matrix)


    # covariance between data and unknown
    covariance_array = np.zeros(shape=(numpoints))
    k_weights = np.zeros(shape=(numpoints))
    covariance_array = make_covariance_array(xy_val, np.tile(pred_xy, numpoints), vario, rotation_matrix)
    covariance_matrix.reshape(((numpoints)), ((numpoints)))

    k_weights, res, rank, s = np.linalg.lstsq(covariance_matrix, covariance_array, rcond=None)
    
    return k_weights, covariance_array
    
    
#########################

#   Rotation Matrix

#########################


def make_rotation_matrix(azimuth, major_range, minor_range):
    """
    Make rotation matrix for accommodating anisotropy
    Inputs:
        azimuth - angle (in degrees from horizontal) of axis of orientation
        major_range - range parameter of variogram in major direction, or azimuth
        minor_range - range parameter of variogram in minor direction, or orthogonal to azimuth
    Outputs:
        rotation_matrix - 2x2 rotation matrix used to perform coordinate transformations
    """
    theta = (azimuth / 180.0) * np.pi
    
    rotation_matrix = np.dot(
        np.array([[np.cos(theta), -np.sin(theta)],
                [np.sin(theta), np.cos(theta)],]),
        np.array([[1 / major_range, 0], [0, 1 / minor_range]]))
    
    return rotation_matrix


#############################

#   Covariance Functions

#############################
    
    
def covar(effective_lag, sill, nug):
    """
    Compute covariance using exponential covariance model
    Inputs:
        effective_lag - lag distance that is normalized to a range of 1
        sill - sill of variogram
        nug - nugget of variogram
    Outputs:
        c - covariance
    """
    c = (sill - nug)*np.exp(-3 * effective_lag)

    return c


def make_covariance_matrix(coord, vario, rotation_matrix):
    """
    Make covariance matrix showing covariances between each pair of input coordinates
    Inputs:
        coord - coordinates of data points
        vario - array of variogram parameters
        rotation_matrix - rotation matrix used to perform coordinate transformations
    Outputs:
        covariance_matrix - nxn matrix of covariance between n points
    """
    nug = vario[1]
    sill = vario[4]
    mat = np.matmul(coord, rotation_matrix)
    effective_lag = pairwise_distances(mat,mat)
    covariance_matrix = covar(effective_lag, sill, nug)

    return covariance_matrix


def make_covariance_array(coord1, coord2, vario, rotation_matrix):
    """
    Make covariance array showing covariances between each data points and grid cell of interest
    Inputs:
        coord1 - coordinates of n data points
        coord2 - coordinates of grid cell of interest (i.e. grid cell being simulated) that is repeated n times
        vario - array of variogram parameters
        rotation_matrix - rotation matrix used to perform coordinate transformations
    Outputs:
        covariance_array - nx1 array of covariance between n points and grid cell of interest
    """
    nug = vario[1]
    sill = vario[4]
    mat1 = np.matmul(coord1, rotation_matrix)
    mat2 = np.matmul(coord2.reshape(-1,2), rotation_matrix)
    effective_lag = np.sqrt(np.square(mat1 - mat2).sum(axis=1))
    covariance_array = covar(effective_lag, sill, nug)

    return covariance_array


#######################################

#   Sequential Gaussian Simulation

#######################################


def sgs_pred_Z(kr_dictionary, df_data, df_nan, gamma, xx, yy, zz, cluster):
    """
    Use previously obtained kriging weights to perform SGS
    Inputs:
        kr_dictionary - dictionary with kriging weights and associated data
        df_data - DataFrame with observed bed elevation values
        df_nan - DataFrame of points to simulate
        gamma - dictionary of variogram parameters for each cluster
        xx - column name for x coordinates of input data frame
        yy - column name for y coordinates of input data frame
        zz - column for z values (or data variable) of input data frame
        cluster - column name for cluster assigned to location
    Outputs:
        pred_xyzk - DataFrame of simulated data with elevation values and cluster assignment
    """
    print('Starting sequential simulation...')
    df_data = df_data.rename(columns = {xx: "X", yy: "Y", zz: "Z", cluster: "K"})
    df_nan = df_nan.rename(columns = {xx: "X", yy: "Y", zz: "Z", cluster: "K"})
    all_df = pd.concat([df_data, df_nan])
    
    zmean = np.average(df_data["Z"].values)
    z_lookup = {row.Index: row.Z for row in df_data.itertuples()}
    
    start = time.time()
    
    # ordered prediction of elevation values using kriging weights and nearest elevations
    for i, row in enumerate(df_nan.itertuples(), start=1):
        
        k_weights, r, nearidx, cluster_num = kr_dictionary[row.Index]
        norm_bed_val = np.array([z_lookup[idx] for idx in nearidx])
        
        # get variance of cluster assigned to current point
        vario = gamma[cluster_num]
        cluster_var = vario[4]
        
        # calculate kriging mean and variance
        est = zmean + np.sum(k_weights[:len(norm_bed_val)] * (norm_bed_val - zmean))
        var = abs(cluster_var - np.sum(k_weights[:len(norm_bed_val)] * r[:len(norm_bed_val)]))
        
        z_lookup[row.Index] = np.random.default_rng().normal(est, math.sqrt(var))
        df_nan.loc[row.Index, 'Z'] = z_lookup[row.Index]
        df_nan.loc[row.Index, 'K'] = cluster_num
    
    data_xyzk = df_data.rename(columns = {"X": xx, "Y": yy, "Z": zz, "K": cluster})
    pred_xyzk = df_nan.rename(columns = {"X": xx, "Y": yy, "Z": zz, "K": cluster})
        
    print(f'\t{round((time.time()-start), 2)} seconds to complete\n')

    return data_xyzk, pred_xyzk


def concat(df_data, pred_xyzk):
    """
    Conatenate real and simulated points and sort data
    Inputs:
        df_data - DataFrame with observed bed elevation values
        pred_xyzk - DataFrame of simulated data with elevation values and cluster assignment
    Outputs:
        df_sim - complete DataFrame of all data required for modeling
    """
    data_xyzk = df_data
    frames = [data_xyzk, pred_xyzk]
    df_total = pd.concat(frames)
    df_sim = df_total.sort_index()
    
    return df_sim

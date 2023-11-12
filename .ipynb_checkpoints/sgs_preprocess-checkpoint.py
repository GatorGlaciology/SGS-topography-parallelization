import numpy as np
import pandas as pd
from sklearn.preprocessing import QuantileTransformer
import skgstat as skg
import multiprocessing as mp
import itertools
import random
import time


#############

#   Menu

#############


def menu():
    """
    Menu to recieve user parameters
    Outputs:
        file_name - name of csv file with ice penetrating radar measurements
        x - column name for x coordinates of input data frame
        y - column name for y coordinates of input data frame
        z - column for z values (or data variable) of input data frame
        xmin - minimum x extent
        xmax - maximum x extent
        ymin - minimum y extent
        ymax - maximum y extent
        res - grid cell resolution
        num_realizations - number of realizations
    """
    print('\nSubglacial Topographic Modeling using Parallel Implimentation of SGS')
    print('--------------------------------------------------------------------')
    file_name = input('Dataset File path (E.g. - Data/test_data.csv): ')
    x = input('\nColumn name for X values: ')
    y = input('\nColumn name for Y values: ')
    z = input('\nColumn name for bed elevation values: ')
    boundsQ = input('\nWould you like to specify bounding coordinates [Y/N]? ')
    if boundsQ == 'Y':
        xmin = float(input('xmin: ')); xmax = float(input('xmax: '))
        ymin = float(input('ymin: ')); ymax = float(input('ymax: '))
    else:
        xmin = None; xmax = None
        ymin = None; ymax = None
    res = int(input('\nResolution of model in meters: '))
    num_realizations = int(input('\nNumber of Realizations: '))
    print(f'\n---------------------------------------------------------------------------------')
    
    
    return file_name, x, y, z, xmin, xmax, ymin, ymax, res, num_realizations


###################

#   Grid Data

###################
    
    
def make_grid(xmin, xmax, ymin, ymax, res):
    """
    Generate coordinates for output of gridded data
    Inputs:
        xmin - minimum x extent
        xmax - maximum x extent
        ymin - minimum y extent
        ymax - maximum y extent
        res - grid cell resolution
    Outputs:
        prediction_grid_xy - x,y array of coordinates
        rows - number of rows
        cols - number of columns
    """
    cols = np.ceil((xmax - xmin)/res).astype(int)
    rows = np.ceil((ymax - ymin)/res).astype(int)
    x = np.arange(xmin,xmax,res); y = np.arange(ymin,ymax,res)
    xx, yy = np.meshgrid(x,y)
    x = np.reshape(xx, (int(rows)*int(cols), 1))
    y = np.reshape(yy, (int(rows)*int(cols), 1))

    pred_grid_xy = np.concatenate((x,y), axis = 1)
    
    return pred_grid_xy, cols, rows


def grid_data(df, xmin, xmax, ymin, ymax, res, xx, yy, zz):
    """
    Grid conditioning data
    Inputs:
        df - DataFrame of conditioning data
        xmin - min x value for this model
        xmax - max x value for this model
        ymin - min y value for this model
        ymax - max y value for this model
        res - grid cell resolution
        xx - column name for x coordinates of input data frame
        yy - column name for y coordinates of input data frame
        zz - column for z values (or data variable) of input data frame
    Outputs:
        df_data - DataFrame of conditioning (observed) data
        grid_matrix - matrix of gridded data
        df_nan - DataFrame of points to simulate
    """
    print(f'Dataframe has {len(df)} points.')
    df = df.rename(columns = {xx: "X", yy: "Y", zz: "Z"})
    
    if xmin == None:
        xmin = df['X'].min()
    if xmax == None:
        xmax = df['X'].max()
    if ymin == None:
        ymin = df['Y'].min()
    if ymax == None:
        ymax = df['Y'].max()
    
    # make array of grid coordinates
    grid_coord, cols, rows = make_grid(xmin, xmax, ymin, ymax, res)
    
    df = df[['X','Y','Z']]
    np_data = df.to_numpy()
    np_resize = np.copy(np_data)
    origin = np.array([xmin,ymin])
    resolution = np.array([res,res])
    
    # shift and re-scale the data by subtracting origin and dividing by resolution
    np_resize[:,:2] = np.rint((np_resize[:,:2]-origin)/resolution)
    
    grid_sum = np.zeros((rows,cols))
    grid_count = np.copy(grid_sum)

    for i in range(np_data.shape[0]):
        xindex = np.int32(np_resize[i,1])
        yindex = np.int32(np_resize[i,0])
        
        if ((xindex >= rows) | (yindex >= cols)):
            continue
            
        grid_sum[xindex,yindex] = np_data[i,2] + grid_sum[xindex,yindex]
        grid_count[xindex,yindex] = 1 + grid_count[xindex,yindex] 
        
    
    np.seterr(invalid='ignore')
    grid_matrix = np.divide(grid_sum, grid_count)
    grid_array = np.reshape(grid_matrix,[rows*cols])
    grid_sum = np.reshape(grid_sum,[rows*cols])
    grid_count = np.reshape(grid_count,[rows*cols])
    
    # make dataframe
    grid_total = np.array([grid_coord[:,0], grid_coord[:,1], grid_sum, grid_count, grid_array])
    df_grid = pd.DataFrame(grid_total.T, columns = ['X', 'Y', 'Sum', 'Count', 'Z'])
    
    print(f'Gridded data has {len(df_grid)} points, {round(df_grid.Z.isnull().sum()/len(df_grid)*100, 1)}% ({df_grid.Z.isnull().sum()}) are missing and will be simulated.')
    
    # seperate conditional data from data points to simulate
    df_nan = df_grid[df_grid["Z"].isnull() == True]
    df_data = df_grid[df_grid["Z"].isnull() == False]
    df_data = df_data.rename(columns = {"Z": zz})
    df_nan = df_nan.rename(columns = {"Z": zz})
    
    df_data.loc[:,'simulated'] = 0
    df_nan.loc[:,'simulated'] = 1
    df_data.loc[:, 'cluster'] = np.nan
    df_nan.loc[:,'cluster'] = np.nan
    
    return df_data.astype('float32'), grid_matrix.astype('float32'), df_nan.astype('float32')
    
    
############################

#   Data Transformation

############################


def nscore(df_data, z):
    """
    Normalize bed elevation values of conditioning data
    Inputs:
        df_data - DataFrame with observed bed elevation values
        z - column for z values
    Outputs:
        norm_bed - normalized bed elevation values
        nst_trans - fitted normal score transformer object
    """
    data = df_data[z].values.reshape(-1,1)
    nst_trans = QuantileTransformer(n_quantiles=500, output_distribution='normal').fit(data)
    norm_bed = nst_trans.transform(data)
    
    return norm_bed, nst_trans


def shuffle_pred_grid(df_nan):
    """
    Shuffle prediction grid to create random simulation path
    Inputs:
        df_nan - ordered predition grid
    Outputs:
        df_nan - shuffled predition grid
    """
    randomize = df_nan.index.tolist()
    random.shuffle(randomize)
    df_nan = df_nan.loc[randomize]
    
    return df_nan
    
    
###############################

#   Adaptive Partitioning

###############################


def adaptive_partitioning(df_data, xmin, xmax, ymin, ymax, max_points, min_length, i = 0, max_iter=None):
    """
    Rercursively split clusters until they are all below max_points, but don't go smaller than min_length
    Inputs:
        df_data - DataFrame with X, Y, and cluster (cluster id)
        xmin - min x value of this partion
        xmax - max x value of this partion
        ymin - min y value of this partion
        ymax - max y value of this partion
        i - keeps track of total calls to this function
        max_points - all clusters will be "quartered" until points below this
        min_length - minimum side length of sqaures, preference over max_points
        max_iter - maximum iterations if worried about unending recursion
    Outputs:
        df_data - updated DataFrame with new cluster assigned the next integer
        i - number of iterations
    """
    if xmin == None:
        xmin = df_data['X'].min()
    if xmax == None:
        xmax = df_data['X'].max()
    if ymin == None:
        ymin = df_data['Y'].min()
    if ymax == None:
        ymax = df_data['Y'].max()
    
    # optional 'safety' if there is concern about runaway recursion
    if max_iter is not None:
        if i >= max_iter:
            return df_data, i
    
    dx = xmax - xmin
    dy = ymax - ymin
    
    # >= and <= greedy so we don't miss any points
    xleft = (df_data.X >= xmin) & (df_data.X <= xmin+dx/2)
    xright = (df_data.X <= xmax) & (df_data.X >= xmin+dx/2)
    ybottom = (df_data.Y >= ymin) & (df_data.Y <= ymin+dy/2)
    ytop = (df_data.Y <= ymax) & (df_data.Y >= ymin+dy/2)
    
    # index the current cell into 4 quarters
    q1 = df_data.loc[xleft & ybottom]
    q2 = df_data.loc[xleft & ytop]
    q3 = df_data.loc[xright & ytop]
    q4 = df_data.loc[xright & ybottom]
    
    # for each quarter, qaurter if too many points, else assign cluster and return
    for q in [q1, q2, q3, q4]:
        if (q.shape[0] > max_points) & (dx/2 > min_length):
            i = i+1
            df_data, i = adaptive_partitioning(df_data, q.X.min(), q.X.max(), q.Y.min(), q.Y.max(),
                                                max_points, min_length, i, max_iter)
        else:
            qcount = df_data.cluster.max()
            # ensure zero indexing
            if np.isnan(qcount) == True:
                qcount = 0
            else:
                qcount += 1
            df_data.loc[q.index, 'cluster'] = qcount
            
    return df_data, i
    
    
####################################################

#   Fit variograms to each cluster in parallel

####################################################


def get_variograms(df_data, n_lags, max_lag, processes):
    """
    Gets isotropic variograms for each cluster in parallel
    Inputs:
        df_data - DataFrame with observed bed elevation values
        n_lags - number of bins
        max_lag - maximum lag distance
        processes - number of processes
    Outputs:
        gamma - dictionary of variogram parameters for each cluster
    """
    start = time.time()
    print('\nGetting parallel variograms...')
    cluster_locs_nres = {}
    
    # seperate by cluster value
    for num, groupdf in df_data.groupby('cluster'):
        cluster_locs_nres[num] = (groupdf[['X','Y']].values, groupdf['Norm_Bed'])
    
    # create generators
    gen_locs = (v[0] for k, v in cluster_locs_nres.items())
    gen_nres = (v[1] for k, v in cluster_locs_nres.items())
    
    # create iterable parameter list
    args = zip(gen_locs, gen_nres, itertools.cycle([n_lags]), itertools.cycle([max_lag]))
    
    pool = mp.Pool(processes)
    out = pool.starmap(Variogram, args)
  
    print(f'\t{round((time.time()-start), 2)} seconds to complete\n')
    
    # save variogram parameters in dictionary [azimuth, nugget, major_range, minor_range, sill]
    azimuth = 0
    gamma = {clusternum: [azimuth, vario.parameters[2], vario.parameters[0],
        vario.parameters[0], vario.parameters[1]] for clusternum, vario in enumerate(out)}
    
    return gamma
    
    
def Variogram(pos, val, n_lags, max_lag):
    """
    Calculates a variogram
    Inputs:
        pos - coordinates of data to fit variogram
        val - observed bed elevation values to fit variogram
        n_lags - number of bins
        max_lag - maximum lag distance
    Outputs:
        vario - fitted variogram
    """
    vario = skg.Variogram(pos, val, bin_func = "even", n_lags = n_lags, maxlag = max_lag, normalize=False)
    
    return vario

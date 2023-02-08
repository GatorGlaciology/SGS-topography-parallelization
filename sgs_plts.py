import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import QuantileTransformer
from mpl_toolkits.axes_grid1 import make_axes_locatable
from numpy.random import default_rng
from matplotlib.colors import LightSource


def plt_graph(sim, df_bed, res, x, y, z, i):

    # 2D hillshade topographic plot
    title = f'Bed Elevation Model {i+1}'

    mu = np.mean(sim[z]); sd = np.std(sim[z])
    vmin = mu - 3*sd ; vmax = mu + 3*sd

    xmin = np.min(df_bed[x]); xmax = np.max(df_bed[x])
    ymin = np.min(df_bed[y]); ymax = np.max(df_bed[y])

    grid_xy, rows, cols = prediction_grid(xmin, xmax, ymin, ymax, res)
    
    plot_i = mplot(grid_xy, sim[[z]].to_numpy(), rows, cols, title, vmin = vmin, vmax = vmax, hillshade=True)
    plot_i.savefig(f'Output/Plot_{i+1}.png', bbox_inches = 'tight')
    
    
def mplot(Pred_grid_xy, sim, rows, cols, title, xlabel='X [m]', ylabel='Y [m]', 
           clabel='Bed [m]', vmin=-400, vmax=600, hillshade=False, titlepad=None):
    x_mat = Pred_grid_xy[:,0].reshape((rows, cols))
    y_mat = Pred_grid_xy[:,1].reshape((rows, cols))
    mat = sim.reshape((rows, cols))

    xmin = Pred_grid_xy[:,0].min(); xmax = Pred_grid_xy[:,0].max()
    ymin = Pred_grid_xy[:,1].min(); ymax = Pred_grid_xy[:,1].max()
    
    cmap=plt.get_cmap('gist_earth')

    fig, ax = plt.subplots(1, figsize=(5,5))
    im = plt.pcolormesh(x_mat, y_mat, mat, vmin=vmin, vmax=vmax, cmap=cmap)
    
    if hillshade == True:
        # Shade from the northeast, with the sun 45 degrees from horizontal
        ls = LightSource(azdeg=45, altdeg=45)
        
        # leaving the dx and dy as 1 means a vertical exageration equal to dx/dy
        hillshade = ls.hillshade(mat, vert_exag=1, dx=1, dy=1, fraction=1.0)
        plt.pcolormesh(x_mat, y_mat, hillshade, cmap='gray', alpha=0.1)
        
    if titlepad is None:
        plt.title(title)
    else:
        plt.title(title, pad=titlepad)
    plt.xlabel(xlabel); plt.ylabel(ylabel)
    plt.xticks(np.linspace(xmin, xmax, 5))
    plt.yticks(np.linspace(ymin, ymax, 5))

    # make colorbar
    cbar = make_colorbar(fig, im, vmin, vmax, clabel)
    
    ax.axis('scaled')
    
    return plt


# for jupyter notebook

def make_colorbar(fig, im, vmin, vmax, clabel, ax=None):
    if ax is None:
        ax = plt.gca()
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.1)
    cbar = plt.colorbar(im, ticks=np.linspace(vmin, vmax, 11), cax=cax)
    cbar.set_label(clabel, rotation=270, labelpad=15)
    return cbar

def splot2D(df, title, xlabel='X [m]', ylabel='Y [m]', clabel='Bed [m]', x='X', y='Y', c='Bed',
            vmin=-400, vmax=600, s=0.5):
    fig, ax = plt.subplots(1, figsize=(5,5))
    im = plt.scatter(df[x], df[y], c=df[c], vmin=vmin, vmax=vmax,
                     marker='.', s=s, cmap='gist_earth')
    plt.title(title)
    plt.xlabel(xlabel); plt.ylabel(ylabel)
    plt.locator_params(nbins=5)

    # make colorbar
    cbar = make_colorbar(fig, im, vmin, vmax, clabel)

    ax.axis('scaled')
    plt.show()
    
def prediction_grid(xmin, xmax, ymin, ymax, res):
        """
        Make prediction grid
        Inputs:
            xmin - minimum x extent
            xmax - maximum x extent
            ymin - minimum y extent
            ymax - maximum y extent
            res - grid cell resolution
        Outputs:
            prediction_grid_xy - x,y array of coordinates
        """ 
        cols = int(np.ceil((xmax - xmin)/res))
        rows = int(np.ceil((ymax - ymin)/res))  
        x = np.linspace(xmin, xmin+(cols*res), num=int(cols), endpoint=False)
        y = np.linspace(ymin, ymin+(rows*res), num=int(rows), endpoint=False)
        xx, yy = np.meshgrid(x,y)
        x = np.reshape(xx, (int(rows)*int(cols), 1))
        y = np.reshape(yy, (int(rows)*int(cols), 1))
        prediction_grid_xy = np.concatenate((x,y), axis = 1)
        
        return prediction_grid_xy, rows, cols

def mplot1(Pred_grid_xy, sim, rows, cols, title, xlabel='X [m]', ylabel='Y [m]', 
           clabel='Bed [m]', vmin=-400, vmax=600, hillshade=False, titlepad=None):
    x_mat = Pred_grid_xy[:,0].reshape((rows, cols))
    y_mat = Pred_grid_xy[:,1].reshape((rows, cols))
    mat = sim.reshape((rows, cols))

    xmin = Pred_grid_xy[:,0].min(); xmax = Pred_grid_xy[:,0].max()
    ymin = Pred_grid_xy[:,1].min(); ymax = Pred_grid_xy[:,1].max()
    
    cmap=plt.get_cmap('gist_earth')

    fig, ax = plt.subplots(1, figsize=(5,5))
    im = plt.pcolormesh(x_mat, y_mat, mat, vmin=vmin, vmax=vmax, cmap=cmap)
    
    if hillshade == True:
        # Shade from the northeast, with the sun 45 degrees from horizontal
        ls = LightSource(azdeg=45, altdeg=45)
        
        # leaving the dx and dy as 1 means a vertical exageration equal to dx/dy
        hillshade = ls.hillshade(mat, vert_exag=1, dx=1, dy=1, fraction=1.0)
        plt.pcolormesh(x_mat, y_mat, hillshade, cmap='gray', alpha=0.1)
        
    if titlepad is None:
        plt.title(title)
    else:
        plt.title(title, pad=titlepad)
    plt.xlabel(xlabel); plt.ylabel(ylabel)
    plt.xticks(np.linspace(xmin, xmax, 5))
    plt.yticks(np.linspace(ymin, ymax, 5))

    # make colorbar
    cbar = make_colorbar(fig, im, vmin, vmax, clabel)
    
    ax.axis('scaled')
    plt.show()


def plt_clusters(df_grid):

    clusters, counts = np.unique(df_grid.cluster, return_counts=True)
    n_clusters = len(clusters)

    # randomize colormap
    rng = default_rng()
    vals = np.linspace(0, 1.0, n_clusters)
    rng.shuffle(vals)
    cmap = plt.cm.colors.ListedColormap(plt.cm.nipy_spectral(vals))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11,4))

    ax1.locator_params(nbins=5)

    im = ax1.scatter(df_grid['X'], df_grid['Y'], c=df_grid['cluster'], cmap=cmap, marker=".", s=1)
    im.set_clim(-0.5, max(clusters)+0.5)
    ax1.set_title('Clusters')
    ax1.set_xlabel('X (m)')
    ax1.set_ylabel('Y (m)')
    cbar = plt.colorbar(im, orientation="vertical", ax=ax1)
    cbar.set_ticks(np.linspace(0, max(clusters), n_clusters))
    cbar.set_ticklabels(range(n_clusters))
    cbar.set_label('Clustered data', rotation=270, labelpad=15)
    ax1.axis('scaled')

    ax2.bar(clusters, counts)
    ax2.set_xlabel('Cluster ID')
    ax2.set_title('Counts')
    plt.show()
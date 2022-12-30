import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import QuantileTransformer


def plt_graph(sim, nst_trans, i):
    """
    performs preliminary calculations, calls plot function and saves result
    Inputs:
        sim - DataFrame with all observed and simulated bed elevation data
        nst_trans - fitted normal score transformer object
        i - simulation number
    """
    # reverse normal score transformation
    tmp = sim['Norm_Bed'].values.reshape(-1,1)
    sim_trans = nst_trans.inverse_transform(tmp)
    
    # mean and SD of bed elevation values (for range)
    mu = np.mean(sim_trans)
    sig = np.std(sim_trans)
    
    title = [f'Bed Elevation Model {i+1}','Bed elevation [m]']
    plt_trans = beddata(sim, sim_trans, mu-(3*sig), mu+(3*sig), title)
    plt_trans.savefig(f'Output/Plot_{i+1}.png', bbox_inches = 'tight')
    
    
def beddata(df, z, min, max, title):
    """
    Plots bed elevation data
    Inputs:
        df - DataFrame with all elevation locations
        z - list with all elevation values
        min - minimum color scale value
        max - maximum color scale value
        title - title of plot
    Outputs:
        plt - Plot of bed elevation data
    """
    plt.clf()
    im = plt.scatter(df['X'],df['Y'], c = z, vmin = min, vmax = max, marker=".", s = 13)
    plt.title(title[0])
    plt.xlabel('X (m)'); plt.ylabel('Y (m)')
    cbar = plt.colorbar(im, orientation="vertical", ticks=np.linspace(min, max, 13))
    cbar.set_label(title[1], rotation=270, labelpad=20)
    plt.subplots_adjust(left=0.0, bottom=0.0, right=1.0, top=1.0)
    plt.axis('scaled')
    
    return plt


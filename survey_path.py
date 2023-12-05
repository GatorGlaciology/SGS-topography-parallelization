import numpy as np
import pandas as pd

# Filters rows of dataset to include points cooresponding to vertical lines  
def vertical(df, rows, cols, start, spacing):
    
    index_list = []
    first_line = start

    for i in range(rows):

        pos = first_line
        
        while np.floor(pos/cols) == i:
            
            index_list.append(pos)
            pos += spacing

        first_line += cols

    df_vert = df.iloc[index_list]

    return df_vert, index_list

# Filters rows of dataset to include points cooresponding to horizontal lines  
def horizontal(df, rows, cols, start, spacing):
    
    index_list = []

    first_line = start * cols 
    line = 0

    while line*spacing + start < rows:

        pos = first_line + (line * spacing * cols)

        for _ in range(cols):

            index_list.append(pos)
            pos += 1
        
        line += 1

    df_horiz = df.iloc[index_list]

    return df_horiz, index_list

# Filters rows of dataset to include points that form a grid pattern
def grid(df, rows, cols, start, spacing):
    
    _, sparse_vert =  vertical(df, rows, cols, start, spacing * 2)
    _, sparse_horiz =  horizontal(df, rows, cols, start, spacing * 2)
    path_grid = sparse_horiz + sparse_vert
    df_grid = df.iloc[path_grid]

    return df_grid


# Creates equally spaced diagonal lines from dataset
# Achives this by creating a 3d grid of points cooresponding to relative location
# and multiplying this by a mask of values in diagonal patern, then filtering NaNs
def diagonal(df, rows, cols, x, y, z, start, spacing, pos = True):

    if pos:
        y_inc = 1
    else:
        y_inc = -1

    # Create a 3d grid from dataframe
    x_list = df[[x]].to_numpy()
    y_list = df[[y]].to_numpy()
    z_list = df[[z]].to_numpy()
    stack = np.concatenate((x_list, y_list, z_list), axis = 1)
    stack = stack.reshape((rows, cols, 3))


    # Initialize diagonal mask
    mask = np.empty((rows, cols))
    mask[:] = np.nan

    # Create masked values along y axis
    # Drawing lines starting at point on y axis 
    y_start = rows - start
    x_start = 0

    while y_start > 0:

        y_pos = y_start
        x_pos = x_start

        while x_pos < cols:

            if pos and y_pos >= rows:
                break
            if not pos and y_pos <= 0:
                break

            mask[y_pos, x_pos] = 1

            y_pos += y_inc
            x_pos += 1

        y_start -= spacing

    # Create masked values along x axis
    # Drawing lines starting at point on x axis 
    if pos:
        x_start = -y_start
        y_start = 0
    else:
        x_start = spacing - start
        y_start = rows - 1 

    while x_start < cols:

        y_pos = y_start
        x_pos = x_start

        while y_pos < rows and x_pos < cols:
            mask[y_pos, x_pos] = 1

            y_pos += y_inc
            x_pos += 1

        x_start += spacing

    # Multiply grided data frame with grided mask and remove nan values
    mask_3d = np.repeat(mask[:, :, np.newaxis], 3, axis=2)
    diag_vals = stack * mask_3d
    diag_vals = diag_vals.reshape((rows*cols, 3))
    df_diag = pd.DataFrame({x: diag_vals[:,0], y: diag_vals[:,1], z: diag_vals[:,2]})
    df_diag = df_diag.dropna()

    return df_diag


def grid_diag(df, rows, cols, x, y, z, start, spacing):
    df_posdiag = diagonal(df, rows, cols, x, y, z, start, spacing*2, pos=True)
    df_negdiag = diagonal(df, rows, cols, x, y, z, start, spacing*2, pos=False)

    df_grid_diag = pd.concat([df_posdiag, df_negdiag])
    
    return df_grid_diag
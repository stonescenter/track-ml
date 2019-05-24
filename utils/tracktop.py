from scipy import special
import matplotlib.pyplot as plt
import random
import numpy as np
import plotly.graph_objs as go
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot

default_err = 0.03
default_err_fad = 0.174
default_len_err = 500
default_center = 1


def scale_range(input, min, max):
    input += -(np.min(input))
    input /= np.max(input) / (max - min)
    input += min
    return input


def err_lin(position, err=default_err, len_err=default_len_err):
    err_array = np.linspace(default_center - err, default_center + err, len_err)
    value_err = position * random.choice(err_array)
    return value_err


def err_normal(position, err=default_err, len_err=default_len_err):
    err_array = np.random.normal(loc=default_center, scale=err, size=len_err)
    value_err = position * random.choice(err_array)
    return value_err


def err_erf(position, err=default_err, len_err=default_len_err):
    int_array = np.linspace(-err * 100, err * 100, len_err)
    err_array = special.erfc(x)
    scaled_err_array = scale_range(
        err_array, default_center - err, default_center + err
    )
    value_err = position * random.choice(scaled_err_array)
    return value_err


def err_faddeeva(position, err=default_err_fad, len_err=default_len_err):
    err_array = np.linspace(-err, err, len_err)
    faddeev_array = special.wofz(err_array)
    factor = [1,-1]
    value_err = round(position * (random.choice(faddeev_array.real) ** 
                                  random.choice(factor)),8)
    return value_err

# Flatten hits, pad the line to make up the original length,
# add back the index, vertex, momentum in the front
# and the 0 (for "False") in the back
def make_random_track(selected_hits, total_hits):
    flat_selected_hits = selected_hits.flatten()
    padded_selected_hits = np.pad(
        flat_selected_hits,
        (0, total_hits * 6 - flat_selected_hits.size),
        "constant",
        constant_values=0,
    )

    candidate_track = np.concatenate(
        [
            np.array([-1]),
            np.array([0, 0, 0]),
            np.array([0, 0, 0]),
            padded_selected_hits,
            np.array([0]),
        ]
    )
    return candidate_track


#Previous version of track sort working with x, y, z
'''
def xyz_swap(df_tb_swap,index_xyz,i,j):
    index_xyz[i], index_xyz[j] = index_xyz[j], index_xyz[i]
    df_tb_swap.iloc[3*i]  , df_tb_swap.iloc[3*j]   = df_tb_swap.iloc[3*j],   df_tb_swap.iloc[3*i]
    df_tb_swap.iloc[3*i+1], df_tb_swap.iloc[3*j+1] = df_tb_swap.iloc[3*j+1], df_tb_swap.iloc[3*i+1]
    df_tb_swap.iloc[3*i+2], df_tb_swap.iloc[3*j+2] = df_tb_swap.iloc[3*j+2], df_tb_swap.iloc[3*i+2]

def xyz_bsort(df_to_be_sorted):
    index_xyz = []
    df_n_col  = df_to_be_sorted.shape[0]//3
    for aux in range(0,df_n_col):
        x = df_to_be_sorted.iloc[3*aux+0]
        y = df_to_be_sorted.iloc[3*aux+1]
        z = df_to_be_sorted.iloc[3*aux+2]
        index_xyz.append(x**2 + y**2 + z**2)
    for i in range(1,len(index_xyz)):
        for j in range(0,len(index_xyz)-i):
            if index_xyz[j]>index_xyz[j+1]:
                xyz_swap(df_to_be_sorted,index_xyz,j,j+1)
'''
# SWAP function to work with bubble sort
def xyz_swap(df_tb_swap, index_xyz, i, j, pivot):
    pivot_i = pivot * i
    pivot_j = pivot * j
    #swapping index array
    index_xyz[i], index_xyz[j] = index_xyz[j], index_xyz[i]
    #swapping tracks array
    for aux in range(pivot):
        df_tb_swap.iloc[pivot_i+aux],df_tb_swap.iloc[pivot_j+aux]=df_tb_swap.iloc[pivot_j+aux],df_tb_swap.iloc[pivot_i+aux]

#function to sort a line of a track dataset divided by hits with 6 elements
def xyz_bsort(df_to_be_sorted, **kwargs):
    pivot = 6
    
    if kwargs.get('pivot'):
        pivot = kwargs.get('pivot')
        
    #df_to_be_sorted = df_to_be_sorted.replace(0.0, np.PINF)
    
    index_xyz = []
    df_n_col  = df_to_be_sorted.shape[0]//pivot
    for aux in range(0,df_n_col):
        pivot_tmp = pivot * aux
        x = df_to_be_sorted.iloc[pivot_tmp + 0]
        y = df_to_be_sorted.iloc[pivot_tmp + 1]
        z = df_to_be_sorted.iloc[pivot_tmp + 2]
        index_xyz.append(x ** 2 + y ** 2 + z ** 2)
    for i in range(1,len(index_xyz)):
        for j in range(0, len(index_xyz) - i):
            if index_xyz[j] > index_xyz[j + 1]:
                xyz_swap(df_to_be_sorted, index_xyz, j, j+1, pivot)
    
    #df_to_be_sorted = df_to_be_sorted.replace(0.0, np.PINF)
    
    

#function to plot tracks
def track_plot(df_tb_plt, **kwargs):
    
    track_color = 'red'
    n_tracks = 1
    pivot = 6
    title = 'Track plots'
    
    if kwargs.get('track_color'):
        track_color = kwargs.get('track_color')
    
    if kwargs.get('n_tracks'):
        n_tracks = kwargs.get('n_tracks')
    
    if kwargs.get('pivot'):
        pivot = kwargs.get('pivot')
    
    if kwargs.get('title'):
        title = kwargs.get('title')
        
        
    dft_size = df_tb_plt.shape[1]
    len_xyz = int(dft_size/pivot)

    # Initializing lists of indexes
    selected_columns_x = np.zeros(len_xyz)
    selected_columns_y = np.zeros(len_xyz)
    selected_columns_z = np.zeros(len_xyz)

    # Generating indexes
    for i in range(len_xyz):
        selected_columns_x[i] = int(i*pivot)
        selected_columns_y[i] = int(i*pivot+1)
        selected_columns_z[i] = int(i*pivot+2)

    # list of data to plot
    data = []
    track = [None] * n_tracks

 
    for i in range(n_tracks):
        track[i] = go.Scatter3d(
            # Removing null values (zeroes) in the plot
            x = df_tb_plt.replace(0.0, np.nan).iloc[i,selected_columns_x],
            y = df_tb_plt.replace(0.0, np.nan).iloc[i,selected_columns_y],
            z = df_tb_plt.replace(0.0, np.nan).iloc[i,selected_columns_z],
            # x,y,z data with null values (zeroes)
            #
            # x = df_hits_x.iloc[i,:],
            # y = df_hits_y.iloc[i,:],
            # z = df_hits_z.iloc[i,:],
            marker = dict(
                size = 1,
                color = track_color,
            ),
            line = dict(
                color = track_color,
                width = 1
            )
        )
        # append the track[i] in the list for plotting
        data.append(track[i])
        
        
    layout = dict(
        width    = 900,
        height   = 750,
        autosize = False,
        title    = title,
        scene = dict(
            xaxis = dict(
                gridcolor       = 'rgb(255, 255, 255)',
                zerolinecolor   = 'rgb(255, 255, 255)',
                showbackground  = True,
                backgroundcolor = 'rgb(230, 230,230)',
                title           ='x (mm)'
            ),
            yaxis=dict(
                gridcolor       = 'rgb(255, 255, 255)',
                zerolinecolor   = 'rgb(255, 255, 255)',
                showbackground  = True,
                backgroundcolor = 'rgb(230, 230,230)',
                title           = 'y (mm)'
            ),
            zaxis=dict(
                gridcolor       = 'rgb(255, 255, 255)',
                zerolinecolor   = 'rgb(255, 255, 255)',
                showbackground  = True,
                backgroundcolor = 'rgb(230, 230,230)',
                title           = 'z (mm)'
            ),
            camera = dict(
                up = dict(
                    x = 0,
                    y = 0,
                    z = 1
                ),
                eye = dict(
                    x = -1.7428,
                    y = 1.0707,
                    z = 0.7100,
                )
            ),
            aspectratio = dict( x = 1, y = 1, z = 0.7),
            aspectmode = 'manual'
        ),
    )

    fig = dict(data = data, layout = layout)

    init_notebook_mode(connected=True)
    iplot(fig, filename='plot_track_real_fake')

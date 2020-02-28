__author__ = "Steve Ataucuri (stonescenter), Jefferson Fialho "
__copyright__ = "Sprace.org.br"
__version__ = "1.0.0"

import numpy as np
import pandas as pd
import datetime as dt
import matplotlib.pyplot as plt
import seaborn as sns
from copy import deepcopy

import plotly.figure_factory as ff
import plotly.graph_objs as go
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot

from pylab import *

from .transformation import *

class Timer():
 
    def __init__(self):
        self.start_dt = None
 
    def start(self):
        self.start_dt = dt.datetime.now()
 
    def stop(self):
        end_dt = dt.datetime.now()
        print('Time taken: %s' % (end_dt - self.start_dt))


##########################################
####                                  ####                      
####   FUNCTIONS FOR DISTANCES        ####
####                                  ####
##########################################

def position_3D_approximationn(y_true, y_predicted, cilyndrical=False):
    '''
        Return the closest points for every hit predicted with y_true and just the true values
    '''

    #this dataframe receives all X,Y,Z predicted considering a set of hists
    df3d = 0
    y_truecopy = np.copy(y_true)
    print(y_truecopy.shape)
    if not cilyndrical:
        df3d = pd.DataFrame({'x':y_predicted[:,0],'y':y_predicted[:,1],'z':y_predicted[:,2]})
        df3d['x-shortest'] = 0
        df3d['y-shortest'] = 0
        df3d['z-shortest'] = 0
        #df3d['dist'] = 0
    else:
        df3d = pd.DataFrame({'rho':y_predicted[:,0],'eta':y_predicted[:,1],'phi':y_predicted[:,2]})
        df3d['rho-shortest'] = 0
        df3d['eta-shortest'] = 0
        df3d['phi-shortest'] = 0
        #df3d['dist'] = 0

    #for each predicted hit, we will approximate to the closest hit considering gemoetric distance
    for index, row in df3d.iterrows():
        #obtain the row with least geometric distance between predicted row and original rows (in yclone)
        # se sao coordenadas cilindricas
        if not cilyndrical:        
            Xpred=df3d.loc[index, 'x']
            Ypred=df3d.loc[index, 'y']
            Zpred=df3d.loc[index, 'z']
        else:
            Xpred=df3d.loc[index, 'rho']
            Ypred=df3d.loc[index, 'eta']
            Zpred=df3d.loc[index, 'phi']
            
        #in this column we will create the geometric distance from all available hits and the current hit
        y_truecopy['dist'] = ( 
                                ((y_truecopy.iloc[:,0] - Xpred) **2) + 
                                ((y_truecopy.iloc[:,1] - Ypred) **2) + 
                                ((y_truecopy.iloc[:,2] - Zpred) **2) ).pow(0.5)

        y_truecopy=y_truecopy.sort_values(by=['dist'])

     
        # se sao coordenadas cilindricas
        if not cilyndrical:
            df3d.loc[index, 'x-shortest'] = y_truecopy.iloc[index,0]
            df3d.loc[index, 'y-shortest'] = y_truecopy.iloc[index,1]
            df3d.loc[index, 'z-shortest'] = y_truecopy.iloc[index,2]
        else:
            df3d.loc[index, 'rho-shortest'] = y_truecopy.iloc[index,0]
            df3d.loc[index, 'eta-shortest'] = y_truecopy.iloc[index,1]
            df3d.loc[index, 'phi-shortest'] = y_truecopy.iloc[index,2]            

    #df3d['dist'] = y_truecopy.iloc[0:,3]
    
    if not cilyndrical:
        
        df3d.drop('x', axis=1, inplace=True)
        df3d.drop('y', axis=1, inplace=True)
        df3d.drop('z', axis=1, inplace=True)
    else:
        df3d.drop('rho', axis=1, inplace=True)
        df3d.drop('eta', axis=1, inplace=True)
        df3d.drop('phi', axis=1, inplace=True) 
        
    #return the fourth hit of all tracks
    return(df3d)

def calculate_distances_vec(e1, e2):
    '''
        Calculate distances between two vectors
    '''
    # others ways:
    #d1 = np.linalg.norm(y_test.values[0]-predicted[0])
    #d2 = distance.euclidean(y_test.values[0], predicted[0])
    #d3 = calculate_distances_vec(y_test.values[0], predicted[0])
    # 
    return pow(
            (e1[0] - e2[0])**2 +
            (e1[1] - e2[1])**2 +
            (e1[2] - e2[2])**2 , 0.5)    

def get_shortest_points(y_true, y_predicted):
    '''
        Return the shortest points for every hit predicted with y_true 
    '''
    new_points = []
    tmp = 0
    neartest = 0
    br = np.array(y_predicted)
    ar = np.array(y_true) 
    
    for i in range(0, len(br)):
        tmp = 10000
        for j in range(0, len(ar)):
            d = calculate_distances_vec(br[i], ar[j])
            if(d<=tmp):             
                tmp = d
                neartest = ar[j]
            
        new_points.append(neartest)
  
    return new_points
        
def calculate_distances(y_true, y_predicted, y_hit_shortest, save_to):
    '''
        This function calculates distances between hits
        
        input:
            y_predicted : predicted hit
            y_hit_shortest: hit near of every predicted hit
            y_true: y_true or y_test
        
        return:
            d1 : distances between y_true and y_predicted
            d2 : distances between y_true and y_hit_shortest
        
    '''
    y_predicted = pd.DataFrame(y_predicted)
    
    dftemp = pd.DataFrame(index=range(len(y_true)),columns=range(12))
    dftemp[0]=y_true.iloc[:,0] # x
    dftemp[1]=y_true.iloc[:,1] # y 
    dftemp[2]=y_true.iloc[:,2] # z

    dftemp[3]=y_predicted.iloc[:,0]
    dftemp[4]=y_predicted.iloc[:,1]
    dftemp[5]=y_predicted.iloc[:,2]

    dftemp[6]=y_hit_shortest.iloc[:,0]
    dftemp[7]=y_hit_shortest.iloc[:,1]
    dftemp[8]=y_hit_shortest.iloc[:,2]

    
    #  y_true - y_predicted
    dftemp[9] =(((dftemp[0]-dftemp[3])**2)+((dftemp[1]-dftemp[4])**2)+((dftemp[2]-dftemp[5])**2)).pow(1./2)
    # y_true - y_hit_shortest
    dftemp[10]=(((dftemp[0]-dftemp[6])**2)+((dftemp[1]-dftemp[7])**2)+((dftemp[2]-dftemp[8])**2)).pow(1./2)
    # y_pred - y_shortest
    dftemp[11]=(((dftemp[3]-dftemp[6])**2)+((dftemp[4]-dftemp[7])**2)+((dftemp[5]-dftemp[8])**2)).pow(1./2)

    #print (dftemp.iloc[0:10,:])
    dftemp=dftemp.sort_values(by=[10], ascending=False)
    

    print ("Average Distance prediction (y_true - y_pred)" , dftemp[9].mean())
    print ("Average Distance Approximation (y_true - y_shortest)" , dftemp[10].mean())
    print ("Average Distance Approximation (y_pred - y_shortest)" , dftemp[11].mean())
    
    #predicted
    dist_pred = dftemp.iloc[:,9:10]
    # Approximated
    dist_approx = dftemp.iloc[:,10:11]
  

    return dist_pred, dist_approx

def calculate_distances_matrix(y_true, y_predicted):
    '''
        Distance calculation between two matrix
    '''
    dist = pow(
            (y_true[:,0] - y_predicted[:,0])**2 +
            (y_true[:,1] - y_predicted[:,1])**2 +
            (y_true[:,2] - y_predicted[:,2])**2 , 0.5)
    #dist.sort_values(ascending=False)

    return dist

##########################################
####                                  ####                      
####   FUNCTIONS FOR VISUALIZATION    ####
####                                  ####
##########################################

def plot_distances(d1, d2, save_to):
    
    sns.set(rc={"figure.figsize": (14, 10)})

    subplot(2,2,1)
    ax = sns.distplot(d1)

    subplot(2,2,2)
    ax = sns.distplot(d1, rug=True, hist=True, color="r")
    plt.savefig(save_to)
    plt.show()
    '''
    fig,axes=plt.subplots(1,2)
    sns.distplot(d1,ax=axes[0])
    plt.grid(True)
    sns.distplot(d2,rug=True,ax=axes[1], color="r")
    plt.show()
    '''

def plot_distances_plotly(d1, d2, save_to):
    '''
        it seens does not save offline
    '''
    hist_data = [d1, d2]

    group_labels = ['Distances predicted', 'Distances Approx']

    # Create distplot with custom bin_size
    #fig = ff.create_distplot(hist_data, group_labels, bin_size=.25, curve_type='normal')
    fig = ff.create_distplot(hist_data, group_labels, show_curve=False, bin_size=.2)

    fig['layout'].update(title='Distances')

    fig.show()

# function to convert tracks with just rho,eta,phi (cylindrical coordinates)
# hit information to x,y,z (cartesian coordinates)

def conv_slice_rhoetaphi_to_xyz(df_aux, n_hits = 5):
    pivot_tmp = 0
    for i in range(n_hits):
        pivot_tmp = i * 3
        rho = df_aux.iat[pivot_tmp + 0]
        eta = df_aux.iat[pivot_tmp + 1]
        phi = df_aux.iat[pivot_tmp + 2]
        if (rho != 0 and eta != 0 and phi != 0):
            x, y, z = convert_rhoetaphi_to_xyz(rho, eta, phi)
            df_aux.iat[pivot_tmp + 0] = x
            df_aux.iat[pivot_tmp + 1] = y
            df_aux.iat[pivot_tmp + 2] = z
    return df_aux

# function to convert tracks with just x,y,z (cartesian coordinates)
# hit information to rho,eta,phi (cylindrical coordinates)
def conv_slice_xyz_to_rhoetaphi(df_in, n_hits = 5):
    pivot_tmp = 0
    for i in range(n_hits):
        pivot_tmp = i * 3
        x = df_aux.iat[pivot_tmp + 0]
        y = df_aux.iat[pivot_tmp + 1]
        z = df_aux.iat[pivot_tmp + 2]
        if (x != 0 and y != 0 and z != 0):
            rho, eta, phi = convert_xyz_to_rhoetaphi(x, y, z)
            df_aux.iat[pivot_tmp + 0] = rho
            df_aux.iat[pivot_tmp + 1] = eta
            df_aux.iat[pivot_tmp + 2] = phi
    return df_aux
    
    
    
#function to plot tracks with just x,y,z hit information
def track_plot_xyz(list_of_df_in = [],
                   n_hits = 5,
                   cylindrical = False,
                   **kwargs):
 
    # deep copy to avoid linking with the original dataframe addresss
    list_of_df = deepcopy(list_of_df_in)
    
    n_tracks = 1
    title = 'Track plots'
    path = 'chart.html'
    opacity = 0.5
    marker_size = 3
    line_size = 3
    len_list_df = len(list_of_df)
    list_of_colors = ['red','blue', 'green', 'magenta', 'chocolate',
                      'teal', 'indianred', 'yellow', 'orange', 'silver']
    
    assert (len_list_df <= len(list_of_colors)), 'The list must contain less than 10 dataframes.'
    
    if kwargs.get('n_tracks'):
        n_tracks = kwargs.get('n_tracks')
        if n_tracks > list_of_df[0].shape[0]:
            n_tracks = abs(list_of_df[0].shape[0])
            wrn_msg = ('The number of tracks to plot is greater than the number of tracks in '
                       'the dataframe.\nn_tracks will be: ' +  str(n_tracks) + 
                       ' (the number of tracks in the dataset)')
            warnings.warn(wrn_msg, RuntimeWarning, stacklevel=2)

    if kwargs.get('title'):
        title = kwargs.get('title')
        
    if kwargs.get('opacity'):
        opacity = kwargs.get('opacity')
        if opacity > 1.0:
            opacity = 1.0
            wrn_msg = ('The opacity value is greater than 1.0\n'
                       'The opacity value is will be set with 1.0 value.')
            warnings.warn(wrn_msg, RuntimeWarning, stacklevel=2)
    
    if kwargs.get('marker_size'):
        marker_size = abs(kwargs.get('marker_size'))
        
    if kwargs.get('line_size'):
        line_size = abs(kwargs.get('line_size'))
    

    # Initializing lists of indexes
    selected_columns_x = np.zeros(n_hits)
    selected_columns_y = np.zeros(n_hits)
    selected_columns_z = np.zeros(n_hits)

    # Generating indexes
    for i in range(n_hits):
        pivot_tmp = i * 3
        selected_columns_x[i] = int(pivot_tmp + 0)
        selected_columns_y[i] = int(pivot_tmp + 1)
        selected_columns_z[i] = int(pivot_tmp + 2)

    # list of data to plot
    data = []
    track = [None] * n_tracks
    
    for i in range(len_list_df):
        try:
            df_name = str(list_of_df[i].name)
        except:
            df_name = 'track[' + str(i) + ']'
            warnings.warn('For a better visualization, set the name of dataframe to plot:'
                          '\nE.g.: df.name = \'track original\'', 
                          RuntimeWarning, stacklevel=2)
        
        for j in range(n_tracks):
            if cylindrical is True:
                # function to convert rho, eta,phi to x,y,z
                list_of_df[i].iloc[j, :] = conv_slice_rhoetaphi_to_xyz(list_of_df[i].iloc[j, :])
            track[j] = go.Scatter3d(
                x = list_of_df[i].iloc[j, selected_columns_x],
                y = list_of_df[i].iloc[j, selected_columns_y],
                z = list_of_df[i].iloc[j, selected_columns_z],
                name = df_name + ' ' + str(j),
                opacity = opacity,
                marker = dict(
                    size = marker_size,
                    opacity = opacity,
                    color = list_of_colors[i],
                ),
                line = dict(
                    color = list_of_colors[i],
                    width = line_size
                )
            )
            # append the track[i] in the list for plotting
            data.append(track[j])
    layout = dict(
        autosize = True,
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
                    x = 1,
                    y = 1,
                    z = -0.5
                ),
                eye = dict(
                    x = -1.7428,
                    y = 1.0707,
                    z = 0.7100,
                )
            ),
            aspectratio = dict( x = 1, y = 1, z = 1),
            aspectmode = 'manual'
        ),
    )
    fig =  go.Figure(data = data, layout = layout)
    #init_notebook_mode(connected=True)
    if kwargs.get('path'):
        path = kwargs.get('path')
        fig.write_html(path, auto_open=True)  
    
    return fig     
#function to plot more than one dataframes

def track_plot_list(list_of_df = [], **kwargs):
    
    global pivot, shift
    
    n_tracks = 1
    title = 'Track plots'
    path = 'chart.html'
    opacity = 0.5
    marker_size = 3
    line_size = 3
    len_list_df = len(list_of_df)
    list_of_colors = ['red','blue', 'green', 'magenta', 'chocolate',
                      'teal', 'indianred', 'yellow', 'orange', 'silver']
    
    assert (len_list_df <= len(list_of_colors)), 'The list must contain less than 10 dataframes.'
    
    if kwargs.get('n_tracks'):
        n_tracks = kwargs.get('n_tracks')
        if n_tracks > list_of_df[0].shape[0]:
            n_tracks = abs(list_of_df[0].shape[0])
            wrn_msg = ('The number of tracks to plot is greater than the number of tracks in '
                       'the dataframe.\nn_tracks will be: ' +  str(n_tracks) + 
                       ' (the number of tracks in the dataset)')
            warnings.warn(wrn_msg, RuntimeWarning, stacklevel=2)
                
    if kwargs.get('pivot'):
        pivot = kwargs.get('pivot')
    
    if kwargs.get('title'):
        title = kwargs.get('title')
        
    if kwargs.get('opacity'):
        opacity = kwargs.get('opacity')
        if opacity > 1.0:
            opacity = 1.0
            wrn_msg = ('The opacity value is greater than 1.0\n'
                       'The opacity value is will be set with 1.0 value.')
            warnings.warn(wrn_msg, RuntimeWarning, stacklevel=2)
    
    if kwargs.get('marker_size'):
        marker_size = abs(kwargs.get('marker_size'))
        
    if kwargs.get('line_size'):
        line_size = abs(kwargs.get('line_size'))
        
    
    
    dft_size = int(list_of_df[0].shape[1])
    len_xyz = int(dft_size/pivot)
    

    # Initializing lists of indexes
    selected_columns_x = np.zeros(len_xyz)
    selected_columns_y = np.zeros(len_xyz)
    selected_columns_z = np.zeros(len_xyz)

    # Generating indexes
    for i in range(len_xyz):
        selected_columns_x[i] = int(i * pivot + 0 + shift)
        selected_columns_y[i] = int(i * pivot + 1 + shift)
        selected_columns_z[i] = int(i * pivot + 2 + shift)

    # list of data to plot
    data = []
    track = [None] * n_tracks
    
    for i in range(len_list_df):
        try:
            df_name = str(list_of_df[i].name)
        except:
            df_name = 'track[' + str(i) + ']'
            warnings.warn('For a better visualization, set the name of dataframe to plot:'
                          '\nE.g.: df.name = \'track original\'', 
                          RuntimeWarning, stacklevel=2)
        
        for j in range(n_tracks):
            track[j] = go.Scatter3d(
                # Removing null values (zeroes) in the plot
                x = list_of_df[i].replace(0.0, np.nan).iloc[j, selected_columns_x],
                y = list_of_df[i].replace(0.0, np.nan).iloc[j, selected_columns_y],
                z = list_of_df[i].replace(0.0, np.nan).iloc[j, selected_columns_z],
                name = df_name + ' ' + str(j),
                opacity = opacity,
                marker = dict(
                    size = marker_size,
                    opacity = opacity,
                    color = list_of_colors[i],
                ),
                line = dict(
                    color = list_of_colors[i],
                    width = line_size
                )
            )
            # append the track[i] in the list for plotting
            data.append(track[j])
    layout = dict(
        #width    = 900,
        #height   = 750,
        autosize = True,
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
                    x = 1,
                    y = 1,
                    z = -0.5
                ),
                eye = dict(
                    x = -1.7428,
                    y = 1.0707,
                    z = 0.7100,
                )
            ),
            aspectratio = dict( x = 1, y = 1, z = 1),
            aspectmode = 'manual'
        ),
    )
    fig =  go.Figure(data = data, layout = layout)
    init_notebook_mode(connected=True)
    if kwargs.get('path'):
        path = kwargs.get('path')
        fig.write_html(path, auto_open=True)  
    else:
        iplot(fig)
        
        
        
def track_plot_hist(df_tb_plt, **kwargs):
    bar_color = 'blue'
    n_bins = 30
    title = 'Histogram of Tracks'
    path = 'histogram.html'
    x_title = 'x'
    y_title = 'y'
    
    if kwargs.get('bar_color'):
        bar_color = kwargs.get('bar_color')
    if kwargs.get('n_bins'):
        n_bins = kwargs.get('n_bins')
    if kwargs.get('title'):
        title = kwargs.get('title')
    if kwargs.get('x_title'):
        x_title = kwargs.get('x_title')
    if kwargs.get('y_title'):
        y_title = kwargs.get('y_title')
        
    fig = go.Figure()

    fig.add_trace(go.Histogram(x = df_tb_plt, nbinsx = n_bins, marker_color = bar_color))
    fig.update_layout(
        title = go.layout.Title(
            text = title,
            font = dict(
                    #family="Courier New, monospace",
                    size=18,
                    color='#000'
            ),
            xref = 'paper',
            x = 0
        ),
        xaxis = go.layout.XAxis(
            title = go.layout.xaxis.Title(
                text = x_title,
                font = dict(
                    #family="Courier New, monospace",
                    size = 18,
                    color = '#000'
                )
            )
        ),
        yaxis = go.layout.YAxis(
            title = go.layout.yaxis.Title(
                text = y_title,
                font = dict(
                    #family="Courier New, monospace",
                    size = 18,
                    color = '#000'
                )
            )
        )
    )

    #init_notebook_mode(connected=True)
    
    
    if kwargs.get('path'):
        path = kwargs.get('path')
        fig.write_html(path, auto_open = True)  
        #fig.show(renderer='html')
    else:
        fig.show()           
 
        
def track_plot_id(df_tb_plt, **kwargs):
    global pivot
    #pivot = 6
    track_color = 'red'
    n_tracks = 1
    title = 'Track plots'
    path = 'chart.html'
    track_id = 0
    
    if kwargs.get('track_color'):
        track_color = kwargs.get('track_color')
    
    if kwargs.get('n_tracks'):
        n_tracks = kwargs.get('n_tracks')
    
    if kwargs.get('pivot'):
        pivot = kwargs.get('pivot')
    
    if kwargs.get('title'):
        title = kwargs.get('title')
    
    if kwargs.get('track_id'):
        track_id = kwargs.get('track_id')
        title = 'plot of track #' + str(track_id)
    
    if kwargs.get('path'):
        path = kwargs.get('path')
        track_plot(df_tb_plt.iloc[track_id:track_id + 1,:],
               track_color=track_color,
               n_tracks=n_tracks,
               title = title,
               pivot=pivot,
               path=path)
    else:
        track_plot(df_tb_plt.iloc[track_id:track_id + 1,:],
               track_color=track_color,
               n_tracks=n_tracks,
               title = title,
               pivot=pivot)



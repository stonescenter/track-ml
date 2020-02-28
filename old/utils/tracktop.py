from scipy import special
import matplotlib.pyplot as plt
import random
import numpy as np
import numexpr as ne

import plotly.graph_objs as go
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
from transformation import *
import pandas as pd
import warnings, sys, os, fnmatch

from trackml.dataset import load_event
from trackml.dataset import load_dataset
from trackml.randomize import shuffle_hits
from trackml.score import score_event

#parameters for error
default_err = 0.03
default_err_fad = 0.174
default_len_err = 500
default_center = 1

#parameters for tracks, particles and hits
pivot = 11
shift = 1
particle_info = 9
ratio_discard_hit = 20
amount_of_hits = 20
n_columns_track = pivot * amount_of_hits + particle_info 

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
    factor = [1, -1]
    value_err = round(
        position * (random.choice(faddeev_array.real) ** random.choice(factor)), 8
    )
    return value_err


# fmt: on
# Flatten hits, pad the line to make up the original length,
# add back the index, vertex, momentum in the front
# INPUT: array of selected_hits, number of total_hits
# OUTPUT: array with dummy idx, vtx and momentum, sorted hits
def make_random_track(selected_hits, total_hits):
    flat_selected_hits = selected_hits.flatten()
    padded_selected_hits = np.pad(
        flat_selected_hits,
        (0, total_hits * 6 - flat_selected_hits.size),
        "constant",
        constant_values=0,
    )

    df = pd.DataFrame(padded_selected_hits)
    print(df)
    xyz_bsort(df)
    sorted_hits = np.asarray(df)

    candidate_track = np.concatenate(
        [np.array([-1]), np.array([0, 0, 0]), np.array([0, 0, 0]), sorted_hits]
    )
    return candidate_track


# fmt:off

def convert_track_xyz_to_rhoetaphi(df_in):

    len_xyz = df_in.shape[0] // pivot

    for i in range(len_xyz):
        pivot_tmp = i * pivot
        x = df_in.iloc[pivot_tmp + 0 + shift]
        y = df_in.iloc[pivot_tmp + 1 + shift]
        z = df_in.iloc[pivot_tmp + 2 + shift]
        if (x != 0 and y != 0 and z != 0):
            rho, eta, phi = convert_xyz_to_rhoetaphi(x, y, z)
            df_in.iloc[pivot_tmp + 0  + shift] = rho
            df_in.iloc[pivot_tmp + 1  + shift] = eta
            df_in.iloc[pivot_tmp + 2  + shift] = phi
            

def convert_track_rhoetaphi_to_xyz(df_in, df_out):

    len_xyz = df_in.shape[0] // pivot

    for i in range(len_xyz):
        pivot_tmp = i * pivot
        rho = df_in.iloc[pivot_tmp + 0 + shift]
        eta = df_in.iloc[pivot_tmp + 1 + shift]
        phi = df_in.iloc[pivot_tmp + 2 + shift]
        if (rho != 0 and eta != 0 and phi != 0):
            x, y, z = convert_rhoetaphi_to_xyz(rho, eta, phi)
            df_out.iloc[pivot_tmp + 0 + shift] = x
            df_out.iloc[pivot_tmp + 1 + shift] = y
            df_out.iloc[pivot_tmp + 2 + shift] = z
            
            
def np_rhoetaphi_to_xyz(np_in):

    np_out = np_in
    
    len_xyz = np_in.shape[0] // pivot

    for i in range(len_xyz):
        pivot_tmp = i * pivot
        rho = np_in[pivot_tmp + 0 + shift]
        eta = np_in[pivot_tmp + 1 + shift]
        phi = np_in[pivot_tmp + 2 + shift]
        if (x != 0 and y != 0 and z != 0):
            x, y, z = convert_rhoetaphi_to_xyz(rho, eta, phi)
            np_in[pivot_tmp + 0  + shift] = x
            np_in[pivot_tmp + 1  + shift] = y
            np_in[pivot_tmp + 2  + shift] = z    
    


def convert_track_etaphi_err(df_in, err_func = err_normal, **kwargs):
    err = default_err

    if kwargs.get('err'):
        err = kwargs.get('err')

    len_xyz = df_in.shape[0] // pivot

    for i in range(len_xyz):
        pivot_tmp = i * pivot
        x = df_in.iloc[pivot_tmp + 0 + shift]
        y = df_in.iloc[pivot_tmp + 1 + shift]
        z = df_in.iloc[pivot_tmp + 2 + shift]
        if (x != 0 and y != 0 and z != 0):
            rho, eta, phi = convert_xyz_to_rhoetaphi(x, y, z)
            eta = err_func(eta, err)
            phi = err_func(phi, err)

            x, y, z = convert_rhoetaphi_to_xyz(rho, eta, phi)

            df_in.iloc[pivot_tmp + 0 + shift] = x
            df_in.iloc[pivot_tmp + 1 + shift] = y
            df_in.iloc[pivot_tmp + 2 + shift] = z   
            
            
            
def geo_dist(x,y,z):
    return ne.evaluate('sqrt(x**2 + y**2 + z**2)')

# SWAP function to work with bubble sort
def xyz_swap(df_tb_swap, index_xyz, i, j, pivot):
    pivot_i = pivot * i
    pivot_j = pivot * j
    #swapping index array
    index_xyz[i], index_xyz[j] = index_xyz[j], index_xyz[i]
    #swapping tracks array
    for aux in range(pivot):
        df_tb_swap.iloc[pivot_i+aux], df_tb_swap.iloc[pivot_j+aux] = df_tb_swap.iloc[pivot_j+aux], df_tb_swap.iloc[pivot_i+aux]

#function to sort a line of a track dataset divided by hits with 8 elements
def xyz_bsort(df_to_be_sorted, df_n_col = amount_of_hits, **kwargs):
    global pivot 
    global shift

    if kwargs.get('pivot'):
        pivot = kwargs.get('pivot')

    index_xyz = []
    #df_n_col  = df_to_be_sorted.shape[0] // pivot

    for aux in range(0,df_n_col):
        pivot_tmp = pivot * aux + shift
        x = df_to_be_sorted.iloc[pivot_tmp + 0]
        y = df_to_be_sorted.iloc[pivot_tmp + 1]
        z = df_to_be_sorted.iloc[pivot_tmp + 2]
        if (x == 0. and y == 0. and z == 0.):
            index_xyz.append(np.PINF)
        else:
            index_xyz.append(geo_dist(x, y, z))
            
    for i in range(1, len(index_xyz)):
        for j in range(0, len(index_xyz) - i):
            if index_xyz[j] > index_xyz[j + 1]:
                xyz_swap(df_to_be_sorted, index_xyz, j, j + 1, pivot)

def np_xyz_swap(np_tb_swap, index_xyz, i, j, pivot):
    # SWAP function to work with bubble sort to work with np
    pivot_i = pivot * i
    pivot_j = pivot * j
    #swapping index array
    index_xyz[i], index_xyz[j] = index_xyz[j], index_xyz[i]
    #swapping tracks array
    for aux in range(pivot):
        np_tb_swap[pivot_i+aux], np_tb_swap[pivot_j+aux] = np_tb_swap[pivot_j+aux], np_tb_swap[pivot_i+aux]

def np_xyz_bsort(np_to_be_sorted, 
                 np_n_col = amount_of_hits, 
                 pivot = pivot, 
                 shift = shift):
    #function to sort a line of a track dataset divided by
    # hits with 8 elements to work if NP

    index_xyz = []

    for aux in range(0,np_n_col):
        pivot_tmp = pivot * aux + shift
        x = np_to_be_sorted[pivot_tmp + 0]
        y = np_to_be_sorted[pivot_tmp + 1]
        z = np_to_be_sorted[pivot_tmp + 2]
        
        if (x == 0. and y == 0. and z == 0.):
            index_xyz.append(np.PINF)
        else:
            index_xyz.append(geo_dist(x, y, z))
            
    for i in range(1, len(index_xyz)):
        for j in range(0, len(index_xyz) - i):
            if index_xyz[j] > index_xyz[j + 1]:
                np_xyz_swap(np_to_be_sorted, index_xyz, j, j + 1, pivot)
                

def f_range(value,float_range = [np.NINF,np.PINF]):
    # Function to return True if the float value is within range
    # or False if the float value is out of range  
    if float_range[0] <= value <= float_range[1]:
        return True
    return False
            

def track_filter(np_in, 
                 n_hits_track, 
                 n_hits_range, 
                 px, 
                 py,
                 eta_range, 
                 phi_range,
                 dif_delta_eta_range, 
                 dif_delta_phi_range,
                 delta_eta_mean_range,
                 delta_phi_mean_range,
                 pt_range):
    
    # Function to filter if pt, eta, phi, delta_eta or 
    # delta_phi is out of range. 
    # This function also filters if the track has no hits.

    # initializing the values and lists
    rho, eta, phi, dif_eta_total, dif_phi_total = 0, 0, 0, 0, 0
    list_eta = []
    list_phi = []
        
    # if n_hits_track == 0., 
    # this track has no hits.
    # the function will return False and 
    # the track will be discarted
    if (n_hits_track == 0.): return False
    
    # If  n_hits_track of n_hits_range, 
    # this function will return False and 
    # the track will be discarted
    if f_range(n_hits_track, n_hits_range) is False: return False 
      
    # Calculating pt
    pt = ne.evaluate('sqrt(px*px + py*py)')
    
    # If  pt is out of range, 
    # this function will return False and 
    # the track will be discarted
    if f_range(pt, pt_range) is False: return False

    for i in range(int(n_hits_track)):
        
        # Calculating the pivot to access each 
        # hit sequentially with each loop pass
        pivot_tmp = i * pivot + shift
        
        # initializing the values for rho, eta and phi
        rho, eta, phi = 0,0,0
        
        # Getting the values of x, y, z 
        # from each hit of this track
        x = np_in[pivot_tmp + 0]
        y = np_in[pivot_tmp + 1]
        z = np_in[pivot_tmp + 2]
        
        # generating the values of rho,eta,phi from x,y,z
        rho, eta, phi = convert_xyz_to_rhoetaphi(x, y, z)

        # If any value of eta is out of range, 
        # this function will return False and 
        # the track will be discarted
        if f_range(eta, eta_range) is False: return False

        # If any value of phi is out of range, 
        # this function will return False and 
        # the track will be discarted
        if f_range(phi, phi_range) is False: return False
        
        # accumulating the value of eta and phi from each hit
        list_eta.append(eta)
        list_phi.append(phi)
        
        # We need at least 2 hits to calculate the difference 
        # of eta. because of that we accumulated the first hit
        if i == 0:
            eta_prev = eta
            phi_prev = phi
        else:
            # We need at least 2 hits to calculate the difference 
            # of eta. because of that we accumulated the first hit
            dif_eta_total += abs(eta_prev - eta)
            dif_phi_total += abs(phi_prev - phi)
            
            # preparing the value for next loop
            eta_prev = eta
            phi_prev = phi
     
    # if this track has only 1 hit the difference in eta and phi is 0
    if (n_hits_track == 1.):
        dif_eta_avg = 0.
        dif_phi_avg = 0.
    else:
        # if this track has only 1 hit the difference in eta and phi is 0
        dif_eta_avg = dif_eta_total / (n_hits_track - 1)
        dif_phi_avg = dif_phi_total / (n_hits_track - 1)
        
    # If  dif_eta_avg is out of range, 
    # this function will return False and 
    # the track will be discarted
    if f_range(dif_eta_avg, dif_delta_eta_range) is False: return False
    
    # If  dif_phi_avg is out of range, 
    # this function will return False and 
    # the track will be discarted
    if f_range(dif_phi_avg, dif_delta_phi_range) is False: return False
    
    # initializing lists for compute tehe diferentes 
    # between hit_value and mean_value
    eta_mean = []
    phi_mean = []
    
    # the average on eta 
    # sum(eta_each_hit) / number_of_hits
    eta_avg = sum(list_eta)/n_hits_track
    
    # the average on phi 
    # sum(phi_each_hit) / number_of_hits
    phi_avg = sum(list_phi)/n_hits_track
    
    # creating a list of diferences 
    # between eta_each_hit - eta_avg
    for i in list_eta: eta_mean.append(i - eta_avg)
        
    # creating a list of diferences 
    # between phi_each_hit - phi_avg    
    for i in list_phi: phi_mean.append(i - phi_avg)
    
    # If  sum(eta_mean) is out of range, 
    # this function will return False and 
    # the track will be discarted
    if f_range(sum(eta_mean), delta_eta_mean_range) is False: return False
    
    # If  sum(eta_mean) is out of range, 
    # this function will return False and 
    # the track will be discarted
    if f_range(sum(phi_mean), delta_phi_mean_range) is False: return False
    
    return True
            
                
def create_input(dirParam,
                 fileParam,
                 n_hits_range = [0, np.PINF], 
                 eta_range = [np.NINF,np.PINF],
                 phi_range = [np.NINF,np.PINF], 
                 dif_delta_eta_range = [np.NINF,np.PINF],
                 dif_delta_phi_range = [np.NINF,np.PINF],
                 delta_eta_mean_range = [np.NINF,np.PINF],
                 delta_phi_mean_range = [np.NINF,np.PINF],
                 pt_range = [np.NINF,np.PINF],
                 ratio_discard_hit = ratio_discard_hit, 
                 sort = True, 
                 silent = False,
                 cylindrical = False,
                 pivot = pivot,
                 n_tracks = None,
                 path = None,
                 output = None):

    #9 columns for particle
    #7 columns for hit
    #one column for cell value (energy deposited)
    #9+20*8 = 169
    
    ###########################
    # DECLARATIONS AND INPUTS #
    ###########################
       
    hits, cells, particles, truth = load_event(os.path.join(dirParam,fileParam))
     
    # getting the standard parameters from tracktop lib
    global particle_info
    global n_columns_track
    global amount_of_hits
    
   
    discarted_tracks = 0
       
    # Getting the number of tracks from file
    
    if n_tracks == None: 
        n_tracks = particles.shape[0]
    else:
        if n_tracks > particles.shape[0]:
            n_tracks = particles.shape[0]
            wrn_msg = ('The number of tracks to plot is greater than the number of tracks in '\
                       'the file.\nn_tracks will be: ' +  str(n_tracks) + 
                       ' (the number of tracks in the file)')
            warnings.warn(wrn_msg, RuntimeWarning, stacklevel=2)

        
    
    df_input_nn = pd.DataFrame()
    row_final_matrix = np.zeros((1,1))
    total_discarded_hits = 0
    track_count = 0
        
    # Getting the parameters
    if path == None:
        path = fileParam + '_tracks.csv'
    
    if output != None:
        path = output

    ####################
    # PARAMETERS ERROR #
    ####################
    
    # sort
    assert ((sort is True) or 
            (sort is False)), '\'sort\' parameter must be True or False'
    
    # silent
    assert ((silent is True) or
            (silent is False)), '\'silent\' parameter must be True or False'
    
    
    # n_tracks 
    assert (type(n_tracks) == int), '\'n_tracks\' must be a integer value'
    err = ('\'n_tracks\' parameter must be a integer value greater than 0')
    assert (n_tracks > 0), err
    
    
    # eta_range    
    err = ('\'eta_range\' parameter must be a list with two \'int\''\
    'or \'float\' elements.\n E.g.: \'eta_range = [-4.5,4]\'')
    assert ((type(eta_range)==list) and 
            (len(eta_range) == 2) and 
            ((type(eta_range[0])==int) or (type(eta_range[0])==float)) and 
            ((type(eta_range[1])==int) or (type(eta_range[1])==float))), err
    
    # phi_range    
    err = ('\'phi_range\' parameter must be a list with two \'int\''\
    'or \'float\' elements.\n E.g.: \'phi_range = [-4.5,4]\'')
    assert ((type(phi_range)==list) and 
            (len(phi_range) == 2) and 
            ((type(phi_range[0])==int) or (type(phi_range[0])==float)) and 
            ((type(phi_range[1])==int) or (type(phi_range[1])==float))), err
    
    # dif_delta_eta_range    
    err = ('\'dif_delta_eta_range\' parameter must be a list with two \'int\''\
    'or \'float\' elements.\n E.g.: \'delta_eta_range = [-4.5,4]\'')
    assert ((type(dif_delta_eta_range)==list) and 
            (len(dif_delta_eta_range) == 2) and 
            ((type(dif_delta_eta_range[0])==int) or (type(dif_delta_eta_range[0])==float)) and 
            ((type(dif_delta_eta_range[1])==int) or (type(dif_delta_eta_range[1])==float))), err
    
    # dif_delta_phi_range    
    err = ('\'dif_delta_phi_range\' parameter must be a list with two \'int\''\
    'or \'float\' elements.\n E.g.: \'delta_phi_range = [-4.5,4]\'')
    assert ((type(dif_delta_phi_range)==list) and 
            (len(dif_delta_phi_range) == 2) and 
            ((type(dif_delta_phi_range[0])==int) or (type(dif_delta_phi_range[0])==float)) and 
            ((type(dif_delta_phi_range[1])==int) or (type(dif_delta_phi_range[1])==float))), err
    
    # delta_eta_mean_range    
    err = ('\'delta_eta_mean_range\' parameter must be a list with two \'int\''\
          'or \'float\' elements.\n E.g.: \'delta_eta_range = [-4.5,4]\'')
    assert ((type(delta_eta_mean_range)==list) and 
           (len(delta_eta_mean_range) == 2) and 
           ((type(delta_eta_mean_range[0])==int) or (type(delta_eta_mean_range[0])==float)) and 
           ((type(delta_eta_mean_range[1])==int) or (type(delta_eta_mean_range[1])==float))), err
    
    # delta_phi_mean_range  
    err = ('\'delta_phi_mean_range\' parameter must be a list with two \'int\''\
    'or \'float\' elements.\n E.g.: \'delta_eta_range = [-4.5,4]\'')
    assert ((type(delta_phi_mean_range)==list) and 
           (len(delta_phi_mean_range) == 2) and 
           ((type(delta_phi_mean_range[0])==int) or (type(delta_phi_mean_range[0])==float)) and 
           ((type(delta_phi_mean_range[1])==int) or (type(delta_phi_mean_range[1])==float))), err
           
    # pt_range    
    err = ('\'pt_range\' parameter must be a list with two \'int\''\
    'or \'float\' elements.\n E.g.: \'pt_range = [-4.5,4]\'')
    assert ((type(pt_range)==list) and 
            (len(pt_range) == 2) and 
            ((type(pt_range[0])==int) or (type(pt_range[0])==float)) and 
            ((type(pt_range[1])==int) or (type(pt_range[1])==float))), err
    
    # n_hits_range    
    err = ('\'n_hits_range\' parameter must be a list with two \'int\''\
    'or \'float\' elements.\n E.g.: \'n_hits_range = [-4.5,4]\'')
    assert ((type(n_hits_range)==list) and 
            (len(n_hits_range) == 2) and 
            ((type(n_hits_range[0])==int) or (type(n_hits_range[0])==float)) and 
            ((type(n_hits_range[1])==int) or (type(n_hits_range[1])==float))), err
        
    ####################
    #  Getting tracks  #
    ####################
        
    if silent is False: print('Getting tracks...')
    
    for index, row in particles.iloc[0:n_tracks,:].iterrows():
        dataP=row.values.flatten()
        #particle information 9 columns
        par = np.reshape(dataP,(1,9))
        #obtain particle track
        truth_0 = truth[truth.particle_id == row['particle_id']]
        prior_position = np.zeros(3)
        row_all_hits = par
        discarded_hit = 0
        #loop all hits in track
        for indexT, rowT in truth_0.iloc[:,:].iterrows():
            #obtain geometric distance, between hit and previous hit
            if ((prior_position[0] !=0) & (prior_position[1] !=0) & 
                (prior_position[2] !=0)):
                geo_dist_hit = geo_dist((rowT['tx'] - prior_position[0]),
                                        (rowT['ty'] - prior_position[1]),
                                        (rowT['tz'] - prior_position[2]))
            else:
                geo_dist_hit = 100
            # if geometric distance is below 20 mm discard hit
            if (geo_dist_hit > ratio_discard_hit):
                #truth information 9 columns
                data = rowT.values.flatten()
                hit = np.reshape(data,(1,9))
                hit_0 = hits[hits.hit_id == rowT['hit_id']]
                
                # if cylindrical is True the rho, eta, phi 
                # cilindrical coordinates will be stored in the 
                # dataset in place of the x, y, z cartesian coordinates
                #if cylindrical is True:
                # getting the x,y,z values from hit
                x_hit = hit_0.loc[:,'x']
                y_hit = hit_0.loc[:,'y']
                z_hit = hit_0.loc[:,'z']

                # getting rho,eta,phi from x,y,z
                rho, eta, phi = convert_xyz_to_rhoetaphi(x_hit, y_hit, x_hit)
                
                # reshaping rho, eta, phi to concatenate
                rho = np.reshape(rho.values.flatten(),(1,1))
                eta = np.reshape(eta.values.flatten(),(1,1))
                phi = np.reshape(phi.values.flatten(),(1,1))

                cells_0 = cells[cells.hit_id == rowT['hit_id']]
                sum_cells_aux = cells_0.iloc[:,3:4].sum()

                data_cells = sum_cells_aux.values.flatten()
   
                sum_cells = np.reshape(data_cells,(1,1))

                #row_final = np.concatenate((hit_0.iloc[:,0:8], sum_cells),axis=1)
                row_final = np.concatenate((hit_0.iloc[:,0:4], rho, eta, phi,
                                            hit_0.iloc[:,4:7], sum_cells),axis=1)
                #create row with particle and all hits
                row_all_hits = np.concatenate((row_all_hits,row_final),axis=1)
                prior_position[0] = rowT['tx']
                prior_position[1] = rowT['ty']
                prior_position[2] = rowT['tz']
            else:
                discarded_hit += 1
        total_discarded_hits += discarded_hit

        #complete empty hits in track with zeros
        aux_0 = np.zeros(n_columns_track - row_all_hits.shape[1])
        data_0 = aux_0.flatten()
        complete_0 = np.reshape(data_0,
                                (1,(n_columns_track - row_all_hits.shape[1])))
                
        #
        #create one line with complete row
        #
        
        row_all_hits_line = np.concatenate((row_all_hits,complete_0),axis=1)
        
        # Removing hits from n_tracks_hits dataset
        row_all_hits_line[0,8] -= discarded_hit
        
        px = row_all_hits_line[0,4]
        py = row_all_hits_line[0,5]
        
        #x,y,z from first track
        n_hits_track = row_all_hits_line[0,8]

        # Run track filter
        bool_filter_etaphipt = track_filter(row_all_hits_line[0,particle_info:],
                                            n_hits_track, n_hits_range, px, py, 
                                            eta_range, phi_range, 
                                            dif_delta_eta_range, 
                                            dif_delta_phi_range,
                                            delta_eta_mean_range, 
                                            delta_phi_mean_range, 
                                            pt_range)
        
        #if (bool_filter_etaphipt is True and n_hits_track != 0):
        if (bool_filter_etaphipt is True):

            # Sorting the track
            if (sort is True):
                np_xyz_bsort(row_all_hits_line[0,particle_info:],np_n_col = int(n_hits_track))

            # if is the first track of the file
            if ((row_final_matrix.shape[0] == 1) & 
                (row_final_matrix.shape[1] == 1)):
                row_final_matrix = row_all_hits_line
            else:
                row_final_matrix = np.concatenate((row_final_matrix, 
                                                  row_all_hits_line),axis=0)
        else:
            discarted_tracks += 1

        track_count += 1
        
        #Showing the status of file creation
        if n_tracks >=10:
            if (track_count % (n_tracks // 10) == 0) and silent is False:
                print (round(track_count / n_tracks * 100, 1), 
                       '% of ', n_tracks, " tracks.")
    
    #writing the final matrix
    df_input_nn = pd.DataFrame(row_final_matrix)

    # Creating a list with particle info
    track_header = ['particle_id', 'vx', 'vy', 'vz', 
                    'px', 'py', 'pz', 'q', 'n_hits']
        
    # Adding a header of each hit in a track_header list
    # rho, eta,phi
    for i in range(amount_of_hits):
        track_header.append('hit_id_' + str(i))
        track_header.append('x_' + str(i))
        track_header.append('y_' + str(i))
        track_header.append('z_' + str(i))
        track_header.append('rho_' + str(i))
        track_header.append('eta_' + str(i))
        track_header.append('phi_' + str(i))
        track_header.append('volume_id_' + str(i))
        track_header.append('layer_id_' + str(i))
        track_header.append('module_id_' + str(i))
        track_header.append('value_' + str(i))

    # Showing a error message if the track_filter cut oll the hits
    err = 'There are no tracks in the \'' + fileParam +'\' with the parameters that were entered.'
    assert (discarted_tracks != n_tracks), err
  
    # Writing the dataframe in the csv file.
    df_input_nn.to_csv(path, index = False, header = track_header)
    
    ####################
    #      OUTPUT      #
    ####################
    
    # showing the information of dataset and filtering
    if silent is False:
        print('\nTracks analised: ', n_tracks)
        print('Total discarded tracks: ', discarted_tracks)
        print('Total discarded hits: ', total_discarded_hits)
        print('Shape of dataset: ', df_input_nn.shape)
        print('Sorted: ',  sort)
        print('Dataset saved at: ',  path)
##########################################
####                                  ####                      
####   FUNCTIONS FOR VISUALIZATION    ####
####                                  ####
##########################################


#function to plot tracks
def track_plot(df_tb_plt, **kwargs):
    
    global pivot, shift
    
    track_color = 'red'
    n_tracks = 1
    title = 'Track plots'
    path = 'chart.html'
    marker_size = 2
    line_size = 2
    
    if kwargs.get('track_color'):
        track_color = kwargs.get('track_color')
    
    if kwargs.get('n_tracks'):
        n_tracks = kwargs.get('n_tracks')
        if n_tracks > df_tb_plt.shape[0]:
            n_tracks = abs(df_tb_plt.shape[0])
            wrn_msg = ('The number of tracks to plot is greater than the number of tracks in '
                       'the dataframe.\nn_tracks will be: ' +  str(n_tracks) + 
                       ' (the number of tracks in the dataset)')
            warnings.warn(wrn_msg, RuntimeWarning, stacklevel=2)
    
    if kwargs.get('pivot'):
        pivot = kwargs.get('pivot')
    
    if kwargs.get('title'):
        title = kwargs.get('title')
        
    if kwargs.get('marker_size'):
        marker_size = abs(kwargs.get('marker_size'))
        
    if kwargs.get('line_size'):
        line_size = abs(kwargs.get('line_size'))
    
    dft_size = df_tb_plt.shape[1]
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
                size = marker_size,
                color = track_color,
            ),
            line = dict(
                color = track_color,
                width = line_size
            )
        )
        # append the track[i] in the list for plotting
        data.append(track[i])
        
        
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


    fig =  go.Figure(data = data,
                     layout = layout)
    
    
    init_notebook_mode(connected=True)
    
    
    if kwargs.get('path'):
        path = kwargs.get('path')
        fig.write_html(path, auto_open=True)  
        #fig.show(renderer='html')
    else:
        iplot(fig)
        

        
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
    


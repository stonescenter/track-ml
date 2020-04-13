__author__ = "unknow"
__copyright__ = "Sprace.org.br"
__version__ = "1.0.0"

from scipy import special
import matplotlib.pyplot as plt
import random
import numpy as np
import numexpr as ne

import pandas as pd
import warnings, sys, os, fnmatch

from trackml.dataset import load_event
from trackml.dataset import load_dataset
from trackml.randomize import shuffle_hits
from trackml.score import score_event

from .transformation import * 

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
            

def track_filter_old(np_in, 
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
            
                
def create_input_old(dirParam,
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
                rho, eta, phi = convert_xyz_to_rhoetaphi(x_hit, y_hit, z_hit)
                
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
        bool_filter_etaphipt = track_filter_old(row_all_hits_line[0,particle_info:],
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
    
#################################################################
###  new version of creat input with eta_phi filter improved  ###
################################################################

def track_filter(np_in, 
                 n_hits_track, 
                 n_hits_range, 
                 px, 
                 py,
                 eta_range, 
                 phi_range,
                 pt_range):
    
    # Function to filter if pt, eta, phi, delta_eta or 
    # delta_phi is out of range. 
    # This function also filters if the track has no hits.

    # initializing the values and lists
    rho, eta, phi = 0, 0, 0
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
        
    
    pivot_tmp = int((n_hits_track - 1) * pivot)
    #print('n_hits_track: ',n_hits_track)
    #print('pivot_tmp: ',pivot_tmp)
    
    eta = np_in[pivot_tmp + 5]
    phi = np_in[pivot_tmp + 6]
  
    # If any value of eta is out of range, 
    # this function will return False and 
    # the track will be discarted
    #if f_range(eta, eta_range) is False: 
    #    print('Return ETA: ',f_range(eta, eta_range))
    #    print('ETA: ',eta)
    #    return False
    
    eta_validation = f_range(eta, eta_range) 
    #print('Return eta: ',eta_validation,'\nETA: ',eta)
    
    if (eta_validation is False): return False   
        
    # If any value of phi is out of range, 
    # this function will return False and 
    # the track will be discarted
    
    phi_validation = f_range(phi, phi_range)
    #print('Return PHI: ',phi_validation,'\nPHI: ',phi)
    
    if (phi_validation is False): return False
    

    return True
            

def create_input(dirParam,
                 fileParam,
                 n_hits_range = [0, np.PINF], 
                 eta_range = [np.NINF,np.PINF],
                 phi_range = [np.NINF,np.PINF],
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
                rho, eta, phi = convert_xyz_to_rhoetaphi(x_hit, y_hit, z_hit)
                
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

        
        
        # Sorting the track
        if ((sort is True) and (n_hits_track != 0)):
            np_xyz_bsort(row_all_hits_line[0,particle_info:],np_n_col = int(n_hits_track))

        # Run track filter

        bool_filter_etaphipt = track_filter(row_all_hits_line[0,particle_info:],
                                            n_hits_track,
                                            n_hits_range, 
                                            px, 
                                            py, 
                                            eta_range, 
                                            phi_range, 
                                            pt_range)
            
        #if (bool_filter_etaphipt is True and n_hits_track != 0):
        if (bool_filter_etaphipt is True):

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


#######################################################        
#######################################################
##                                                   ##
##   lib to get predict hit (x,y,z coord) e return   ## 
##       the coordinates from nearest real hit       ##
##                                                   ##
#######################################################
#######################################################


def nearest_hit(input_hits = None, 
                predict_hit_coord = None, 
                std_df = True, 
                engine = 'numpy', 
                metric = 'sqeuclidean',
                n_jobs = 96,
                silent = True,
                dist_hit = False):   

    import time
    import numpy as np
    from scipy.spatial import distance
    from sklearn.cluster import AgglomerativeClustering
    from sklearn.metrics import pairwise_distances
    from sklearn.metrics import pairwise_distances_argmin
    
    global pivot
    global shift
    len_xyz_values = 3
    
    # create a list with x,y,z positions
    if std_df is True:
        len_xyz = input_hits.shape[0] // pivot
        list_xyz= []

        for i in range(len_xyz):
            pivot_tmp = i * pivot
            x = pivot_tmp + 0 + shift
            y = pivot_tmp + 1 + shift
            z = pivot_tmp + 2 + shift
            list_xyz.append(x)
            list_xyz.append(y)
            list_xyz.append(z)

        list_hits = list_xyz[:30]
        
        #removing particle information
        input_hits = input_hits.iloc[:,9:]

        #get just xyz coord
        input_hits = input_hits.iloc[:,list_hits]
        
    # reshaping the dataframe to create a bag of hits
    
    len_rows_before_reshape = input_hits.shape[0]
    len_cols_before_reshape = input_hits.shape[1]

    len_rows_after_reshape = int(len_rows_before_reshape * len_cols_before_reshape / len_xyz_values)
    
    bag_of_hits = input_hits.values.reshape(len_rows_after_reshape,len_xyz_values)
    bag_of_hits = bag_of_hits[~np.all(bag_of_hits == 0, axis=1)]
    
    # initialize the time
    start_time = time.time()
    
    if engine == 'numpy':
        dist = distance.cdist(bag_of_hits, predict_hit_coord, metric)
    
    if engine == 'sklearn':
        dist = pairwise_distances(bag_of_hits, predict_hit_coord, metric = metric, n_jobs = n_jobs)
    
    # get the index of minimum distance
    target_hit_index = np.argmin(dist)
    
    if silent is False:    
        print("--- N_hits: %s" % bag_of_hits.shape[0])
        print("--- Hit index: %s" % target_hit_index)
        print("--- " + str(metric) + " distance: " + str(dist[target_hit_index]))
        print("--- time: %s seconds" % (time.time() - start_time))
    
    # get hits coordinates 
    real_hit = bag_of_hits[target_hit_index, :]
    
    # removing the hit from bag
    bag_of_hits = np.delete(bag_of_hits, target_hit_index, 0)
    
    if dist_hit is False:
        return real_hit
    else:
        return real_hit,np.min(dist)


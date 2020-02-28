__author__ = "unknow"
__copyright__ = "Sprace.org.br"
__version__ = "1.0.0"

import sys, os, fnmatch

import pandas as pd
import numpy as np
import sys
import random

from trackml.dataset import load_event
from trackml.dataset import load_dataset
from trackml.randomize import shuffle_hits
from trackml.score import score_event

import multiprocessing
from multiprocessing import Process, Value, Lock
import glob, os

#sys.path.append('/home/silvio/github/track-ml-1/utils')
#from tracktop import *

#obtain amount of columns
def amount_of_columns(cell):

    indt=0
    test=0
    indret=0

    for z in cell:
        indt=indt+1
        #print("z")

        if ((z == 0) and (test == 0)) :
            test=1
            indret=indt
    return(indret) # ind is the  amount of columns =! 0

def create_directory_for_results(temporary_directory):
    if (os.path.isdir(temporary_directory)):
        temp_dir = temporary_directory+"//*"
        files_in_temp_dir = glob.glob(temp_dir)

        for file in files_in_temp_dir:
            #print("remove ", file)
            os.remove(file)
    else:
        os.mkdir(temporary_directory)

def join_files(temporary_directory, output_file_real):

    files = []
    for r, d, f in os.walk(temporary_directory):
        for file in f:
            files.append(os.path.join(r, file))

    with open(output_file_real, 'w') as outfile:
        for fname in files:
            with open(fname) as infile:
                outfile.write(infile.read())

#function to put particle informatio and all the hits of each track in a single line
def create_tracks(tot_columns , hits, cells, particles, truth, be,e,pid,temporary_directory):

    b = np.zeros((0))

    for index, row in particles.iloc[be:e,:].iterrows():

        truth_0 = truth[truth.particle_id == row['particle_id']]

        par=particles[['vx','vy','vz','px','py','pz']].loc[particles['particle_id'] == row['particle_id']]
        particleRow = [par['vx'].values[0],par['vy'].values[0],par['vz'].values[0],par['px'].values[0],par['py'].values[0],par['pz'].values[0]]

        psize=par.size

        b = np.concatenate((b, particleRow))

        #print(truth_0.size)
        #print(truth_0.shape)
        h = np.zeros((0))
        #jj=0
        for index, row in truth_0.iterrows():

            ch=cells[['ch0']].loc[cells['hit_id'] == row['hit_id']].mean()
            ch1=cells[['ch1']].loc[cells['hit_id'] == row['hit_id']].mean()
            vl=cells[['value']].loc[cells['hit_id'] == row['hit_id']].mean()

            hitRow = [row['tx'],row['ty'],row['tz'],ch[0], ch1[0], vl[0]]
            h= np.concatenate((h, hitRow))

        hsize=h.size
        b=np.concatenate((b, h))

        aux = np.zeros((0))
        remaing_columns_to_zero=tot_columns-1-h.size-6
        if (remaing_columns_to_zero > 0):
            aux = np.zeros(remaing_columns_to_zero)
            auxsize=aux.size
            b=np.concatenate((b, aux))
        #print("bb ", b)

    #print("psize ", psize, "hsize ", hsize, "auxsize ", auxsize, "sum ", psize+hsize+auxsize)

    rw=(e-be)
    b = b.reshape(rw, (tot_columns-1))
    np.savetxt(temporary_directory+"//arr"+str(pid), b, fmt="%s")

def createTracks(event_prefix , dir_event_prefix, diroutput):
    #global Am_of_cores
    #global Am_of_particles
    #global total_of_loops
    #global remaining_tracks
    print(dir_event_prefix + " - " + event_prefix)
    hits, cells, particles, truth = load_event(os.path.join(dir_event_prefix, event_prefix))

    #X = np.zeros((0))

    #121 columns -> 6 particles columns; 19 hits (6 columns); um result columns (fake or real) ==> 6x19 + 6 +1 =121
    #tot_columns=121
    tot_columns=175

    Am_of_particles  = particles.shape[0]
    Am_of_cores      = multiprocessing.cpu_count()-2
    total_of_loops   = Am_of_particles // Am_of_cores
    remaining_tracks = (Am_of_particles-(total_of_loops*Am_of_cores))

    #output_file_all = "/data/output/TracksRealFake"+str(event_prefix)+".csv"
    #output_file_real = "/data/output/"+str(dir_event_prefix)+"/TracksReal"+str(event_prefix)+".csv"
    output_file_real = str(diroutput)+"/TracksReal"+str(event_prefix)+".csv"
    #output_file_real_aux = "/data/output/TracksRealAUX"+str(event_prefix)+".csv"
    #output_file_fake = "/data/output/TracksFake"+str(event_prefix)+".csv"
    temporary_directory = "/tmp/res/"+str(event_prefix)+"/"


    #output_file_all = "/data/output/TracksRealFake"+str(event_prefix)+".csv"
    #output_file_real = "/data/output/TracksReal"+str(event_prefix)+".csv"
    #output_file_real_aux = "/data/output/TracksRealAUX"+str(event_prefix)+".csv"
    #output_file_fake = "/data/output/TracksFake"+str(event_prefix)+".csv"

    #print("temporary_directory: ", temporary_directory)
    #print("Amount of Particles: ", Am_of_particles)
    #print("Amount of Processing cores: ", Am_of_cores)
    #print("total of loops: ", total_of_loops)
    #print("remaing tracks : ", remaining_tracks)
    print("output_file_real : " , output_file_real)

    step=1
    pid=0

    create_directory_for_results(temporary_directory)

    jobs = []
    for i in range(Am_of_cores+1):
    #for i in range(1):

        b=i*total_of_loops

        if (i == Am_of_cores):
            e=b+remaining_tracks
        else:
            e=b+total_of_loops
        #e=10
        #b=1

        p = multiprocessing.Process(target=create_tracks, args=(tot_columns, hits, cells, particles, truth,b,e,pid,temporary_directory))
        #p = multiprocessing.Process(target=count_hits, args=(b,e,pid,temporary_directory))

        #print ("multiprocessing: ", b,e)
        pid=pid+1
        jobs.append(p)
        p.start()

    for proc in jobs:
        proc.join()

    del jobs[:]

    join_files(temporary_directory, output_file_real)
    tracks = pd.read_csv(output_file_real,header = None, sep = " ")
    tracks.to_csv(output_file_real)

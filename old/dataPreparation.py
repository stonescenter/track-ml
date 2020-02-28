import sys
import os

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

#def load_cms_data():
    #using trackml to read the challenge input data

    #event_prefix = 'event000002290'
#    event_prefix = 'event000002878'
#    hits, cells, particles, truth = load_event(os.path.join('/data/trackMLDB/train_2/train_2', event_prefix))
    #type(cells)

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

#select Random columns
def select_random_columns(cols_num):
    cols  = np.zeros((cols_num)) # cols is the  that will be switched
    aux=0
    rand=0
    for y in cols: # for total amount of columns that will be switched
        rand=random.randrange(6, cols_num)
        #print("rand")
        #print(rand)
        cols[aux]=rand # define which column will be switched
        aux=aux+1
    return(cols)

#switch random columns
def switch_random_column( randomrow , currentrow, column_to_switch ):
    #if ((randomrow != 0) and (currentrow !=0) and (cols !=0)):
    #if (currentrow >0):
    #if ((cols.size > 0)):
    #global X
    #auxx = 0
    #for column_to_switch in cols:
        #print("currentrow")
        #print(currentrow)
        #print("randomrow")
        #print(randomrow)
        #print("column_to_switch")
        #print(column_to_switch)

    auxx = X[int(currentrow)][int(column_to_switch)]
    X[int(currentrow)][int(column_to_switch)] = X[int(randomrow)][int(column_to_switch)]
    X[int(randomrow)][int(column_to_switch)] = auxx

#function to put particle informatio and all the hits of each track in a single line
def create_tracks(be,e,pid,temporary_directory):

    b = np.zeros((0))

    for index, row in particles.iloc[be:e,:].iterrows():

        truth_0 = truth[truth.particle_id == row['particle_id']]

        par=particles[['vx','vy','vz','px','py','pz']].loc[particles['particle_id'] == row['particle_id']]
        particleRow = [par['vx'].values[0],par['vy'].values[0],par['vz'].values[0],par['px'].values[0],par['py'].values[0],par['pz'].values[0]]

        psize=par.size

        b = np.concatenate((b, particleRow))

        h = np.zeros((0))
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

    #print("psize ", psize, "hsize ", hsize, "auxsize ", auxsize, "sum ", psize+hsize+auxsize)

    rw=(e-be)
    b = b.reshape(rw, (tot_columns-1))
    np.savetxt(temporary_directory+"//arr"+str(pid), b, fmt="%s")

def create_directory_for_results(temporary_directory):
    if (os.path.isdir(temporary_directory)):
        temp_dir = temporary_directory+"//*"
        files_in_temp_dir = glob.glob(temp_dir)

        for file in files_in_temp_dir:
            print("remove ", file)
            os.remove(file)
    else:
        os.mkdir(temporary_directory)

def join_files(temporary_directory):
    #join all files created by several process
    #path = '/tmp/res/'
    files = []
    for r, d, f in os.walk(temporary_directory):
        for file in f:
            files.append(os.path.join(r, file))

    #with open(temporary_directory+'//merged_1', 'w') as outfile:
    with open(output_file_real, 'w') as outfile:
        for fname in files:
            with open(fname) as infile:
                outfile.write(infile.read())

def createFakes():
    global X
    percentage_of_fake_rows=90

    #c=np.loadtxt(open(temporary_directory+"/merged_1"))
    c=np.loadtxt(output_file_real)

    AmountOfFakeRows=int(c.shape[0]*percentage_of_fake_rows/100)
    print("Amount Of Fake Rows that will be created: ", AmountOfFakeRows)

    #track without particle information only hits
    aux=c[:AmountOfFakeRows,]
    X=aux[:AmountOfFakeRows,6:120]
    aux2=aux[:AmountOfFakeRows,0:6]

    #rand_cols_set=[0,1,2]
    rand_cols_set = [0,1,2,6,7,8,12,13,14,18,19,20,24,25,26,30,31,32,36,37,38,42,43,44,48,49,50,54,55,56,60,61,62,66,67,68,72,73,74,78,79,80,84,85,86,90,91,92,96,97,98,102,103,104,108,109,110]
    amount_of_cols_than_can_be_changed=len(rand_cols_set)
    currentrow=0
    for cell in X: # for each row in fake dataset
        cont=0
        changed = 0
        if (X[int(currentrow)][0] != 0.0):
            while changed < 1:

                rand_col = random.randrange(0, amount_of_cols_than_can_be_changed)
                randomrow=random.randrange(1, AmountOfFakeRows)

                rand_col_to_change = rand_cols_set[rand_col]

                if ((X[int(currentrow)][int(rand_col_to_change)] != 0.0) and (X[int(randomrow)][int(rand_col_to_change)] != 0.0)):
                    switch_random_column( randomrow , currentrow, rand_col_to_change )
                    changed += 1
                    cont    += 1
                    #print(cont)
        currentrow=currentrow+1

    xtotnp=np.concatenate((aux2, X), axis=1)

    vfinalReal = np.hstack((c, np.ones((c.shape[0], 1), dtype=c.dtype)))
    vfinalFake = np.hstack((xtotnp, np.zeros((xtotnp.shape[0], 1), dtype=xtotnp.dtype)))

    dtpd = pd.DataFrame(data=vfinalFake)
    dtpd.to_csv(output_file_fake)

    ds=np.concatenate((vfinalReal, vfinalFake), axis=0)
    np.random.shuffle(ds)
    dtpd = pd.DataFrame(data=ds[0:,0:])
    dtpd.to_csv(output_file_all)

def createTracks():
# This is the Main function

    global Am_of_particles
    global Am_of_cores
    global total_of_loops
    global remaining_tracks

    output_file_all = "/data/output/TracksRealFake"+str(event_prefix)+".csv"
    output_file_real = "/data/output/TracksReal"+str(event_prefix)+".csv"
    output_file_fake = "/data/output/TracksFake"+str(event_prefix)+".csv"

    step=1
    pid=0

    create_directory_for_results(temporary_directory)

    jobs = []
    for i in range(Am_of_cores+1):
    #for i in range(3):

        b=i*total_of_loops

        if (i == Am_of_cores):
            e=b+remaining_tracks
        else:
            e=b+total_of_loops
        #e=100
        #b=1

        p = multiprocessing.Process(target=create_tracks, args=(b,e,pid,temporary_directory))
        pid=pid+1
        jobs.append(p)
        p.start()

        for proc in jobs:
            proc.join()

        del jobs[:]

    join_files(temporary_directory)



    #print(aux[0,0:6])
    #aux=c.sample(n=AmountOfFakeRows)
    #aux c[np.random.randint(c.shape[0], size=2), :]
    #print(X[0,0:10])
    #print(xtotnp[0,0:10])
    #print(xtotnp.shape[0])
    #print(xtotnp.shape[1])

    #print(vfinal.shape[0])
    #print(vfinal.shape[1])
    #print(vfinal[0,120])

    #print(xtotnp[0,120])

    #np.c_[ xtotnp, np.ones(xtotnp.shape[0]) ]
    #xtotnp[:, tot_columns:]=int(1)
    #amm=xtotnp.shape[0]
    #yy = np.ones((xtotnp.shape[0], 1))
    #yy = np.ones((xtotnp.shape[0]))
    #print("yy.shape[0] " , yy.shape[0])
    #print("yy.shape[1] ", yy.shape[1])
    #yy = np.zeros(amm,1, dtype=int)
    #yy.reshape(1, xtotnp.shape[1])

    #print("yy.shape[0] " , yy.shape[0])
    #print("yy.shape[1] ", yy.shape[1])

    #np.append(xtotnp, yy, axis=1)
    #print(xtotnp.shape[0])
    #print(xtotnp.shape[1])

    #print(xtotnp[0,120])

    #currentrow=0


    #for cell in X: # for each row in fake dataset
    #    rand_col = random.randrange(0, 114)
    #    randomrow=random.randrange(1, AmountOfFakeRows)
    #    switch_random_column( randomrow , currentrow, rand_col )
    #    currentrow=currentrow+1

    '''
        (df == 0).astype(int).sum(axis=1)

        indd=amount_of_columns(cell)
        if (indd > 5) :
            cols_num = random.randrange(5, indd) #cols_num is the  amount of columns that will be switched
            cols = select_random_columns(cols_num)

            randomrow=random.randrange(1, AmountOfFakeRows)
            switch_random_columns( randomrow , currentrow, cols )
            currentrow=currentrow+1

            X[:, tot_columns-1:]=int(1)
    '''
    '''
    print("currentrow ", currentrow )
    aux2=aux[:AmountOfFakeRows,0:6]
    print(aux2.shape[0])
    print(aux2.shape[1])
    #c = np.concatenate((a, b), 1)
    xtotnp=np.concatenate((aux2, X), axis=1)

    print("xtotnp.shape[0]", xtotnp.shape[0])
    print("xtotnp.shape[1]", xtotnp.shape[1])

    xtotnp[:, tot_columns+1:]=int(0)
    c[:, tot_columns+1:]=int(1)

    print(xtotnp.shape[0])
    print(xtotnp.shape[1])

    print(c.shape[0])
    print(c.shape[1])

    ds=np.concatenate((c, xtotnp), axis=0)

    np.random.shuffle(ds)

    dtpd = pd.DataFrame(data=ds[0:,0:])#,    # values

    dtpd.to_csv(output_file_path)
    '''
    #dtpd.head(2)
#file with track events
print ("This is the name of the script: ", sys.argv[0])
print ("event_prefix: ", sys.argv[1])
#event_prefix = 'event000002878'
event_prefix = sys.argv[1]
#hits, cells, particles, truth = load_event(os.path.join('/data/trackMLDB/train_2/train_2', event_prefix))
hits, cells, particles, truth = load_event(os.path.join('/data/trackMLDB/train_1', event_prefix))

X = np.zeros((0))

#121 columns -> 6 particles columns; 19 hits (6 columns); um result columns (fake or real) ==> 6x19 + 6 +1 =121
#tot_columns=121
tot_columns      = 121
Am_of_particles  = particles.shape[0]
Am_of_cores      = multiprocessing.cpu_count()-2
total_of_loops   = Am_of_particles // Am_of_cores
remaining_tracks = (Am_of_particles-(total_of_loops*Am_of_cores))
output_file_all = "/data/output/TracksRealFake"+str(event_prefix)+".csv"
output_file_real = "/data/output/TracksReal"+str(event_prefix)+".csv"
output_file_fake = "/data/output/TracksFake"+str(event_prefix)+".csv"
temporary_directory = "/tmp/res/"+str(event_prefix)+"/"

print("output_file_all: ", output_file_all)
print("output_file_real: ", output_file_real)
print("output_file_fake: ", output_file_fake)

print("temporary_directory: ", temporary_directory)
print("Amount of Particles: ", Am_of_particles)
print("Amount of Processing cores: ", Am_of_cores)
print("total of loops: ", total_of_loops)
print("remaing tracks : ", remaining_tracks)

createTracks()
createFakes()

print("end execution")

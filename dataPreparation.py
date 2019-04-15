#coment about dependencies
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
        if ((z == 0) and (test==0)) : 
            test=1
            indret=indt
    return(indret) # ind is the  amount of columns =! 0

#select Random columns
def select_random_columns(cols_num):
    cols  = np.zeros((cols_num)) # cols is the  that will be switched
    aux=0
    rand=0
    for y in cols: # for total amount of columns that will be switched
        rand=random.randrange(1, cols_num)    
        #print("rand")
        #print(rand)
        cols[aux]=rand # define which column will be switched
        aux=aux+1
    return(cols)

#switch random columns
def switch_random_columns( randomrow , currentrow, cols ):
    #if ((randomrow != 0) and (currentrow !=0) and (cols !=0)):
    #if (currentrow >0):
    #if ((cols.size > 0)):
    global X
    auxx = 0
    for column_to_switch in cols:
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
def create_tracks(be,e,pid):
    #global ttt
    #print(be)    
    c = np.zeros((0))
    
    #for index, row in particles.head(tot_rows).iterrows():
    for index, row in particles.iloc[be:e,:].iterrows():
    
        b = np.zeros((0))

        truth_0 = truth[truth.particle_id == row['particle_id']]
        par=particles[['vx','vy','vz','px','py','pz']].loc[particles['particle_id'] == row['particle_id']]

        p = [par['vx'].values[0],par['vy'].values[0],par['vz'].values[0],par['px'].values[0],par['py'].values[0],par['pz'].values[0]]

        b= np.concatenate((b, p))

        for index, row in truth_0.iterrows():
            ch=cells[['ch0']].loc[cells['hit_id'] == row['hit_id']].mean()
            ch1=cells[['ch1']].loc[cells['hit_id'] == row['hit_id']].mean()
            vl=cells[['value']].loc[cells['hit_id'] == row['hit_id']].mean()
    
            h = [row['tx'],row['ty'],row['tz'],ch[0], ch1[0], vl[0]]
            b= np.concatenate((b, h))

        toti=tot_columns-b.size
        cc = np.zeros((toti))
        tot=np.concatenate((b, cc))
            #if (len(tot) != 121) :
            #    print("len(b) " , len(b) , " len(cc) " , len(cc) , "len(tot) " , len(tot))

        c = np.concatenate((c, tot))    
    #print(e)
    #print(be)
    rw=((e-be))
    #print(rw)
    
    #print(tot_columns)

    #print(c)
    c = c.reshape((rw, tot_columns))
    #print(pid)
    np.savetxt("//tmp//res//arr"+str(pid), c, fmt="%s")
    
def main():
    global X
    #load_cms_data()

    #Minimum amount of hits
    #print(particles['nhits'].min())
    #Maximum amount of hits
    #print(particles['nhits'].max())
    #print(particles.size)
    #print(particles.head(5))

    #load correct tracks

    #particles 6 fields
    #hit id 6 fields x 19 -> maximum amount of hits = 114
    #1 - output => 0 real 1 fake
    # total 121

    perc=40

    #tot_rows=particles.size
    tot_rows=10
    ttt = np.zeros((0))


    test = "/tmp/res/*"
    r = glob.glob(test)
    for i in r:
        os.remove(i)

    step=100
    pid=0
    multiprocess=80

    #exit()

    for ii in range(3):

        bi=ii*multiprocess
        ei=bi+multiprocess
    
        jobs = []

        for i in range(bi,ei):
            b=i*step
            e=b+step

            #print("b-e")
            #print(b)
            #print(e)
            #Process(target=print_func, args=(name,))
    
            p = multiprocessing.Process(target=create_tracks, args=(b,e,pid))
            pid=pid+1
            jobs.append(p)
            p.start()
    
        for proc in jobs:
            proc.join()


        del jobs[:]


    path = '/tmp/res/'

    files = []
    # r=root, d=directories, f = files
    for r, d, f in os.walk(path):
        for file in f:
            #if '.txt' in file:
            files.append(os.path.join(r, file))

    #for f in files:
    #    print(f)

    with open('/data/merged_1', 'w') as outfile:
        for fname in files:
            with open(fname) as infile:
                outfile.write(infile.read())

    c=np.loadtxt(open("/data/merged_1"))
    #print(c.shape)
    #print(c.shape[0])

    perc=40

    #tot_rows = c.shape[0]

    AmountOfFakeRows=int(c.shape[0]*perc/100)
    
    print(AmountOfFakeRows)

    X=c[:AmountOfFakeRows,]

    currentrow=0

    for cell in X: # for each row in fake dataset
        indd=amount_of_columns(cell)
        #print(indd)
        if (indd > 2) :
            cols_num = random.randrange(2, indd) #cols_num is the  amount of columns that will be switched
            cols = select_random_columns(cols_num)
    
            randomrow=random.randrange(1, AmountOfFakeRows)
            #print("randomrow-currentrow")
            #print(randomrow)
            #print(currentrow)
            switch_random_columns( randomrow , currentrow, cols )
            currentrow=currentrow+1

            X[:, tot_columns-1:]=int(1)

    #print("X.shape")
    #print(X.shape)
    #print("X ")
    #print(X[:, tot_columns-1:])

    #print("ds 11")
    #concatenate Real and Fake tracks
    ds=np.concatenate((c, X), axis=0)
    np.random.shuffle(ds)
    #print("ds 1")
    #print(ds.shape)
    #print(ds[:, tot_columns-1:])

    dtpd = pd.DataFrame(data=ds[0:,0:])#,    # values
    #print("1 ")
    dtpd.to_csv("/data/TrackFakeReal.csv")
    #print("2 ")
    dtpd.head(2)
    #print(dtpd.loc[dtpd[121] == 0])
    #dtpd.iloc[:,121]

event_prefix = 'event000002878'
hits, cells, particles, truth = load_event(os.path.join('/data/trackMLDB/train_2/train_2', event_prefix))
tot_columns=121
X = np.zeros((0))

main()

print("end execution")

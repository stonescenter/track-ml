import sys
import numpy as np
import pandas as pd
from sklearn.utils import shuffle
import os, fnmatch
import random

sys.path.append('/home/silvio/github/track-ml-1/utils/')
from tracktop import *
from tracks_manipulation_lib import *

def create_fake(err_index):

    global tracks_real
    print (err_index)

    tracks_fake_sort = tracks_real

    for i in range(tracks_fake_sort.shape[0]):
        convert_track_etaphi_err(tracks_fake_sort.iloc[i,:], err = err_index)

    #Sort Fake Tracks
    tracks_fake_sort2 = tracks_fake_sort.replace(0.0, np.PINF).copy()

    for i in range(tracks_fake_sort2.shape[0]):
        xyz_bsort(tracks_fake_sort2.iloc[i,:])

    tracks_fake_sort3 = tracks_fake_sort2.replace(np.PINF,0.0).copy()

    #Include new column result =1 -> indicates fake track
    tracks_fake_sort3['result'] = 1
    #Include new column result =0 -> indicates real track
    tracks_real['result'] = 0

    #join fake and real
    result = tracks_real.append(tracks_fake_sort3, ignore_index=True)

    #shuffle dataset
    result2 = shuffle(result)

    #cut cell and value
    result2aux = result2.iloc[:, np.r_[ 0 : 3 , 6 : 9 , 12 : 15 , 18 : 21 , 24 : 27 , 30 : 33 , 36 : 39 , 42 : 45 , 48 : 51 , 54 : 57 , 60 : 63 , 66 : 69 , 72 : 75 , 78 : 81 , 84 : 87 , 90 : 93 , 96 : 99 , 102 : 105 , 108 : 111 , 114 : 117 , 120 : 123 , 126 : 129 , 132 : 135 , 138 : 141 , 144 : 147 , 150 : 153 , 156 : 159 , 162 : 165]]

    tracks_fake_sort3.to_csv("/data/DSfake"+str(err_index))
    result2aux.to_csv("/data/DStotalV2"+str(err_index))
    result2.to_csv("/data/DStotal"+str(err_index))

errlist=[0.005, 0.01, 0.02, 0.03, 0.1, 0.15]

for err_index in errlist:
    #tracks      = pd.read_csv('/data/trackMLDB/analysis/train_2_realv4aux')
    tracks      = pd.read_csv(sys.argv[1])

    #cut Particle Information
    tracks_real = tracks.iloc[:,5:-1]

    create_fake(err_index)















#tracks      = pd.read_csv('/data/trackMLDB/analysis/train_2_realv4aux')

#cut Particle Information
#tracks_real = tracks.iloc[:,5:-1]

#convert_track_etaphi_err(tracks_fake_sort.iloc[i,:], err = 0.005)
#convert_track_etaphi_err(tracks_fake_sort.iloc[i,:], err = 0.01)
#convert_track_etaphi_err(tracks_fake_sort.iloc[i,:], err = 0.02)
#convert_track_etaphi_err(tracks_fake_sort.iloc[i,:], err = 0.03)
#convert_track_etaphi_err(tracks_fake_sort.iloc[i,:], err = 0.1)
#convert_track_etaphi_err(tracks_fake_sort.iloc[i,:], err = 0.15)
#errlist=[0.01, 0.005]#, 0.01, 0.02, 0.03, 0.1, 0.15]

    '''
    print (err_index)

    tracks_fake_sort = tracks_real

    for i in range(tracks_fake_sort.shape[0]):
        convert_track_etaphi_err(tracks_fake_sort.iloc[i,:], err = err_index)

    #Sort Fake Tracks
    tracks_fake_sort2 = tracks_fake_sort.replace(0.0, np.PINF).copy()

    for i in range(tracks_fake_sort2.shape[0]):
        xyz_bsort(tracks_fake_sort2.iloc[i,:])

    tracks_fake_sort3 = tracks_fake_sort2.replace(np.PINF,0.0).copy()

    #Include new column result =1 -> indicates fake track
    tracks_fake_sort3['result'] = 1
    #Include new column result =0 -> indicates real track
    tracks_real['result'] = 0

    #join fake and real
    result = tracks_real.append(tracks_fake_sort3, ignore_index=True)

    #shuffle dataset
    result2 = shuffle(result)

    #cut cell and value
    result2aux = result2.iloc[:, np.r_[ 0 : 3 , 6 : 9 , 12 : 15 , 18 : 21 , 24 : 27 , 30 : 33 , 36 : 39 , 42 : 45 , 48 : 51 , 54 : 57 , 60 : 63 , 66 : 69 , 72 : 75 , 78 : 81 , 84 : 87 , 90 : 93 , 96 : 99 , 102 : 105 , 108 : 111 , 114 : 117 , 120 : 123 , 126 : 129 , 132 : 135 , 138 : 141 , 144 : 147 , 150 : 153 , 156 : 159 , 162 : 165]]

    tracks_fake_sort3.to_csv("/data/DSfake"+str(err_index))
    result2aux.to_csv("/data/DStotalV2"+str(err_index))
    result2.to_csv("/data/DStotal"+str(err_index))
    '''

#input_dir_cfg="/data/trackMLDB/train/"
#input_analysis_cfg="/data/trackMLDB/analysis/"
#read Real Tracks
#tracks      = pd.read_csv('/data/totalrealmini2')
#tracks      = pd.read_csv('/data/output/TracksRealevent000002994.csv')
#Sort Particles
#tracks_real_sort = tracks_real.replace(0.0, np.PINF).copy()
#for i in range(tracks_real_sort.shape[0]):
#    xyz_bsort(tracks_real_sort.iloc[i,:])
#tracks_real_sort3 = tracks_real_sort.replace(np.PINF, 0.0).copy()
#tracks_real_sort3=tracks_real_sort
#tracks_fake_sort = tracks_real.iloc[:,:].apply(err_normal)

# In[10]:
#result2aux = result2.iloc[:, np.r_[ 0 : 3 , 6 : 9 , 12 : 15 , 18 : 21 , 24 : 27 , 30 : 33 , 36 : 39 , 42 : 45 , 48 : 51 , 54 : 57 , 60 : 63 , 66 : 69 , 72 : 75 , 78 : 81 , 84 : 87 , 90 : 93 , 96 : 99 , 102 : 105 , 108 : 111 , 114 : 117 , 120 : 123 , 126 : 129 , 132 : 135 , 138 : 141 , 144 : 147 , 150 : 153 , 156 : 159 , 162 : 165]]
#result2aux.to_csv("/data/ds")
# In[ ]:
#print(print(result2aux)
#b=0
#e=3
#pivot=6
#for i in range(28):
#    print(b,':',e,",", end =" ")
#    b=b+pivot
#    e=e+pivot

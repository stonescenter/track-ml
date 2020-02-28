import sys
import numpy as np
import pandas as pd

from sklearn.utils import shuffle
import os, fnmatch
import random

sys.path.append('.')
from tracktop import *
from tracks_manipulation_lib import *

input_dir_cfg="/data/trackMLDB/train/"
input_analysis_cfg="/data/trackMLDB/analysis/"

train_dir = ["train_1/","train_2","train_3","train_4","train_5"]

dir_ev = sys.argv[1]

listOfFiles = os.listdir(input_dir_cfg+train_dir[int(dir_ev)])

print (train_dir[int(dir_ev)])

if os.path.exists(input_analysis_cfg+train_dir[int(dir_ev)]):
    print (input_analysis_cfg+train_dir[int(dir_ev)])
else:
    os.mkdir(input_analysis_cfg+train_dir[int(dir_ev)])

pattern = "*-particles.csv"
for entry in listOfFiles:
    if fnmatch.fnmatch(entry, pattern):
        print (entry[0:14])
        print (input_dir_cfg+train_dir[int(dir_ev)])
        print (input_analysis_cfg+train_dir[int(dir_ev)])
        createTracks(event_prefix=entry[0:14],dir_event_prefix=input_dir_cfg+train_dir[int(dir_ev)],diroutput=input_analysis_cfg+train_dir[int(dir_ev)])
#event_file=random.choice(listOfFiles)

'''
createTracks(event_prefix=event_file[0:14],dir_event_prefix=dir_ev)



#read Real Tracks
#tracks      = pd.read_csv('/data/totalrealmini2')
tracks      = pd.read_csv('/data/output/TracksRealevent000002994.csv')

#cut Particle Information
tracks_real = tracks.iloc[:,5:-1]

#Sort Particles
tracks_real_sort = tracks_real.replace(0.0, np.PINF).copy()

for i in range(tracks_real_sort.shape[0]):
    xyz_bsort(tracks_real_sort.iloc[i,:])

tracks_real_sort3 = tracks_real_sort.replace(np.PINF, 0.0).copy()

#Include new column result =0 -> indicates real track
tracks_real_sort3['result'] = 0

'''




# In[9]:


#tracks_real_sort2=tracks_real_sort3

#Generate Fake Tracks
#tracks_fake_sort = tracks_real_sort3.iloc[:,:].apply(err_normal)

#Sort Fake Tracks
#tracks_fake_sort2 = tracks_fake_sort.replace(0.0, np.PINF).copy()

#for i in range(tracks_fake_sort2.shape[0]):
#    xyz_bsort(tracks_fake_sort2.iloc[i,:])

#tracks_fake_sort3 = tracks_fake_sort2.replace(np.PINF,0.0).copy()

#Include new column result =1 -> indicates fake track
#tracks_fake_sort3['result'] = 1

#joinfake and real
#result = tracks_real_sort2.append(tracks_fake_sort3, ignore_index=True)

#shuffle dataset
#result2 = shuffle(result)

#result2.to_csv("/data/ds2")


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

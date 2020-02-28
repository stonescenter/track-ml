#!/usr/bin/env python
# coding: utf-8

# In[35]:


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


# In[36]:


from scipy import special

def err_gauss(position, err, len_err = 50):
    err_array = np.linspace(1 - err, 1 + err, len_err)
    value_err = position * random.choice(err_array)
    return value_err


# In[37]:


#rand_cols_set = [0,1,2,6,7,8,12,13,14,18,19,20,24,25,26,30,31,32,36,37,38,42,43,44,48,49,50,54,55,56,60,61,62,66,67,68,72,73,74,78,79,80,84,85,86,90,91,92,96,97,98,102,103,104,108,109,110]
#print(len(rand_cols_set))
#rand_cols_set[10]


# In[38]:


percentage_of_fake_rows=90
temporary_directory = "/tmp/res/"

c=np.loadtxt(open(temporary_directory+"/merged_1"))
AmountOfFakeRows=int(c.shape[0]*percentage_of_fake_rows/100)
print("Amount Of Fake Rows that will be created: ", AmountOfFakeRows)


# In[39]:


#track without particle information only hits
aux=c[:AmountOfFakeRows,]
X=aux[:AmountOfFakeRows,6:120]
aux2=aux[:AmountOfFakeRows,0:6]

#rand_cols_set=[0,1,2]
rand_cols_set = [0,1,2,6,7,8,12,13,14,18,19,20,24,25,26,30,31,32,36,37,38,42,43,44,48,49,50,54,55,56,60,61,62,66,67,68,72,73,74,78,79,80,84,85,86,90,91,92,96,97,98,102,103,104,108,109,110]
amount_of_cols_than_can_be_changed=len(rand_cols_set)
currentrow=0
for cell in X: # for each row in fake dataset
    
    changed = 0
    while (changed < 1):
        rand_col = random.randrange(0, amount_of_cols_than_can_be_changed)
        changed = X[int(currentrow)][rand_col]
   
    X[int(currentrow)][rand_col] = err_gauss(X[int(currentrow)][rand_col], 0.03)
    X[int(currentrow)][rand_col+1] = err_gauss(X[int(currentrow)][rand_col+1], 0.03)
    X[int(currentrow)][rand_col+2] = err_gauss(X[int(currentrow)][rand_col+2], 0.03)
   
xtotnp=np.concatenate((aux2, X), axis=1)

vfinalReal = np.hstack((c, np.ones((c.shape[0], 1), dtype=c.dtype)))
vfinalFake = np.hstack((xtotnp, np.zeros((xtotnp.shape[0], 1), dtype=xtotnp.dtype)))

ds=np.concatenate((vfinalReal, vfinalFake), axis=0)
np.random.shuffle(ds)
dtpd = pd.DataFrame(data=ds[0:,0:])
output_file_path = "/data/TrackFakeRealGauss.csv"
dtpd.to_csv(output_file_path)


# In[ ]:





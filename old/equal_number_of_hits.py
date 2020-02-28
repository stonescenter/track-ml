import matplotlib.pyplot as plt

import tensorflow as tf

from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

from random import seed
from numpy import array
from random import randint

from math import sqrt

import numpy as np
import pandas as pd

import sys
import os
import ntpath
import datetime

event_prefix = sys.argv[1]
output_prefix = sys.argv[2]
aux_am_per_hit = sys.argv[3]

flag_am_per_hit = int(aux_am_per_hit)

dfOriginal      = pd.read_csv(event_prefix)
totdf = pd.read_csv(event_prefix)

dfOriginal['totalZeros'] = (dfOriginal == 0).sum(axis=1)
dfOriginal['totalHits'] = dfOriginal['totalZeros']/6
#dfOriginal['totalHits2'] = dfOriginal['totalZeros']%6

org = dfOriginal.sort_values(by=['totalHits'])
min=dfOriginal['totalHits'].min()
max=dfOriginal['totalHits'].max()
#min2=dfOriginal['totalHits2'].min()
#max2=dfOriginal['totalHits2'].max()

#print(min,max)
#print(min2,max2)

min=13
maxt=int(max)
mint=int(min)


totaux=maxt-mint
print(totaux)
tot=totaux+1
print(tot)

minV=np.zeros((tot))

print(min,max,tot)

ind=0
for i in range(int(min), (int(max)+1)):
    auxDF = dfOriginal.loc[dfOriginal['totalHits'] == i]
    minV[ind]=auxDF.shape[0]
    ind=ind+1

minAux=int(minV.min())

if (minAux > flag_am_per_hit):
    minAux = flag_am_per_hit

print(minAux)
print(minV)

totdf = totdf.iloc[0:0]
for i in range(int(min), (int(max)+1)):
    auxDF = dfOriginal.loc[dfOriginal['totalHits'] == i]
    auxDF2 = auxDF.iloc[0:minAux,:]
    print(auxDF2.shape)
    totdf = totdf.append(auxDF2)

totdf.to_csv(output_prefix)

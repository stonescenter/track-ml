from keras.models import load_model
from random import seed
from random import randint
import numpy as np
import pandas as pd
import sys

import tensorflow as tf
from keras.backend.tensorflow_backend import set_session

def position_3D_approximation(result, strategy):
    # result => predicted
    #yclone = np.copy(y)
    #dfyclone = pd.DataFrame.from_records(yclone)

    global dfyclone
    global yclone

    #print(result)
    #print(result.shape)
    #print(result.shape[0], 3)
    #result.reshape((result.shape[0], 3))
    #a = result.reshape(result.shape[0], 3)
    #print(a.shape)

    if (Neural_Network_Model == 3): #LSTM without channel
        result = result.reshape(result.shape[0], 3)
    #print(result.shape)
    #b = result.values.flatten()
    #n_patterns=len(result)
    #print(n_patterns)
    #result = np.reshape(b,(n_patterns,3))
    #print(result)


    df3d = pd.DataFrame({'X':result[:,0],'Y':result[:,1],'Z':result[:,2]})

    #print(df3d)
    df3d['X-pred'] = 0
    df3d['Y-pred'] = 0
    df3d['Z-pred'] = 0
    df3d['ch0'] = 0
    df3d['ch1'] = 0
    df3d['w'] = 0

    #dfyclone = pd.DataFrame.from_records(yclone)

    for index, row in df3d.iterrows():
        #obtain the row with least geometric distance between predicted row and original rows (in yclone)
        Xpred=df3d.loc[index, 'X']
        Ypred=df3d.loc[index, 'Y']
        Zpred=df3d.loc[index, 'Z']

        if (strategy==0):
            dfyclone['geodist'] = ( ((dfyclone[0] - Xpred) **2) + ((dfyclone[1] - Ypred) **2)   + ((dfyclone[2] - Zpred) **2) )
            row=dfyclone.loc[dfyclone['geodist'].idxmin()]
            lowerGeodistIndex=row.name
        else:
            lowerGeodistIndex=randint(0,dfyclone.shape[0]-1)


        X=yclone[lowerGeodistIndex,0]
        Y=yclone[lowerGeodistIndex,1]
        Z=yclone[lowerGeodistIndex,2]
        ch0=yclone[lowerGeodistIndex,3]
        ch1=yclone[lowerGeodistIndex,4]

        df3d.loc[index, 'X-pred'] = X
        df3d.loc[index, 'Y-pred'] = Y
        df3d.loc[index, 'Z-pred'] = Z
        df3d.loc[index, 'ch0'] = ch0
        df3d.loc[index, 'ch1'] = ch1

        #remove the hit used as approximation
        dfyclone.drop(dfyclone.index[lowerGeodistIndex])

    #print(df3d)
    df3d.drop('X', axis=1, inplace=True)
    df3d.drop('Y', axis=1, inplace=True)
    df3d.drop('Z', axis=1, inplace=True)
    #print(df3d)

    #return the fourth hit of all tracks
    return(df3d)

seed(1)

# configure  gpu_options.allow_growth = True in order to CuDNNLSTM layer work on RTX
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
set_session(sess)

#Read input file with first 3 hits to dataframe
tracks_to_be_reconstructed_file = sys.argv[1]
all_hits_available_file = sys.argv[2]
trained_model_file = sys.argv[3]
reconstructed_tracks_file = sys.argv[4]

#strategyFlag=0 -> LSTM
#strategyFlag=1 -> Random
strategyFlag= int(sys.argv[5])
Neural_Network_Model= int(sys.argv[6])

df = pd.read_csv(tracks_to_be_reconstructed_file)

#create dataset ofa ll hists in dfyclone
df_all_hits = pd.read_csv(all_hits_available_file)
y=df_all_hits.iloc[:, [ 25,26,27,28,29 ]]
yclone = np.copy(y)
dfyclone = pd.DataFrame.from_records(yclone)

#all tracks have at maximum 29 hits
hits = np.arange(0,29,1)

firstHit=1
lastHit=4

df1 = df.iloc[:,1:25]

for firstHit in range(1, 26):
#for firstHit in range(1, 2):

    lastHit = lastHit + 1
    #begin with 3 know hits
    known_hits= np.arange(firstHit,lastHit,1)
    #next hit to predict
    hit_to_predict=known_hits[3]

    dataX2=df1.iloc[:, [ (hits[known_hits[0]]*6)+1,(hits[known_hits[0]]*6)+2,(hits[known_hits[0]]*6)+3 , (hits[known_hits[1]]*6)+1,(hits[known_hits[1]]*6)+2,(hits[known_hits[1]]*6)+3 , (hits[known_hits[2]]*6)+1,(hits[known_hits[2]]*6)+2,(hits[known_hits[2]]*6)+3 ]]
    dataXfeatures=df1.iloc[:, [  (hits[known_hits[0]]*6)+4,(hits[known_hits[0]]*6)+5 , (hits[known_hits[1]]*6)+4 , (hits[known_hits[1]]*6)+5  , (hits[known_hits[2]]*6)+4 , (hits[known_hits[2]]*6)+5  ]]#dataXfeatures=df.iloc[:, [  hits[known_hits[0]]+4,hits[known_hits[0]]+5,hits[known_hits[0]]+6 , hits[known_hits[1]]+4,hits[known_hits[1]]+5,hits[known_hits[1]]+6 , hits[known_hits[2]]+4,hits[known_hits[2]]+5,hits[known_hits[2]]+6  ]]

    #prepare data to inference
    b = dataX2.values.flatten()
    bfeat=dataXfeatures.values.flatten()
    n_patterns=len(df)
    X     = np.reshape(b,(n_patterns,3,3))
    Xfeat = np.reshape(bfeat,(n_patterns,3,2))

    #perform the prediction
    model = load_model(trained_model_file)

    if (Neural_Network_Model == 1): #LSTM with channel as features
        result = model.predict([X, Xfeat],verbose=1)
    if (Neural_Network_Model == 2): #MLP
        result = model.predict(dataX2,verbose=1)
    if (Neural_Network_Model == 3): #LSTM without channel
        result = model.predict(X,verbose=1)

    pred = position_3D_approximation(result, strategyFlag)

    #concat tracks with predicted positions
    #print("df1")
    #print(df1)
    df1= pd.concat([df1,pred],axis=1) #, keys=['One','Two'])
    #print("df1-concat")
    #print(df1)

print(reconstructed_tracks_file)
df1.to_csv(reconstructed_tracks_file)












'''
print(known_hits)
print(hit_to_predict)
print("print(df1.shape) : ",df1.shape)
print ("X")
print ( (hits[known_hits[0]]*6)+1,(hits[known_hits[0]]*6)+2,(hits[known_hits[0]]*6)+3 , (hits[known_hits[1]]*6)+1,(hits[known_hits[1]]*6)+2,(hits[known_hits[1]]*6)+3 , (hits[known_hits[2]]*6)+1,(hits[known_hits[2]]*6)+2,(hits[known_hits[2]]*6)+3)
print ("Xfeat")
print ( (hits[known_hits[0]]*6)+4,(hits[known_hits[0]]*6)+5 , (hits[known_hits[1]]*6)+4 , (hits[known_hits[1]]*6)+5  , (hits[known_hits[2]]*6)+4 , (hits[known_hits[2]]*6)+5)
print ("Y")
print ( (hits[hit_to_predict]*6)+1,(hits[hit_to_predict]*6)+2,(hits[hit_to_predict]*6)+3,(hits[hit_to_predict]*6)+4,(hits[hit_to_predict]*6)+5 )
'''

#cpton=((hits[known_hits[2]]*6)+5)+1
#print("cpton ",cpton)
#print((hits[known_hits[0]]*6)+1,(hits[known_hits[0]]*6)+2,(hits[known_hits[0]]*6)+3 , (hits[known_hits[1]]*6)+1,(hits[known_hits[1]]*6)+2,(hits[known_hits[1]]*6)+3 , (hits[known_hits[2]]*6)+1,(hits[known_hits[2]]*6)+2,(hits[known_hits[2]]*6)+3)
#print((hits[known_hits[0]]*6)+4,(hits[known_hits[0]]*6)+5 , (hits[known_hits[1]]*6)+4 , (hits[known_hits[1]]*6)+5  , (hits[known_hits[2]]*6)+4 , (hits[known_hits[2]]*6)+5)

'''
    # evaluation
    dfglobal_original = np.hstack((dataX2,yaux))
    dfglobal_predicted = np.hstack((dataX2,pred))

    dftot = pd.DataFrame.from_records(dfglobal_predicted)
    dftot2 = pd.DataFrame.from_records(dfglobal_original)

    #print(dftot)
    #print(dftot2)

    dftotfinal = pd.DataFrame( abs(((dftot[9] - dftot2[9]) **2) + ((dftot[10] - dftot2[10]) **2)   + ((dftot[11] - dftot2[11]) **2)) )

    seriesObj = dftotfinal.apply(lambda x: True if x[0] == 0 else False , axis=1)
    numOfRows = len(seriesObj[seriesObj == True].index)
    print('equal 0 : ', numOfRows)

    maxV=dftotfinal[0].max()
    minV=dftotfinal[0].min()

    if ((maxV == 0) and (minV == 0)):
        print('max : ' , maxV )
        print('min : ' , minV )
    else:
        print('mean geodist from original values : ' , dftotfinal[0].mean())
        print('max : ' , maxV )
        print('min : ' , minV )
        #print('equal 0: ', dftotfinal[dftotfinal[0] == 0].sum()  )

    plot_results_inferences(dftotfinal)
'''

'''
    print('===========================================')
    print('X')
    print(X)
    print('===========================================')
    print('Xfeat')
    print(Xfeat)
    print('===========================================')
    print('y')
    print(y)
    print('===========================================')
'''

'''
    print('===========================================')
    print('dfglobal_original')
    print(dfglobal_original)
    print('===========================================')
    print('dfglobal_predicted')
    print(dfglobal_predicted)
    print('===========================================')
'''

    #dftotfinal['Xdiff'] = pd.DataFrame(abs(dftot[9]) - abs(dftot2[9])) + (abs(dftot[10]) - abs(dftot2[10])) + (abs(dftot[11]) - abs(dftot2[11]))
    #dftotfinal = pd.DataFrame( (abs((dftot[9] - dftot2[9]))) + (abs((dftot[10] - dftot2[10]))) + (abs(dftot[11]) - abs(dftot2[11])) )
    #df22=dftotfinal.sort_values(by='Xdiff')
    #print(dftotfinal)
    #print('sum : ' , dftotfinal[0].sum() )
    #print('max : ' , dftotfinal[0].max() )
    #print('min : ' , dftotfinal[0].min() )
    #print('shape : ' , dftotfinal.shape[0] )
    #ss=dftotfinal[0].sum()
    #sh=dftotfinal.shape[0]
    #av=ss/sh
    #print('av : ' , av )
#all_hits_available_file=ntpath.basename(event_prefix)
#from keras.models import Sequential
#from keras.layers import Dense
#from keras.layers import LSTM
#from keras.layers import concatenate,Input,Flatten
#from keras.models import Model

#from sklearn.metrics import mean_squared_error
#from sklearn.model_selection import train_test_split


#from math import sqrt


#import os
#import ntpath

#import scipy.stats

#import seaborn as sns


#from pandas.plotting import register_matplotlib_converters

#import datetime

#import matplotlib.pyplot as plt
#register_matplotlib_converters()

#def plot_results_inferences(df22):
#    sns.distplot(df22[0], color="blue") #, ax=axes[0,0], axlabel='X-original')
    #plt.show()

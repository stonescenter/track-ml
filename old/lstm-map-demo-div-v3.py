import matplotlib.pyplot as plt

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
from keras.layers import CuDNNLSTM
from keras.layers import concatenate,Input,Flatten
from keras.models import Model
from keras.utils import plot_model


import tensorflow as tf
from keras.backend.tensorflow_backend import set_session

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

'''
def prepare_data(event_prefix):

    event_file_name=ntpath.basename(event_prefix)

    df = pd.read_csv(event_prefix)

    BG=1
    #EN=25
    EN=3
    dftot = df.iloc[:, BG:EN]
    print("df!!")
    print(df)
    print("dftot!!")
    print(dftot)
    #print("dftot!!!!!!")
    #print(dftot.iloc[:, 7])
    #print(dftot.iloc[:, 13])
    #print(dftot.iloc[:, 19])

    #for column in dftot:
    #    print(dftot[column])
    #for ind, column in enumerate(dftot.columns):
    #    print(ind, column)

    for ida in range(2):
        #BG=BG+13
        #EN=EN+13
        BG=BG+3
        EN=EN+3

        dfaux=df.iloc[:, BG:EN]
        #print("dfaux1")
        #print(dfaux)
        #print(dfaux.shape)
        for ind, column in enumerate(dfaux.columns):
        #    print(ind, column)
            #dfaux.rename(columns={column: ind})
            dfaux.rename(columns={ dfaux.columns[ind]: ind }, inplace = True)
        #for ind, column in enumerate(dfaux.columns):
        #    print(ind, column)
        #dftot.append(dfaux) #, ignore_index = True)
        #dftot.append(dfaux, ignore_index = True)
        #dftot.join(dfaux, how='below')
        #print("dfaux1")
        #print(dfaux)
        #print(dfaux.shape)

        #dftot2=dftot.append(dfaux)

        #dftot2=pd.concat([dftot, dfaux], axis = 0, ignore_index=True)
        print("dftot2")
        print(dftot2)
        print(dftot2.shape)
        #print(dftot2.iloc[:, 7])
        #print(dftot2.iloc[:, 13])
        #print(dftot2.iloc[:, 19])
    #dftot=dftot+dfaux

'''

def prepare_training_data_LSTM_SEQ(event_prefix):

    event_file_name=ntpath.basename(event_prefix)

    df = pd.read_csv(event_prefix)

    dataX2=df.iloc[:, [7,8,9,13,14,15,19,20,21]]
    dataXfeatures=df.iloc[:, [10,11,16,17,22,23]]
    b = dataX2.values.flatten()
    bfeat=dataXfeatures.values.flatten()
    n_patterns=len(df)
    X     = np.reshape(b,(n_patterns,3,3))
    Xfeat = np.reshape(bfeat,(n_patterns,3,2))

    YY=df.iloc[:, [25,26,27]]
    yb = YY.values.flatten()
    y     = np.reshape(yb,(n_patterns,3,1))

    #y=df.iloc[:, [25,26,27]]
    return(X, Xfeat, y)

# prepare training data
def prepare_training_data(event_prefix):

    event_file_name=ntpath.basename(event_prefix)

    df = pd.read_csv(event_prefix)

    dataX2=df.iloc[:, [7,8,9,13,14,15,19,20,21]]
    dataXfeatures=df.iloc[:, [10,11,16,17,22,23]]
    b = dataX2.values.flatten()
    bfeat=dataXfeatures.values.flatten()
    n_patterns=len(df)
    X     = np.reshape(b,(n_patterns,3,3))
    Xfeat = np.reshape(bfeat,(n_patterns,3,2))
    y=df.iloc[:, [25,26,27]]
    return(X, Xfeat, y)


def prepare_training_data_MLP(event_prefix):

    event_file_name=ntpath.basename(event_prefix)

    df = pd.read_csv(event_prefix)

    #dataX2=df.iloc[:, [7,8,9,13,14,15,19,20,21]]
    #dataXfeatures=df.iloc[:, [10,11,16,17,22,23]]
    #b = dataX2.values.flatten()
    #bfeat=dataXfeatures.values.flatten()
    #n_patterns=len(df)
    #X     = np.reshape(b,(n_patterns,3,3))

    X=df.iloc[:, [7,8,9,13,14,15,19,20,21]]
    #X=df.iloc[:, [7,8,9,10,11,13,14,15,16,17,19,20,21,22,23]]

    #Xfeat = np.reshape(bfeat,(n_patterns,3,2))
    Xfeat=df.iloc[:, [10,11,16,17,22,23]]
    y=df.iloc[:, [25,26,27]]
    return(X, Xfeat, y)


def create_Neural_Network_LSTM_wt_chs(neurons,init_mode,activation,optf,lossf):
    #create Neural Network

    activation = 'hard_sigmoid'
    init_mode  = 'he_normal'
    #opt        = 'Adadelta'

    data_dim = 3
    timesteps = 3
    num_classes = 10

    model = Sequential()
    model.add(CuDNNLSTM(neurons, return_state=False, return_sequences=True, input_shape=(timesteps, data_dim)))  # returns a sequence of vectors of dimension 32
#    model.add(Dense(9, activation=activation, kernel_initializer=init_mode, input_dim=9))
#   model.add(CuDNNLSTM(neurons,return_sequences=False,return_state=False))
#    model.add(LSTM(neurons,return_sequences=False,return_state=False))
    model.add(Dense(neurons, activation = 'relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation = 'relu')) #, activation=activation, kernel_initializer=init_mode))
    model.compile(optimizer = optf,loss=lossf, metrics =['accuracy'])
    return(model)

def create_Neural_Network(neurons,init_mode,activation,optf,lossf):
    #create Neural Network
    xshape=Input(shape=(3,3))
    yshape=Input(shape=(3,2))

    admi=CuDNNLSTM(neurons,return_sequences=False,return_state=False)(xshape)
    pla=CuDNNLSTM(neurons,return_sequences=False,return_state=False)(yshape)

    #admi=LSTM(neurons,return_sequences=False)(xshape)
    #pla=LSTM(neurons,return_sequences=False)(yshape)
    out=concatenate([admi,pla],axis=-1)

    #output=Dense(3, activation=activation, kernel_initializer=init_mode)(out)
    output2=Dense(neurons, activation = 'relu')(out)
    output3=Dense(neurons, activation = 'relu')(output2)
    output4=Dropout(0.5)(output3)
    #output=Dense(3)(output3)

    output=Dense(3, activation = 'relu')(output4)

    model = Model(inputs=[xshape, yshape], outputs=output)
    model.compile(optimizer = optf,loss=lossf, metrics =['accuracy'])
    plot_model(model, to_file='/home/silvio/modellstm.png',  show_shapes=True)
    return(model)

def create_Neural_Network_MLP(neurons,init_mode,activation,optf,lossf):
    #create Neural Network

    #activation = 'hard_sigmoid'
    #neurons    = 100
    #init_mode  = 'he_normal'
    #opt        = 'Adadelta'
    #epochs_num = 10

    model = Sequential()

    #model.add(Dense(neurons, activation=activation, kernel_initializer=init_mode, input_dim=9))
    #model.add(Dense(neurons, activation=activation, kernel_initializer=init_mode, input_dim=15))
    model.add(Dense(neurons, input_dim=9))
    model.add(Dense(neurons, activation = 'relu'))
    model.add(Dropout(0.5))
    model.add(Dense(neurons, activation = 'relu'))
    #model.add(Dense(neurons, activation=activation, kernel_initializer=init_mode))
    #model.add(Dense(3, activation=activation, kernel_initializer=init_mode))
    model.add(Dense(3, activation = 'relu'))
    model.compile(optimizer = optf,loss=lossf, metrics =['accuracy'])
    return(model)



    #xshape=Input(shape=(3,3))
    #yshape=Input(shape=(3,2))

    #admi=CuDNNLSTM(neurons,return_sequences=False,return_state=False)(xshape)
    #pla=CuDNNLSTM(neurons,return_sequences=False,return_state=False)(yshape)

    #admi=LSTM(neurons,return_sequences=False)(xshape)
    #pla=LSTM(neurons,return_sequences=False)(yshape)
    #out=concatenate([admi,pla],axis=-1)

    #output=Dense(3, activation=activation, kernel_initializer=init_mode)(out)
    #output=Dense(3)(out)
    #model = Model(inputs=[xshape, yshape], outputs=output)
    #model.compile(optimizer = opt,loss=lossf, metrics =['accuracy'])
    #model.compile(loss='mean_squared_error', optimizer='adam',metrics =['accuracy'])
    #return(model)

def evaluate_model(history,model,NNmodel):

    #Evaluate Model
    if (NNmodel==1):
        eval_model=model.evaluate([X_test, Xfeat_test], y_test)
    else:
        eval_model=model.evaluate(X_test, y_test)

    print('loss: {:4f}'.format(eval_model[0]))
    print('accuracy: {:4f}'.format(eval_model[1]))


    plt.plot(history.history['loss'])
    #plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper right')
    plt.savefig(lossfile)
    plt.show()

# configure  gpu_options.allow_growth = True in order to CuDNNLSTM layer work on RTX
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
set_session(sess)

seed(1)
event_prefix = sys.argv[1]
modelfile = sys.argv[2] #"model.h5"
lossfile = sys.argv[3]
NNmodelSTR = sys.argv[4]
NNmodel = int(NNmodelSTR)
#prepare_data(event_prefix)
#sys.exit(0)

if (NNmodel==1):
    X, Xfeat, y = prepare_training_data(event_prefix)
if (NNmodel==2):
    X, Xfeat, y = prepare_training_data_MLP(event_prefix)
if (NNmodel==3):
    X, Xfeat, y = prepare_training_data_LSTM_SEQ(event_prefix)

#X_train, X_test, Xfeat_train, Xfeat_test, y_train, y_test = train_test_split(X, Xfeat, y, test_size=0.01)
X_train, X_test, Xfeat_train, Xfeat_test, y_train, y_test = train_test_split(X, Xfeat, y, test_size=0.3)
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

#model=create_Neural_Network(300,'normal','linear','SGD', 'cosine_proximity')
if (NNmodel==1):
    model=create_Neural_Network(200,'normal','linear','Adadelta', 'mean_absolute_error')

if (NNmodel==2):
    #model=create_Neural_Network_MLP(200,'normal','linear','RMSprop', 'logcosh')
    #model=create_Neural_Network_MLP(200,'normal','linear','Adamax', 'mean_squared_error')
    #   mean_squared_logarithmic_error
    #model=create_Neural_Network_MLP(200,'normal','linear','RMSprop', 'cosine_proximity')
    #model=create_Neural_Network_MLP(200,'normal','linear','Adagrad', 'mean_squared_error')
    #model=create_Neural_Network_MLP(200,'normal','linear','RMSprop', 'mean_squared_logarithmic_error')
    model=create_Neural_Network_MLP(200,'normal','linear','Adadelta', 'mean_absolute_error')
    #model=create_Neural_Network_MLP(200,'normal','linear','RMSprop', 'mean_absolute_percentage_error')
    #model=create_Neural_Network_MLP(200,'normal','linear','RMSprop', 'cosine_proximity')


# good combinations:
# Adadelta + 'mean_squared_error
# RMSprop + 'mean_squared_error
#Adagrad   + 'mean_squared_error

#'Adadelta' ok
#'SGD' nok
#'Adamax' nok
#'RMSprop' ok
#'Adagrad'
#'Adam'
#'Nadam'




if (NNmodel==3):
    #model=create_Neural_Network_LSTM_wt_chs(256,'normal','linear','RMSprop', 'mean_absolute_percentage_error')
    model=create_Neural_Network_LSTM_wt_chs(64,'normal','linear','RMSprop', 'mean_absolute_percentage_error')


#model.compile(loss='mean_squared_error', optimizer='adam',metrics =['accuracy'])
#model=create_Neural_Network(300,'normal','linear','SGD', 'mean_squared_error')

#run training
n_batch = 1
n_epoch = 20 #2000 #1000 #20 #300 #00

#print("22 ",datetime.datetime.now())
if (NNmodel==1):
    history = model.fit([X_train, Xfeat_train], y_train, epochs=n_epoch, verbose=2)
else:
    history = model.fit(X_train, y_train, epochs=n_epoch, verbose=2)

model.save(modelfile) #Save the model

#evaluate model
if (NNmodel==1):
    eval_model=model.evaluate([X_test, Xfeat_test], y_test, verbose=2)
else:
    eval_model=model.evaluate(X_test, y_test, verbose=2)

evaluate_model(history,model,NNmodel)

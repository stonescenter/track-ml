import matplotlib.pyplot as plt

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
from keras.layers import CuDNNLSTM
from keras.layers import concatenate,Input,Flatten
from keras.models import Model
from keras import metrics
from keras.utils import plot_model
from keras.callbacks import TensorBoard

import keras.backend as K

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

from time import sleep

# prepare training data
def prepare_training_data(event_prefix):

    event_file_name=ntpath.basename(event_prefix)

    df22 = pd.read_csv(event_prefix)
    df = df22.iloc[:,9:]
    bp=1
    ep=4
    bpC=7
    #epC=8
    epC=8

    interval=8

    #print(df.shape)
    #print(bp,ep,bp+interval,ep+interval,bp+(interval*2),ep+(interval*2),bp+(interval*3),ep+(interval*3))
    #print(bpC,epC,bpC+interval,epC+interval,bpC+(interval*2),epC+(interval*2),bpC+(interval*3),epC+(interval*3))

    #print(df.iloc[1:10,:])
    #print(df.iloc[1:10, np.r_[bp:ep,bp+interval:ep+interval,bp+(interval*2):ep+(interval*2),bp+(interval*3):ep+(interval*3) ]])

    dataX2=df.iloc[:, np.r_[bp:ep,bp+interval:ep+interval,bp+(interval*2):ep+(interval*2),bp+(interval*3):ep+(interval*3)]]
    dataXfeatures=df.iloc[:, np.r_[bpC:epC,bpC+interval:epC+interval,bpC+(interval*2):epC+(interval*2),bpC+(interval*3):epC+(interval*3)]]

    #evalX=(bp+(interval*3))
    #evalX3=evalX+3
    #print(bp+(interval*4),(bp+(interval*4)+3))

    y=df.iloc[:, np.r_[bp+(interval*4):(bp+(interval*4)+3)]]

    #print("y")
    #print(df.iloc[:, np.r_[bp+(interval*4):(bp+(interval*4)+3)]])
    #print("dataX2")
    #print(dataX2.iloc[1:10,:])
    #print("dataXfeatures")
    #print(dataXfeatures.iloc[1:10, :])

    b = dataX2.values.flatten()
    bfeat=dataXfeatures.values.flatten()
    n_patterns=len(df)
    X     = np.reshape(b,(n_patterns,4,3))
    #Xfeat = np.reshape(bfeat,(n_patterns,3,4))
    Xfeat = np.reshape(bfeat,(n_patterns,4,1))

    return(X, Xfeat, y)

def mean_pred(y_true, y_pred):
    return ( ((y_true - y_pred) **2) )

def B_accuracy(y_true, y_pred):
    '''Calculates the mean accuracy rate across all predictions for binary
    classification problems.

    return K.mean(K.equal(y_true, K.round(y_pred)))
    '''
    '''Calculates the mean accuracy rate across all predictions for
    multiclass classification problems.
    '''
    return K.mean(K.equal(K.argmax(y_true, axis=-1),
                  K.argmax(y_pred, axis=-1)))

def create_Neural_Network(neurons,init_mode,activation,optf,lossf):
    #create Neural Network
    xshape=Input(shape=(4,3))
    #yshape=Input(shape=(3,4))
    yshape=Input(shape=(4,1))

    admi=CuDNNLSTM(neurons,return_sequences=False,return_state=False)(xshape)
    pla=CuDNNLSTM(neurons,return_sequences=False,return_state=False)(yshape)

    #admi=LSTM(neurons,return_sequences=False)(xshape)
    #pla=LSTM(neurons,return_sequences=False)(yshape)
    out=concatenate([admi,pla],axis=-1)

    #output=Dense(3, activation=activation, kernel_initializer=init_mode)(out)

    output2=Dense(neurons, activation = 'relu')(out)
    output3=Dropout(0.5)(output2)
    output4=Dense(neurons, activation = 'relu')(output3)

    output=Dense(3)(output4)
    #output=Dense(3)(out)

    #output=Dense(3, activation = 'relu')(output4)
    #output=Dense(3, activation = 'relu')(out)

    model = Model(inputs=[xshape, yshape], outputs=output)

    model.compile(optimizer = optf,loss=lossf, metrics =['accuracy'])
    #model.compile(optimizer = optf,loss=lossf, metrics =['mse'])
    #model.compile(optimizer = optf,loss=lossf, metrics =['mse', 'mae', 'mape', 'cosine',mean_pred])
    #model.compile(optimizer = optf,loss=lossf, metrics =['accuracy' , B_accuracy])
    #model.compile(optimizer = optf,loss=lossf, metrics={'output_a': 'accuracy'})

    plot_model(model, to_file='/home/silvio/modellstm.png',  show_shapes=True)
    return(model)

def evaluate_model(history,model,NNmodel):

    #Evaluate Model
    if (NNmodel==1):
        eval_model=model.evaluate([X_test, Xfeat_test], y_test)
    else:
        eval_model=model.evaluate(X_test, y_test)

    print('loss: {:4f}'.format(eval_model[0]))
    print('accuracy: {:4f}'.format(eval_model[1]))

    plt.plot(history.history['loss'])

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

print(event_prefix)

X, Xfeat, y = prepare_training_data(event_prefix)

X_train, X_test, Xfeat_train, Xfeat_test, y_train, y_test = train_test_split(X, Xfeat, y, test_size=0.1)

model=create_Neural_Network(1000,'normal','linear','RMSprop', 'mean_absolute_error')

#run training
n_batch = 1
n_epoch = 10

#save tensorflow log
tensorboard = TensorBoard(log_dir="/home/silvio/logs/{}")
history = model.fit([X_train, Xfeat_train], y_train, epochs=n_epoch, validation_split=0.3, batch_size=1, shuffle=False, verbose=2, callbacks=[tensorboard])

#Save the model
model.save(modelfile)

#evaluate model
eval_model=model.evaluate([X_test, Xfeat_test], y_test, verbose=2)

evaluate_model(history,model,NNmodel)

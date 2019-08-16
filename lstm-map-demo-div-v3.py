import matplotlib.pyplot as plt

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import CuDNNLSTM
from keras.layers import concatenate,Input,Flatten
from keras.models import Model

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


def create_Neural_Network(neurons,init_mode,activation,opt,lossf):
    #create Neural Network
    xshape=Input(shape=(3,3))
    yshape=Input(shape=(3,2))

    admi=CuDNNLSTM(neurons,return_sequences=False,return_state=False)(xshape)
    pla=CuDNNLSTM(neurons,return_sequences=False,return_state=False)(yshape)

    #admi=LSTM(neurons,return_sequences=False)(xshape)
    #pla=LSTM(neurons,return_sequences=False)(yshape)
    out=concatenate([admi,pla],axis=-1)

    #output=Dense(3, activation=activation, kernel_initializer=init_mode)(out)
    output=Dense(3)(out)
    model = Model(inputs=[xshape, yshape], outputs=output)
    model.compile(optimizer = opt,loss=lossf, metrics =['accuracy'])
    #model.compile(loss='mean_squared_error', optimizer='adam',metrics =['accuracy'])
    return(model)

def evaluate_model(history,model):
    #Evaluate Model
    eval_model=model.evaluate([X_test, Xfeat_test], y_test)
    print('loss: {:4f}'.format(eval_model[0]))
    print('accuracy: {:4f}'.format(eval_model[1]))


    plt.plot(history.history['loss'])
    #plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper right')
    plt.savefig("/home/silvio/loss.png")
    #plt.show()

print("1 ", datetime.datetime.now())

seed(1)
event_prefix = sys.argv[1]
#prepare_data(event_prefix)
#sys.exit(0)

X, Xfeat, y = prepare_training_data(event_prefix)
X_train, X_test, Xfeat_train, Xfeat_test, y_train, y_test = train_test_split(X, Xfeat, y, test_size=0.01)

#model=create_Neural_Network(300,'normal','linear','SGD', 'cosine_proximity')
model=create_Neural_Network(200,'normal','linear','Adamax', 'mean_squared_error')
#model.compile(loss='mean_squared_error', optimizer='adam',metrics =['accuracy'])
#model=create_Neural_Network(300,'normal','linear','SGD', 'mean_squared_error')

#run training
n_batch = 1
n_epoch = 50

print("22 ",datetime.datetime.now())
history = model.fit([X_train, Xfeat_train], y_train, epochs=n_epoch, verbose=2)
#model.save("model.h5") #Save the model
#evaluate model

eval_model=model.evaluate([X_test, Xfeat_test], y_test, verbose=0)

'''
#neurons = [100,200,300,400,500,600,700,800]#,4,8]#,16,32,64,128] #[2,4,8,16] #,32,64,128,512,1024,2048]
neurons = [300]#,4,8]#,16,32,64,128] #[2,4,8,16] #,32,64,128,512,1024,2048]
#init = ['he_normal','uniform', 'lecun_uniform', 'normal', 'zero', 'glorot_normal', 'glorot_uniform', 'he_uniform']
init = ['normal']
#activation =  ['hard_sigmoid' , 'relu', 'tanh', 'sigmoid', 'linear', 'softmax','softplus', 'softsign' ]
activation =  ['linear']
opt = [ 'Adadelta','SGD', 'RMSprop', 'Adagrad', 'Adam', 'Adamax', 'Nadam']
losses = ['mean_squared_error','mean_absolute_error','mean_absolute_percentage_error',
          'mean_squared_logarithmic_error','squared_hinge','hinge','categorical_hinge',
          'logcosh','categorical_crossentropy','sparse_categorical_crossentropy',
          'binary_crossentropy','kullback_leibler_divergence','poisson','cosine_proximity']

#lst = [0,1,2,3,4,5,6,7]

#for ida in range(len(neurons)):
#    for idb in range(len(init)):
#        for idc in range(len(activation)):
ida=0
idb=0
idc=0
for idd in range(len(opt)):
    for ide in range(len(losses)):

        #print(idx)
        #model=create_Neural_Network(100,'uniform','linear', 'Adamax','mean_squared_error')
        model=create_Neural_Network(neurons[ida],init[idb],activation[idc], opt[idd],losses[ide])

        #run training
        n_batch = 1
        n_epoch = 20
        history = model.fit([X_train, Xfeat_train], y_train, epochs=n_epoch, verbose=0)
        #model.save("model.h5") #Save the model
        #evaluate model

        eval_model=model.evaluate([X_test, Xfeat_test], y_test, verbose=0)
        print(ida,",",idb,",",idc,",",idd,",",ide,',{:4f}'.format(eval_model[0]),',{:4f}'.format(eval_model[1]))
        #print('loss: {:4f}'.format(eval_model[0]))
        #print('accuracy: {:4f}'.format(eval_model[1]))

        #evaluate_model(history,model)
'''

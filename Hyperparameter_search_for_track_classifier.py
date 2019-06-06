import sys
import os
import numpy as np
import pandas as pd
import seaborn as sns
import keras.optimizers
import matplotlib.pyplot as plt

import keras
from keras import regularizers
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import multi_gpu_model
from keras.optimizers import SGD
from keras.constraints import maxnorm

from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV

def eval_loss():
    # summarize history for loss
    plt.plot(history.history['loss'])
    #plt.plot(history.history['val_loss'])
    #plt.title('model loss')
    #plt.ylabel('loss')
    #plt.xlabel('epoch')
    #plt.legend(['train', 'test'], loc='upper right')
    #plt.show()

#pd.read_csv("/data/TrackFakeReal.csv")
#df = pd.read_csv("/data/TrackFakeReal.csv")
#df = pd.read_csv("/data/TrackFakeRealGauss.csv")
#df = pd.read_csv("/data/dataset200")
#df = pd.read_csv("/data/output/bigds")
#X= df.iloc[:,0:58]
#y= df.iloc[:,59]

#X= df.iloc[:,0:120]
#y= df.iloc[:,121]

#print(event_prefix)
#print(len(df.columns))

seed = 7
np.random.seed(seed)
event_prefix = sys.argv[1]
df = pd.read_csv(event_prefix)

#cols=df.columns


#Separate predictors X and target y

#88
cols=len(df.columns)-1
print(cols)
#print(df.num_columns())


X= df.iloc[:,0:cols]
y= df.iloc[:,cols]

#Scale data
sc = StandardScaler()
X = sc.fit_transform(X)

encoder = LabelEncoder()
encoder.fit(y)
encoded_Y = encoder.transform(y)

#Separate Train and Test
X_train, X_test, y_train, y_test = train_test_split(X, encoded_Y, test_size=0.05)

def create_baseline(init_mode='uniform', neurons = 8,activation='relu',learn_rate=0.0001, opt='RMSprop'):
    init_mode=init_mode

    classifier = Sequential()
    classifier.add(Dense(neurons, activation=activation, kernel_initializer=init_mode, input_dim=cols))
    classifier.add(Dense(neurons, activation=activation, kernel_initializer=init_mode))
    classifier.add(Dense(1, activation=activation, kernel_initializer=init_mode)) #Output Layer

    #opt = keras.optimizers.RMSprop(lr=learn_rate, rho=0.9, epsilon=None, decay=0.0)
    #classifier.compile(optimizer = opt,loss='binary_crossentropy', metrics =['accuracy'])
    classifier.compile(optimizer =opt,loss='binary_crossentropy', metrics =['accuracy'])

    return classifier

model=create_baseline()

# create model
# init_mode='uniform',
model = KerasClassifier(build_fn=create_baseline,epochs=10,init_mode='uniform',neurons=8,activation='relu',learn_rate=0.0001)

# use verbose=0 if you do not want to see progress
########################################################
# Use scikit-learn to grid search
activation =  ['hard_sigmoid'] # , 'relu', 'tanh', 'sigmoid', 'linear'] # softmax, softplus, softsign
momentum = [0.0, 0.2, 0.4, 0.6, 0.8, 0.9]
learn_rate = [0.001, 0.01, 0.1, 0.2, 0.3]
dropout_rate = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
weight_constraint=[1, 2, 3, 4, 5]
neurons = [2]#,4,8]#,16,32,64,128] #[2,4,8,16] #,32,64,128,512,1024,2048]

init = ['he_normal',] #'uniform', 'lecun_uniform', 'normal', 'zero', 'glorot_normal', 'glorot_uniform', 'he_uniform']
opt = [ 'Adadelta'] #,'SGD', 'RMSprop', 'Adagrad', 'Adam', 'Adamax', 'Nadam']

##############################################################
# grid search epochs, batch size
epochs = [30] # add 50, 100, 150 etc
#batch_size = [100,200] #,5,10,20] #1000, 5000] # add 5, 10, 20, 40, 60, 80, 100 etc
param_grid = dict(epochs=epochs, neurons=neurons,init_mode=init,activation=activation, opt=opt)
#param_grid = dict(epochs=epochs, batch_size=batch_size, init_mode=init, neurons=neurons,activation=activation,learn_rate=learn_rate)

##############################################################
grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1)
#grid =RandomizedSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1)

grid_result = grid.fit(X_train,y_train)
##############################################################
# summarize results

print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']

for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))
##############################################################

#print(grid_result.cv_results_)
#history=model.fit(X_train,y_train, epochs=100, validation_data=(X_test, y_test)) #,  batch_size=20)
#eval_model=model.evaluate(X_test, y_test)
#print('Final test set loss: {:4f}'.format(eval_model[0]))
#print('Final test set accuracy: {:4f}'.format(eval_model[1]))

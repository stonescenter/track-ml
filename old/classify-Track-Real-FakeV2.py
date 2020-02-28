import sys
import os
import numpy as np
import pandas as pd
import seaborn as sns

import matplotlib.pyplot as plt

from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import multi_gpu_model

from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score

# fix random seed for reproducibility
seed = 7
np.random.seed(seed)

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
df = pd.read_csv("/data/TrackFakeRealGauss.csv")
#df = pd.read_csv("/data/output/bigds")

#Separate predictors X and target y
X= df.iloc[:,0:120]
y= df.iloc[:,121]

#Scale data
sc = StandardScaler()
X = sc.fit_transform(X)

#Separate Train and Test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

def create_baseline():
    classifier = Sequential()
    classifier.add(Dense(1200, activation='relu', kernel_initializer='random_normal', input_dim=120))
    classifier.add(Dropout(0.5))
    classifier.add(Dense(2560, activation='relu', kernel_initializer='random_normal'))
    classifier.add(Dropout(0.5))
    classifier.add(Dense(2560, activation='relu', kernel_initializer='random_normal'))
    classifier.add(Dropout(0.5))
    classifier.add(Dense(2560, activation='relu', kernel_initializer='random_normal'))
    classifier.add(Dense(1, activation='sigmoid', kernel_initializer='random_normal')) #Output Layer
    classifierp = multi_gpu_model(classifier, gpus=2)
    classifierp.compile(optimizer ='rmsprop',loss='binary_crossentropy', metrics =['accuracy'])
    return classifierp

def create_with_dropout():
    classifier = Sequential()
    classifier.add(Dense(2400, input_dim=120, activation='relu'))
    classifier.add(Dropout(0.5))
    classifier.add(Dense(2400, activation='relu'))
    classifier.add(Dropout(0.5))
    classifier.add(Dense(1, activation='sigmoid'))
    classifierp = multi_gpu_model(classifier, gpus=2)
    classifierp.compile(optimizer ='rmsprop',loss='binary_crossentropy', metrics =['accuracy'])
    return classifierp

# evaluate model with standardized dataset
#estimator = KerasClassifier(build_fn=create_baseline, epochs=5, batch_size=5, verbose=3)
#kfold = StratifiedKFold(n_splits=3, shuffle=True, random_state=seed)
##results = cross_val_score(estimator, X, y, cv=kfold)
#print("Results: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))

# evaluate baseline model with standardized dataset
np.random.seed(seed)
estimators = []
estimators.append(('standardize', StandardScaler()))
estimators.append(('mlp', KerasClassifier(build_fn=create_baseline, epochs=5, batch_size=5, verbose=3)))
pipeline = Pipeline(estimators)
kfold = StratifiedKFold(n_splits=3, shuffle=True, random_state=seed)
results = cross_val_score(pipeline, X, y, cv=kfold)
print(results)
print("Standardized: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))

np.random.seed(seed)
estimators = []
estimators.append(('standardize', StandardScaler()))
estimators.append(('mlp2', KerasClassifier(build_fn=create_with_dropout, epochs=5, batch_size=5, verbose=3)))
pipeline = Pipeline(estimators)
kfold = StratifiedKFold(n_splits=3, shuffle=True, random_state=seed)
results = cross_val_score(pipeline, X, y, cv=kfold)
print(results)
print("Standardized 2: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))


#history=classifier.fit(X_train,y_train, epochs=1000, validation_data=(X_test, y_test)) #,  batch_size=20000)
#eval_model=classifier.evaluate(X_test, y_test)
#print('Final test set loss: {:4f}'.format(eval_model[0]))
#print('Final test set accuracy: {:4f}'.format(eval_model[1]))

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

# fix random seed for reproducibility
seed = 7
np.random.seed(seed)
   
event_prefix = sys.argv[1]
df = pd.read_csv(event_prefix)

cols=len(df.columns)-1

X= df.iloc[:,0:cols]
y= df.iloc[:,cols]

#Scale data
sc = StandardScaler()
X = sc.fit_transform(X)

encoder = LabelEncoder()
encoder.fit(y)
encoded_Y = encoder.transform(y)

#Separate Train and Test
X_train, X_test, y_train, y_test = train_test_split(X, encoded_Y, test_size=0.3)


activation = 'hard_sigmoid'
neurons    = 8
init_mode  = 'he_normal'
opt        = 'Adadelta'
epochs_num = 500

classifier = Sequential()

classifier.add(Dense(neurons, activation=activation, kernel_initializer=init_mode, input_dim=cols))
classifier.add(Dense(neurons, activation=activation, kernel_initializer=init_mode))
classifier.add(Dense(1, activation=activation, kernel_initializer=init_mode)) 

classifier.compile(optimizer =opt,loss='binary_crossentropy', metrics =['accuracy'])

history=classifier.fit(X_train,y_train, epochs=epochs_num, validation_data=(X_test, y_test))
eval_model=classifier.evaluate(X_test, y_test)

print('Final test set loss: {:4f}'.format(eval_model[0]))
print('Final test set accuracy: {:4f}'.format(eval_model[1]))

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper right')
plt.show()

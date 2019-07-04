from random import seed
from random import randint
from numpy import array

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM

from math import sqrt

from sklearn.metrics import mean_squared_error

import numpy as np
import pandas as pd

import sys
import os
import ntpath

cols=13
nextpart=cols+3

seq_length=cols-1
# define LSTM configuration
n_batch = 1
#n_epoch = 100
n_epoch = 150

# generate training data
seed(1)
event_prefix = sys.argv[1]
event_file_name=ntpath.basename(event_prefix)

df = pd.read_csv(event_prefix)
dataX= df.iloc[:,1:cols]
b = dataX.values.flatten()

n_patterns=len(df)

X=np.reshape(b,(n_patterns,12,1))
y= df.iloc[:,cols:nextpart]

# create LSTM
model = Sequential()
model.add(LSTM(10, stateful=False, input_shape=(cols-1, 1)))
model.add(Dense(3))
model.compile(loss='mean_squared_error', optimizer='adam',metrics =['accuracy'])

model.fit(X, y, epochs=n_epoch, verbose=2) #, batch_size=n_batch

result = model.predict(X, batch_size=n_batch, verbose=0)

print(y)
print(result)

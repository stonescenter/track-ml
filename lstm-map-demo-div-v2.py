import matplotlib.pyplot as plt

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import concatenate,Input,Flatten
from keras.models import Model

from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

from random import seed
from random import randint
from numpy import array

from math import sqrt

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
n_epoch = 250

# generate training data
seed(1)
event_prefix = sys.argv[1]
event_file_name=ntpath.basename(event_prefix)

df = pd.read_csv(event_prefix)

dataX2=df.iloc[:, [1,2,3,7,8,9,13,14,15]]
dataXfeatures=df.iloc[:, [4,5,6,10,11,12,16,17,18]]

b = dataX2.values.flatten()
bfeat=dataXfeatures.values.flatten()

n_patterns=len(df)

X     = np.reshape(b,(n_patterns,3,3))
Xfeat = np.reshape(b,(n_patterns,3,3))
y=df.iloc[:, [19,20,21]]

X_train, X_test, Xfeat_train, Xfeat_test, y_train, y_test = train_test_split(X, Xfeat, y, test_size=0.1)

xshape=Input(shape=(3,3))
yshape=Input(shape=(3,3))

#LSTM layers
admi=LSTM(120,return_sequences=False)(xshape)
pla=LSTM(120,return_sequences=False)(yshape)
out=concatenate([admi,pla],axis=-1)

output=Dense(3)(out)
model = Model(inputs=[xshape, yshape], outputs=output)
#model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])
model.compile(loss='mean_squared_error', optimizer='adam',metrics =['accuracy'])
#model.fit([data_a, data_b],labels,batch_size=1, epochs=10)

history = model.fit([X_train, Xfeat_train], y_train, epochs=n_epoch, verbose=2) #validation_data=([X_test, Xfeat_test],

eval_model=model.evaluate([X_test, Xfeat_test], y_test)

print('Final test set loss: {:4f}'.format(eval_model[0]))
print('Final test set accuracy: {:4f}'.format(eval_model[1]))

model.save("model.h5")
print("Saved model to disk")

plt.plot(history.history['loss'])
#plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper right')
plt.savefig("/home/silvio/loss.png")
#plt.show()

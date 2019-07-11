import matplotlib.pyplot as plt

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import concatenate,Input,Flatten
from keras.models import Model
from keras.models import load_model

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

import scipy.stats

import seaborn as sns

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

X_train, X_test, Xfeat_train, Xfeat_test, y_train, y_test = train_test_split(X, Xfeat, y, test_size=0.3)

xshape=Input(shape=(3,3))
yshape=Input(shape=(3,3))

# load model
model = load_model('model.h5')
result = model.predict([X_test, Xfeat_test]) #, batch_size=n_batch, verbose=0)

x0 = y_test.iloc[:, [0]]
x1 = y_test.iloc[:, [1]]
x2 = y_test.iloc[:, [2]]
pred0 = result[:, [0]]
pred1 = result[:, [1]]
pred2 = result[:, [2]]

# plot
fig, axes = plt.subplots(1, 3, figsize=(25, 10), sharey=True, dpi=100)
sns.distplot(x0 , color="dodgerblue", ax=axes[0], axlabel='X')
sns.distplot(pred0 , color="dodgerblue", ax=axes[0], axlabel='X')
sns.distplot(x1 , color="deeppink", ax=axes[1], axlabel='Y')
sns.distplot(pred1 , color="deeppink", ax=axes[1], axlabel='Y')
sns.distplot(x2 , color="gold", ax=axes[2], axlabel='Z')
sns.distplot(pred2 , color="gold", ax=axes[2], axlabel='Z')

plt.savefig("/home/silvio/eval.png")
plt.show()

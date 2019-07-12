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

from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()


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

diff0 = x0.copy()
diff0['Score_diff'] =  x0.sub(pred0, axis=0) 
df0 = diff0.iloc[:, [1]]

diff1 = x1.copy()
diff1['Score_diff'] =  x1.sub(pred1, axis=0) 
df1 = diff1.iloc[:, [1]]

diff2 = x2.copy()
diff2['Score_diff'] =  x2.sub(pred2, axis=0) 
df2 = diff2.iloc[:, [1]]

# plot

fig, axes = plt.subplots(2, 3, figsize=(25, 10), sharey=True, dpi=100)
sns.distplot(x0 , color="blue", ax=axes[0,0], axlabel='X-original')
sns.distplot(pred0 , color="dodgerblue", ax=axes[0,0], axlabel='X-predicted')
sns.distplot(x1 , color="pink", ax=axes[0,1], axlabel='Y-original')
sns.distplot(pred1 , color="deeppink", ax=axes[0,1], axlabel='Y-predicted')
sns.distplot(x2 , color="gold", ax=axes[0,2], axlabel='Z-original')
sns.distplot(pred2 , color="yellow", ax=axes[0,2], axlabel='Z-predicted')

sns.distplot(df0 , color="blue", ax=axes[1,0], axlabel='X-difference')
sns.distplot(df1 , color="pink", ax=axes[1,1], axlabel='Y-difference')
sns.distplot(df2 , color="gold", ax=axes[1,2], axlabel='Z-difference')
plt.savefig("/home/silvio/eval.png")
plt.show()

'''
# plot
#fig, axes = plt.subplots(1, 1, figsize=(25, 10), sharey=True, dpi=100)
sns.plot(x0 , color="blue",  axlabel='X-original')
sns.plot(pred0 , color="dodgerblue", axlabel='X-predicted')

plt.show()
'''

'''
#print(df.iloc[:, [0]])
#print(df)
rg=range(20000)


dt = x0.copy()
#dt["id"]=rg
dt['id'] = dt.reset_index().index
dt["X-original"]=y_test.iloc[:, [0]]
dt["X-predicted"]=result[:, [0]]

print(dt)

dt["Y-original"]=y_test.iloc[:, [1]]
dt["Y-predicted"]=result[:, [1]]


dt["Z-original"]=y_test.iloc[:, [2]]
dt["Z-predicted"]=result[:, [2]]
'''
'''
sns.set(style="whitegrid")
values = dt
data = pd.DataFrame(values, columns=["X-original", "X-predicted"])
data = data.rolling(7).mean()

data1 = pd.DataFrame(values, columns=["Y-original", "Y-predicted"])
data1 = data.rolling(7).mean()

data2 = pd.DataFrame(values, columns=["Z-original", "Z-predicted"])
data2 = data.rolling(7).mean()

fig, axes = plt.subplots(1, 3, figsize=(25, 10), sharey=True, dpi=100)
sns.relplot(data=data, palette="tab10", ax=axes[0])# , linewidth=2.5)
sns.relplot(data=data1, palette="tab10", ax=axes[1])# ,linewidth=2.5)
sns.relplot(data=data2, palette="tab10", ax=axes[2]) #,linewidth=2.5)



plt.show()
'''
'''
#sns.scatterplot(x="X-original", y="X-predicted", data=dt)
values = dt
data = pd.DataFrame(values, columns=["id","X-original", "X-predicted"])
data = data.rolling(7).mean()

sns.relplot(x="id", y="X-predicted", kind="scatter", data=data) #col="region", 
plt.show()
'''



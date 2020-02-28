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

from keras.models import model_from_json

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
X_train, X_test, y_train, y_test = train_test_split(X, encoded_Y, test_size=0.99)

# load json and create model
json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("model.h5")
print("Loaded model from disk")

# evaluate loaded model on test data
loaded_model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
score = loaded_model.evaluate(X_test, y_test, verbose=0)
print("%s: %.2f%%" % (loaded_model.metrics_names[1], score[1]*100))

ok=0
nok=0

for i in range(X_test.shape[0]):
    t=X_test[i,]
    res=y_test[i,]
    t = t.reshape(1, 170)
    ynew = loaded_model.predict(t)
    if (ynew[0] == res):
        ok=ok+1
    else:
        nok=nok+1
    #print("Predicted=%s Original=%s," % (ynew[0],res))
print("ok=%s nok=%s," % (ok,nok))

#print("4169,0.16257267554244034,0.3948779823450284,29.73094544685053,13.883099555969238,15.327799797058105,330.0,289.50000081431295,0.07129331647641537,29.705357616804587,14.267600059509276,16.293699264526367,76.6,307.099999655413,0.044928381651056705,66.877500806563,30.0802001953125,53.34590148925781,74.375,965.8749999086886,0.05630402043971279,105.59660393366276,49.28499984741211,93.74629974365234,204.42857142857144,439.42857161639756,0.05457108987021191,154.63027791807266,77.07489776611328,145.4029998779297,249.7142857142857,113.71428491676852,0.06380640539452452,224.8849390698438,129.92399597167972,229.2720031738281,208.5,63.99961349682345,1.0244375321185175,336.82463810393995,197.2239990234375,320.260009765625,156.5,54.00014785596944,0.991983635670883,372.15739072499633,317.0530090332031,459.3370056152344,298.0,83.99924115715092,1.0618309765701055,450.69596600975694,479.1919860839844,624.7890014648438,587.0,49.99990095436578,1.0049400746369888,544.3073602360178,472.0409851074219,617.7760009765625,29.8,44.20079848150896,0.9640609923836267,580.7384252322953,658.7109985351562,796.426025390625,280.5,2.0105380222583684,0.978640312399509,438.67932445374765,913.4580078125,1041.550048828125,274.0,6.0,1.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0")
#print("----")
#print(X_test[1,])
#print(t.shape)
#print(t.shape[0])
#print(t)
#print(t[3])
#print(t.shape[1])

#t = t.reshape((t.shape[1], 1))
#print(t.shape[0])
#print(t.shape[1])


#ynew = loaded_model.predict(X_test[1,])
# show the inputs and predicted outputs

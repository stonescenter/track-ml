import sys
import os
import numpy as np
import pandas as pd
import seaborn as sns

import matplotlib.pyplot as plt 

from keras.models import Sequential
from keras.layers import Dense

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler

#pd.read_csv("/data/TrackFakeReal.csv")
df = pd.read_csv("/data/TrackFakeReal.csv")
#print("2 ")
#pd.head(2)
#print(df.head(2))
print(df.shape[1])

#Separate predictors X and target y
X= df.iloc[:,0:120]
#print(X)    
y= df.iloc[:,121]
#print(y)

#Scale data
sc = StandardScaler()
X = sc.fit_transform(X)
X

#Train and Test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

#create NN
classifier = Sequential()
#First Hidden Layer
classifier.add(Dense(120, activation='relu', kernel_initializer='random_normal', input_dim=120))
#Second  Hidden Layer
classifier.add(Dense(32, activation='relu', kernel_initializer='random_normal'))
#Output Layer
classifier.add(Dense(1, activation='sigmoid', kernel_initializer='random_normal'))

#Compiling the neural network
classifier.compile(optimizer ='adam',loss='binary_crossentropy', metrics =['accuracy'])

#Fitting the data to the training dataset batch_size=10, 
history=classifier.fit(X_train,y_train, epochs=10)

eval_model=classifier.evaluate(X_train, y_train)
eval_model

print('Final test set loss: {:4f}'.format(eval_model[0]))
print('Final test set accuracy: {:4f}'.format(eval_model[1]))

# summarize history for loss
plt.plot(history.history['loss'])
#plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper right')
plt.show()

#Compare Predicted an Original
ynew = classifier.predict_classes(X_test)
correct = 0

#Cell for testing prediction model
import sys
import numpy as np
import pandas as pd
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
from keras.models import load_model

sys.path.append('/home/silvio/git/track-ml-1')
from lib_data_manipulation import *

def position_3D_approximation(result):
    # result => predicted

    global dfyclone

    #this dataframe receives all X,Y,Z predicted considering a set of hists
    df3d = pd.DataFrame({'X':result[:,0],'Y':result[:,1],'Z':result[:,2]})

    df3d['X-pred'] = 0
    df3d['Y-pred'] = 0
    df3d['Z-pred'] = 0
    df3d['hit_id'] = 0
    df3d['volume_id'] = 0
    df3d['layer_id'] = 0
    df3d['module_id'] = 0
    df3d['value'] = 0

    #for each predicted hit, we will approximate to the closest hit considering gemoetric distance
    for index, row in df3d.iterrows():
        #obtain the row with least geometric distance between predicted row and original rows (in yclone)
        Xpred=df3d.loc[index, 'X']
        Ypred=df3d.loc[index, 'Y']
        Zpred=df3d.loc[index, 'Z']

        #in this column we will create the geometric distance from all available hits and the current hit
        dfyclone['geodist'] = ( ((dfyclone[1] - Xpred) **2) + ((dfyclone[2] - Ypred) **2)   + ((dfyclone[3] - Zpred) **2) )

        dfyclone=dfyclone.sort_values(by=['geodist'])

        df3d.loc[index, 'X-pred'] = dfyclone[1].values[0]
        df3d.loc[index, 'Y-pred'] = dfyclone[2].values[0]
        df3d.loc[index, 'Z-pred'] = dfyclone[3].values[0]
        df3d.loc[index, 'hit_id'] = dfyclone[0].values[0]
        df3d.loc[index, 'volume_id'] = dfyclone[4].values[0]
        df3d.loc[index, 'layer_id'] = dfyclone[5].values[0]
        df3d.loc[index, 'module_id'] = dfyclone[6].values[0]
        df3d.loc[index, 'value'] = dfyclone[7].values[0]

        #dfyclone.drop(dfyclone.index[0], inplace=True)

    #print(df3d)
    df3d.drop('X', axis=1, inplace=True)
    df3d.drop('Y', axis=1, inplace=True)
    df3d.drop('Z', axis=1, inplace=True)

    #return the fourth hit of all tracks
    return(df3d)

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
set_session(sess)

NN  = int(sys.argv[1])
file = sys.argv[2]
h5file = sys.argv[3]

if (NN == 1):
    #trained_model_file="/home/silvio/input_files_for_track/model_top04_1.h5"
    trained_model_file=h5file
else:
    trained_model_file="/home/silvio/input_files_for_track/model_top04_2.h5"

model = load_model(trained_model_file)
#inputfile="/home/silvio/testInf"
inputfile=file

#Cell for testing prediction model
df = pd.read_csv(inputfile)
#df = pd.read_csv("/home/silvio/all-Train.csv")

#df1 = df.iloc[:,:39]
df1 = df.iloc[:,10:54]

#create input data for LSTM Neural Network
#dataX2 = df1.iloc[:, [ 2,3,4,10,11,12,18,19,20]]
#dataXfeatures = df1.iloc[:, [ 9,17,24 ]]

dataX2 = df1.iloc[:, [ 0,1,2,8,9,10,16,17,18,24,25,26]]
dataXfeatures = df1.iloc[:, [ 6,14,22,30 ]]

b = dataX2.values.flatten()
bfeat = dataXfeatures.values.flatten()
n_patterns = len(df)
X     = np.reshape(b,(n_patterns,4,3))
Xfeat = np.reshape(bfeat,(n_patterns,4,1))

#create input data for MLP  Neural Network
XMLP= df1.iloc[:, [ 2,3,4,9,10,11,12,17,18,19,20,24]]

#original Result
#resorg= df1.iloc[:, [ 26,27,28]]
#resorg= df1.iloc[:, [ 27,28,29]]
resorg= df1.iloc[:, [ 32,33,34]]
resorg.to_csv('/home/silvio/resorg', index = False)


print(NN)
if (NN == 1):
    result = model.predict([X, Xfeat],verbose=1)
    print("!X")
    print(X[0:10,:])
    print("Xfeat")
    print(Xfeat[0:10,:])
    print("resorg")
    print(resorg.iloc[0:10,:])
else:
    result = model.predict(XMLP,verbose=1)
    print("!XMLP")
    #print(XMLP)

df3d = pd.DataFrame({'X':result[:,0],'Y':result[:,1],'Z':result[:,2]})


#print("resorg")
#print(resorg)
#print("result")
#print(df3d)


#all_hits_available_file="/home/silvio/testInf-allhits"
#allhits= df1.iloc[:, [ 26,27,28,29,30,31,32]]
#allhits.to_csv(all_hits_available_file)
#df_all_hits = pd.read_csv(all_hits_available_file)

df_all_hits = df1.iloc[:, [ 31,32,33,34,35,36,37,38]]

#df_all_hits = df1.iloc[:, [ 26,27,28,29,30,31,32,33]]
yy=df_all_hits.iloc[:,:]
yclone = np.copy(yy)
dfyclone = pd.DataFrame.from_records(yclone)

df3dapp=position_3D_approximation(result)

print("original: ")
print(df1.iloc[:,0:34])
print("All hits: ")
print(df_all_hits)
print("Predicted: ")
print(df3dapp)

df3dapp.to_csv('/home/silvio/Res3', index = False)
df3d.to_csv('/home/silvio/beforemappingRes3', index = False)

dftemp = pd.DataFrame(index=range(len(df)),columns=range(12))
dftemp[0]=resorg.iloc[:,[0]]
dftemp[1]=resorg.iloc[:,[1]]
dftemp[2]=resorg.iloc[:,[2]]

dftemp[3]=df3d.iloc[:,[0]]
dftemp[4]=df3d.iloc[:,[1]]
dftemp[5]=df3d.iloc[:,[2]]

dftemp[6]=df3dapp.iloc[:,[0]]
dftemp[7]=df3dapp.iloc[:,[1]]
dftemp[8]=df3dapp.iloc[:,[2]]

dftemp[9]=   (((dftemp[0]-dftemp[3])**2)+((dftemp[1]-dftemp[4])**2)+((dftemp[2]-dftemp[5])**2)).pow(1./2)
dftemp[10]=   (((dftemp[0]-dftemp[6])**2)+((dftemp[1]-dftemp[7])**2)+((dftemp[2]-dftemp[8])**2)).pow(1./2)
dftemp[11]=   (((dftemp[3]-dftemp[6])**2)+((dftemp[4]-dftemp[7])**2)+((dftemp[5]-dftemp[8])**2)).pow(1./2)

print (dftemp.iloc[0:10,:])
dftemp=dftemp.sort_values(by=[10])
print (dftemp)

print ("average distance prediction" , dftemp[9].mean())
print ("average distance approximation" , dftemp[10].mean())

dftemp22 = dftemp[ dftemp[10] == 0]
print ("0 diff" , dftemp22.shape[0])

outputfig="plot-original-predicted.png"
sns_plot = sns.distplot(dftemp.iloc[:,9:10])
sns_plot.set(xlabel='Average Distance in MM - original x predicted ', ylabel='Frequency')
plt.savefig(outputfig)

outputfig="plot-original-approximated.png"
sns_plot2 = sns.distplot(dftemp.iloc[:,10:11])
sns_plot2.set(xlabel='Average Distance in MM - original x approximated ', ylabel='Frequency')
plt.savefig(outputfig)

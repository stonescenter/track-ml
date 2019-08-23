from math import sqrt
import sys
import scipy.stats
import seaborn as sns
import pandas as pd
from matplotlib import pyplot as plt

original_tracks      = sys.argv[1]
reconstructed_tracks = sys.argv[2]
outputfig            = sys.argv[3]   
dfOriginal      = pd.read_csv(original_tracks)
dfReconstructed = pd.read_csv(reconstructed_tracks)
lines   = dfOriginal.shape[0]
columns = dfOriginal.shape[1]

dfEval  = pd.DataFrame(index=range(lines),columns=range(29))

ind_dfEval=0

#for hit in range(1, 20):
for hit in range(1, 28):
    dataOR=dfOriginal.iloc[:, [ (hit*6)+1,(hit*6)+2,(hit*6)+3 ]]
    dataRE=dfReconstructed.iloc[:, [ (hit*6)+1,(hit*6)+2,(hit*6)+3 ]]

    dftemp = pd.DataFrame(index=range(lines),columns=range(7))
    dftemp[0]=dataOR.iloc[:,[0]]
    dftemp[1]=dataOR.iloc[:,[1]]
    dftemp[2]=dataOR.iloc[:,[2]]
    dftemp[3]=dataRE.iloc[:,[0]]
    dftemp[4]=dataRE.iloc[:,[1]]
    dftemp[5]=dataRE.iloc[:,[2]]
    dftemp[6]=((dftemp[0]-dftemp[3])**2)+((dftemp[1]-dftemp[4])**2)+((dftemp[2]-dftemp[5])**2)

    dfEval[ind_dfEval] = dftemp[6]
    ind_dfEval=ind_dfEval+1

col = dfEval.loc[: , 0:26]
dfEval[27] = col.mean(axis=1)
dfEval[28] = col.std(axis=1)

print(dfEval.shape)
print(dfEval.iloc[:,[27]].min())
print(dfEval.iloc[:,[27]].max())
plt.figure()
#sns_plot = sns.pairplot(dfEval.iloc[:,[27]]) #, hue='27', size=2.5)
#sns_plot = sns.pairplot(dfEval) #, hue='27', size=2.5)
sns_plot = sns.distplot(dfEval.iloc[:,[27]]);
plt.savefig(outputfig)

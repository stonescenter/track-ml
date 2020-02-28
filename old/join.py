
#join files generated

import sys
sys.path.append('/home/silvio/git/track-ml-1/utils')
from tracktop import *

tracks_final = pd.DataFrame()

rangeBeginSTR = sys.argv[1]
rangeEndSTR = sys.argv[2]
dirout = sys.argv[3]
dir = sys.argv[4]
#/home/silvio/trainF

rangeBegin = int(rangeBeginSTR)
rangeEnd  = int(rangeEndSTR) +1

print("dirout ", dirout, " rangeBegin ", rangeBegin, " rangeEnd ", rangeEnd )

for i in range(rangeBegin, rangeEnd):
    print(dir+"/"+str(i)+".csv")
    tracks = pd.read_csv(dir+"/"+str(i)+".csv")
    tracks_final = pd.concat([tracks_final, tracks], ignore_index=True)

tracks_final.to_csv(dirout, index = False)

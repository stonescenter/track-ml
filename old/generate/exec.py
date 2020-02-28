import sys
import os

#Id of first event file
rangeBeginSTR = sys.argv[1]
#Amount of files
rangeEndSTR = sys.argv[2]
#Original directory
dir = sys.argv[3]
#Dir with all output files for each event
dirout = sys.argv[4]

rangeBegin = int(rangeBeginSTR)
interval  = int(rangeEndSTR) +1

b=rangeBegin
#b=1000
#e=0

#executes 40 times different event files
for k in range(0,40):
    e=b+interval
    print(b,e)
    execfile="python /home/silvio/generate.py "+str(b)+" "+str(e)+" " + dir + " " + dirout + "/" + str(k)+".csv &"
    print(execfile)
    os.system(execfile)
    #call(["python", execfile])
    b=e+1

import pandas as pd
import sys
#import socket
import os

sys.path.append('/home/silvio/git2/track-ml-1')
from lib_data_manipulation import *

event_prefix  = sys.argv[1]
output_prefix = sys.argv[2]
input_for_trainning = sys.argv[3]
modelF = sys.argv[4]
lossF  = sys.argv[5]

#hostname = socket.gethostname()
NN  = sys.argv[6]

print (event_prefix,output_prefix)
#new dataset with cell unrolled
print ("create_input_data 1 ")
#create_input_data(event_prefix = event_prefix,output_prefix = output_prefix, aux_am_per_hit=10000, min=4, max=4, maximunAmountofHitsinDB=20, columnsperhit=8, firstColumnAfterParticle=10)
print ("create_input_data 2 ")

#put_each_hit_in_a_single_line_train(event_prefix =output_prefix, output_prefix =input_for_trainning)
put_each_hit_in_a_single_line_train(event_prefix =event_prefix, output_prefix =input_for_trainning)

#input_for_trainning=event_prefix #output_prefix


execline="python ~/git2/track-ml-1/lstm-NN-cell-unrolled.py " + input_for_trainning + " " + modelF + " " + lossF + " " + str(NN)
print (execline)
os.system(execline)

#create_input_data(event_prefix = event_prefix,output_prefix = output_prefix, aux_am_per_hit=20, min=4, max=6, maximunAmountofHitsinDB=28, columnsperhit=6, firstColumnAfterParticle=7)
#create_input_data_24(event_prefix = output_prefix,output_prefix = input_for_trainning)
#execline="python ~/git/track-ml-1/lstm-map-demo-div-v3.py " + input_for_trainning + " " + modelF + " " + lossF + " " + str(NN)

#execline="python ~/git/track-ml-1/lstm-map-demo-div-v3.py "+input_for_trainning+" ~/input_files_for_track/model_"+hostname+"_"+str(NN)+".h5 " + " ~/input_files_for_track/loss_"+hostname+"_"+str(NN)+".png " + str(NN)
#python ~/git/track-ml-1/lstm-map-demo-div-v3.py input_for_trainning ~/input_files_for_track/model_`hostname`_"$NN".h5 ~/input_files_for_track/loss_`hostname`_"$NN".png $NN

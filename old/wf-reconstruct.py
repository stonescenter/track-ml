import pandas as pd
import sys
import os

sys.path.append('/home/silvio/git2/track-ml-1')
from lib_data_manipulation import *

event_prefix  = sys.argv[1]
output_prefix = sys.argv[2]
input_for_reconstruct = sys.argv[3]
modelF  = sys.argv[4]
reconstructed_file  = sys.argv[5]
Neural_Network_Model= sys.argv[6]
eval_file= sys.argv[7]
outputfigseaborn = sys.argv[8]
org_track_viz = sys.argv[9]
recons_track_viz = sys.argv[10]
hist_eval  = sys.argv[11]


#print(event_prefix)
#print(output_prefix)
#print(input_for_reconstruct )
#print(modelF  )
#print(reconstructed_file  )
#print(Neural_Network_Model )
#print(eval_file )
#print(outputfigseaborn  )
#print(org_track_viz )
#print(recons_track_viz  )
#Neural_Network_Model=1

#create_input_data_for_reconstruct(event_prefix = event_prefix,output_prefix = output_prefix, aux_am_per_hit=20, min=1, max=20, maximunAmountofHitsinDB=20, columnsperhit=8, firstColumnAfterParticle=10)
#create_input_data(event_prefix = event_prefix,output_prefix = output_prefix, aux_am_per_hit=20, min=4, max=6)
#create_input_data_6(event_prefix = output_prefix,output_prefix = input_for_reconstruct)

print ("create_input_data 1 ")
create_input_data(event_prefix = event_prefix,output_prefix = output_prefix, aux_am_per_hit=100, min=5, max=20, maximunAmountofHitsinDB=20, columnsperhit=8, firstColumnAfterParticle=10)
print ("create_input_data 2 ")

'''
#output_prefix=event_prefix
put_each_hit_in_a_single_line(event_prefix =output_prefix, output_prefix =input_for_reconstruct)

execline="python ~/git/track-ml-1/lstm-map-demo-div-v2-inf.py"+" "+output_prefix+" "+input_for_reconstruct+" "+modelF+" "+reconstructed_file+" "+Neural_Network_Model
print (execline)
os.system(execline)

#Results Visualization
create_diference_per_track(reconstructed_tracks=reconstructed_file, original_tracks=output_prefix, eval_file=eval_file)
create_graphic(reconstructed_tracks = reconstructed_file, original_tracks = output_prefix, tracks_diffs = eval_file, path_original_track=org_track_viz,  path_recons_track=recons_track_viz)
create_histogram(tracks_diffs = eval_file, path_hist=hist_eval)
create_histogram_seaborn(tracks_diffs = eval_file, outputfig = outputfigseaborn)
'''

'''
create_input_data(event_prefix = event_prefix,output_prefix = output_prefix, aux_am_per_hit=10000, min=4, max=4, maximunAmountofHitsinDB=20, columnsperhit=8, firstColumnAfterParticle=10)
create_graphic_org(original_tracks = output_prefix, path_original_track=org_track_viz)
'''

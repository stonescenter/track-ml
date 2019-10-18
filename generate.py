import sys
sys.path.append('/home/silvio/git/track-ml-1/utils')
from tracktop import *

event_files = []
tracks_final = pd.DataFrame()

rangeBeginSTR = sys.argv[1]
rangeEndSTR = sys.argv[2]
dir = sys.argv[3]
dirout = sys.argv[4]

range_phiB=-0.5
range_phiE=0.5
range_etaB=-0.5
range_etaE=0.5

rangeBegin = int(rangeBeginSTR)
rangeEnd  = int(rangeEndSTR)

for i in range(rangeBegin, rangeEnd):
    event_files.append("event00000"+str(i))

for i in event_files:
    create_input(dir, str(i),
             # be sure to change the amount of tracks. remove
             # this parameter to cycle through all tracks in the file
             #n_tracks = 5000,
             output = str(i),
             ratio_discard_hit = 20,
             n_hits_range = [0,np.PINF],
             phi_range=[-0.5,0.5],
             eta_range=[-0.5,0.5],
             #delta_eta_range = [0,0.04],
             #delta_phi_range = [0,1],
             pt_range=[np.NINF,np.PINF]
             #silent = 'False'
             )
    tracks = pd.read_csv(str(i))
    tracks_final = pd.concat([tracks_final, tracks], ignore_index=True)

tracks_final.to_csv(dirout, index = False)

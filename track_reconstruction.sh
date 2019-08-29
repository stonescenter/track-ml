#!/bin/bash

#Track Reconstruction Workflow

#am -> amount of tracks to be reconstructed
#$1 -> type of neural network 1 lstm with features 2 mlp 3 lstm without features

user_dir=`echo ~`
dir="$user_dir"/input_files_for_track
NN=$1

if [ -d $dir ] ; then

  #python t2.py /data/trackMLDB/analysis/pt1p0_train_2_realv3 ~/input_files_for_track/input-reconstruct 5

  ~/git/track-ml-1/input-for-lstm-inf.sh ~/input_files_for_track/input-reconstruct ~/input_files_for_track/train_inferences_`hostname`
  python ~/git/track-ml-1/lstm-map-demo-div-v2-inf.py ~/input_files_for_track/input-reconstruct ~/input_files_for_track/train_inferences_`hostname` ~/input_files_for_track/model_`hostname`_"$NN".h5 ~/input_files_for_track/reconstructed_track_"$NN"_`hostname`.csv 0 $NN
#  python ~/git/track-ml-1/lstm-map-Evaluate.py ~/input_files_for_track/input-reconstruct ~/input_files_for_track/reconstructed_track_$NN_`hostname`.csv ~/input_files_for_track/outputhist_$NN_`hostname`.png ~/input_files_for_track/eval_$NN_`hostname`

  #am=10000

  #obtain a subset of tracks to be reconstructed
  #head -n $am /data/trackMLDB/analysis/pt1p0_train_2_realv3 > ~/input_files_for_track/pt1p0_train_2_realv3_"$am"_`hostname`
  #head -n $am /data/trackMLDB/analysis/pt2p0_train_1_realv3  > ~/input_files_for_track/pt2p0_train_1_realv3_"$am"_`hostname`
  #head -n $am /home/silvio/inf.csv > ~/input_files_for_track/pt1p0_train_2_realv3_"$am"_`hostname`

  #python ~/git/track-ml-1/equal_number_of_hits.py ~/input_files_for_track/pt2p0_train_1_realv3_"$am"_`hostname` ~/input_files_for_track/pt2p0_train_1_realv3_"$am"_`hostname`_v2 5

  #echo "~/input_files_for_track/pt2p0_train_1_realv3_"$am"_`hostname`_v2"
  #create a file with all hits to be reconstructed


  # reconstruct tracks
  #echo python ~/git/track-ml-1/lstm-map-demo-div-v2-inf.py ~/input_files_for_track/pt1p0_train_2_realv3_"$am" ~/input_files_for_track/train_2_"$am"_inferences ~/input_files_for_track/model.h5 ~/input_files_for_track/reconstructed_track.csv


  #time python ~/git/track-ml-1/lstm-map-demo-div-v2-inf.py ~/input_files_for_track/pt1p0_train_2_realv3_"$am"_`hostname` ~/input_files_for_track/train_2_"$am"_inferences_`hostname` ~/input_files_for_track/model_`hostname`.h5 ~/input_files_for_track/reconstructed_track_RANDOM_`hostname`.csv 1

  #echo python ~/git/track-ml-1/lstm-map-Evaluate.py ~/input_files_for_track/pt1p0_train_2_realv3_"$am" ~/input_files_for_track/reconstructed_track.csv
  #python ~/git/track-ml-1/lstm-map-Evaluate.py ~/input_files_for_track/pt1p0_train_2_realv3_"$am"_`hostname` ~/input_files_for_track/reconstructed_track_RANDOM_`hostname`.csv ~/input_files_for_track/outputhist_RANDOM_`hostname`.png

else
    echo "Error: Directory ~/input_files_for_track does not exists."
fi

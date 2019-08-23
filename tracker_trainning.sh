#!/bin/bash

user_dir=`echo ~`
dir="$user_dir"/input_files_for_track

if [ -d $dir ] ; then

am=1000

#obtain a subset of tracks to be reconstructed
head -n $am /data/trackMLDB/analysis/pt1p0_train_1_realv3 > ~/input_files_for_track/pt1p0_train_1_realv3_"$am"_`hostname`

#create a file with all hits to be reconstructed
~/git/track-ml-1/input-for-lstm.sh ~/input_files_for_track/pt1p0_train_1_realv3_"$am"_`hostname` ~/input_files_for_track/train_1_"$am"_`hostname`

python ~/git/track-ml-1/lstm-map-demo-div-v3.py ~/input_files_for_track/train_1_"$am"_`hostname` ~/input_files_for_track/model_`hostname`.h5 ~/input_files_for_track/loss_`hostname`.png

else
    echo "Error: Directory ~/input_files_for_track does not exists."
fi

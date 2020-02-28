#!/bin/bash

user_dir=`echo ~`
dir="$user_dir"/input_files_for_track
NN=$1
# NN -> type of neural network 1 lstm with features 2 mlp 3 lstm without features
#1 ->
#file="pt1p0_train_1_realv3"
#am=500 #0000
#am=100000

if [ -d $dir ] ; then

  #python t2.py /data/trackMLDB/analysis/pt1p0_train_1_realv3 /home/silvio/input_files_for_track/output 250
  ~/git/track-ml-1/input-for-lstm.sh ~/input_files_for_track/output ~/input_files_for_track/train_`hostname`
  python ~/git/track-ml-1/lstm-map-demo-div-v3.py ~/input_files_for_track/train_`hostname` ~/input_files_for_track/model_`hostname`_"$NN".h5 ~/input_files_for_track/loss_`hostname`_"$NN".png $NN

  #echo /data/trackMLDB/analysis/"$file"
  #echo ~/input_files_for_track/"$file"_"$am"_`hostname`

  #obtain a subset of tracks to train the model
  #head -n $am /data/trackMLDB/analysis/"$file" > ~/input_files_for_track/"$file"_"$am"_`hostname`

  #balanced file with the same amount of tracks per amount of hits
  #python ~/git/track-ml-1/equal_number_of_hits.py ~/input_files_for_track/"$file"_"$am"_`hostname` ~/input_files_for_track/"$file"_"$am"_`hostname`_v2 10 #2000

  #create a file with all hits to be reconstructed
  #~/git/track-ml-1/input-for-lstm.sh ~/input_files_for_track/"$file"_"$am"_`hostname`_v2 ~/"$file"/train_1_"$am"_`hostname`

  #python ~/git/track-ml-1/lstm-map-demo-div-v3.py ~/input_files_for_track/train_1_"$am"_`hostname` ~/input_files_for_track/model_`hostname`_"$1".h5 ~/input_files_for_track/loss_`hostname`_"$1".png $1

else
    echo "Error: Directory ~/input_files_for_track does not exists."
fi

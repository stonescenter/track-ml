#!/bin/bash

user_dir=`echo ~`
dir="$user_dir"/input_files_for_track
NN=$1 # NN -> type of neural network 1 lstm with features 2 mlp 3 lstm without features

if [ -d $dir ] ; then

  #modeldir="~/input_files_for_track/1/"
  modeldir="~/input_files_for_track/"
  #python ~/git/track-ml-1/wf-trainning.py /data/trackMLDB/analysis/pt1p0_train_1_realv3 "$modeldir"output_`hostname` "$modeldir"train_`hostname` "$modeldir"model_`hostname`_"$NN".h5 "$modeldir"loss_`hostname`_"$NN".png

  python ~/git2/track-ml-1/wf-trainning.py /home/silvio/testNN "$modeldir"output_`hostname` "$modeldir"train_`hostname` "$modeldir"model_`hostname`_"$NN".h5 "$modeldir"loss_`hostname`_"$NN".png $NN
  #python ~/git/track-ml-1/wf-trainning.py /home/silvio/testNN2 "$modeldir"output_`hostname` "$modeldir"train_`hostname` "$modeldir"model_`hostname`_"$NN"2.h5 "$modeldir"loss_`hostname`_"$NN"2.png $NN
  #python ~/git/track-ml-1/wf-trainning.py /home/silvio/testNN3 "$modeldir"output_`hostname` "$modeldir"train_`hostname` "$modeldir"model_`hostname`_"$NN"3.h5 "$modeldir"loss_`hostname`_"$NN"3.png $NN
  #python ~/git/track-ml-1/wf-trainning.py /home/silvio/all.csv "$modeldir"output_`hostname` "$modeldir"train_`hostname` "$modeldir"model_`hostname`_"$NN".h5 "$modeldir"loss_`hostname`_"$NN".png $NN

else
    echo "Error: Directory ~/input_files_for_track does not exists."
fi

#python ~/git/track-ml-1/lstm-map-demo-div-v3.py ~/input_files_for_track/train_`hostname` ~/input_files_for_track/model_`hostname`_"$NN".h5 ~/input_files_for_track/loss_`hostname`_"$NN".png $NN
#python t2.py /data/trackMLDB/analysis/pt1p0_train_1_realv3 ~/input_files_for_track/output 10 7 10
#python t3.py ~/input_files_for_track/output ~/input_files_for_track/train_`hostname`

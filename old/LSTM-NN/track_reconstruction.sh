#!/bin/bash

#Track Reconstruction Workflow

user_dir=`echo ~`
dir="$user_dir"/input_files_for_track
NN=$1 #$1 ->  NN type of neural network 1 lstm with features 2 mlp 3 lstm without features

if [ -d $dir ] ; then

  #orgtracks="/data/trackMLDB/analysis/pt1p0_train_2_realv3"


  orgtracks="/home/silvio/all-Train.csv"
  #orgtracks="/home/silvio/all-Inf.csv"
  #orgtracks="/home/silvio/all.csv"
  #orgtracks="/home/silvio/inf2.csv"
  inputreconstruct="~/input_files_for_track/input-reconstruct_`hostname`"
  allhits="~/input_files_for_track/train_inferences_`hostname`"
  model="~/input_files_for_track/model_`hostname`_$NN.h5"
  outputreconstruct="~/input_files_for_track/reconstructed_track_$NN-`hostname`.csv"
  eval="~/input_files_for_track/eval_$NN_`hostname`"
  seabornhist="/home/silvio//input_files_for_track/hist-seaborn_`hostname`_$NN.png"
  orgviz="/home/silvio/input_files_for_track/org-_`hostname`_$NN.html"
  recviz="/home/silvio/input_files_for_track/recons_`hostname`_$NN.html"
  hist="/home/silvio/input_files_for_track/hist-_`hostname`_$NN.html"

  #echo $orgtracks $inputreconstruct $allhits $model $outputreconstruct $NN $eval $seabornhist $orgviz $recviz $hist

  python ~/git2/track-ml-1/wf-reconstruct.py $orgtracks $inputreconstruct $allhits $model $outputreconstruct $NN $eval $seabornhist $orgviz $recviz $hist
  #python ~/git/track-ml-1/wf-reconstruct.py $orgtracks $inputreconstruct $allhits $model $outputreconstruct $NN $eval $seabornhist $orgviz $recviz $hist
else
    echo "Error: Directory ~/input_files_for_track does not exists."
fi

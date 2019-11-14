# LSTM Neural Network

Neural Network trainning is managed by script: wf-trainning.py that expects the following parameters:

* event_prefix
* output_prefix
* input_for_trainning 
* modelF: generated model  
* lossF:  generated loff graph

To train the Neural Network:

```
n 1000 /home/silvio/all-Train.csv > /home/silvio/T.csv
python wf-trainning.py /home/silvio/T.csv /home/silvio/Tr.csv /home/silvio/Tr.csv /home/silvio/input_files_for_track/model_top04_1.h5 /home/silvio/input_files_for_track/loss_top04_1.png
```

To make inferences:

```
python inference.py 1 /home/silvio/inf_1.csv /home/silvio/input_files_for_track/model_top04_1.h5
```

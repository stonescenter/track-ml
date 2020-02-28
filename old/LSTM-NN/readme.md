# LSTM Neural Network

Neural Network trainning is managed by script: wf-trainning.py that expects the following parameters:

* event_prefix
* output_prefix
* input_for_trainning
* modelF: generated model  
* lossF:  generated loff graph

Comand line exemplt to train the Neural Network:

```
python wf-trainning.py tr.csv ~/aux/tr-aux.csv ~/aux/tr-aux.csv ~/aux/model_top04_1.h5 ~/aux/loss_top04_1.png 1
```

inferences can be done using script: inference.py that expects the following parameters:

* NN  : type of neural network, accepts only 1 for LSTM
* file : dataset with 4 hits to predict the 5th
* h5file = Neural network trained

```
python inference.py 1 if.csv ~/aux/model_top04_1.h5
```

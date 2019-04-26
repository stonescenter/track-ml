# track-ml

Output of Data Preparation is a CSV file 

Each line is a particle and several hits of this particle 

Each line has 121 columns organized in the following way:

  - first 6 columns is particle information: tx, ty, tx, px, py, pz

  - 19 sets of 6 columns representing hits: tx,ty,tz, ch0, ch1, value
  
If the has less than 19 hits the remaings hits is fullfiled with zeros

the last colum 121 has 1 for fake hist and 0 for real hit

In oder to use more than one GPU in the same machine to train a keras model use the following comannd:

```
    multi_gpu_model(<keras_model>, gpus=<amout of GPUs>)
```

Code Sample:

```
classifier = Sequential()
classifier.add(Dense(2400, input_dim=120, activation='relu'))
classifierp = multi_gpu_model(classifier, gpus=2)
classifierp.compile(optimizer ='rmsprop',loss='binary_crossentropy', metrics =['accuracy'])
```

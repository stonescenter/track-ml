# Track Particle
This project is the first version of Track Particles problem and it is part of [SPRACE](https://sprace.org.br/) sponsored by [Serrapilheira](https://serrapilheira.org/). 
<p align="center">
	 <img width="450" height="400" src="./imgs/internal_reconstruction_3d.png"></img>
</p>

## Run
To run:
1. Clone our repository
2. Configure your conda environtment with env.yml file
3. There are many scripts to test with diferents models(MLP, CNN, LSTM, CNN-parallel and others). For example: `python main_train.py --config config_lstm.json`. It will create a model with a default LSTM configuration in a json file. You can inference data test with ` python main_inference.py --config config_lstm.json `.

## Accuracy of Algorithm
We are using regressions metrics for accuracy of models. We shows 3 kinds of metrics. The first is the geral metrics per layer, it measures the error between real and predicted hits. The second are axes metrics that measures coordinate erros per layer and the final is an score of quantity of correct hits by layers and quantity of tracks reconstructed. For example, to see the accuracy of training algorithm, go to `results/encrypt_name/results-train.txt` file and accuracy of testing and tracks reconstructed using that algorithm is in `results/encrypt_name/results-test.txt` file. 

```
[Output] Results 
---Parameters--- 
         Model Name    :  lstm
         Dataset       :  phi025-025_eta025-025_train1_lasthit_20200219.csv
         Tracks        :  528
         Model saved   :  /compiled/model-lstm-DCtuvkiXn32hugVsTaokcp-coord-xyz-normalise-true-epochs-21-batch-6.h5
         Test date     :  10/06/2020 12:09:34
         Coordenates   :  xyz
         Model Scaled   :  True
         Model Optimizer :  adam
         Prediction Opt  :  nearest
         Total correct hits per layer  [256. 251. 213. 194. 157. 126.] of 528 tracks tolerance=0.0: 
         Total porcentage correct hits : ['48.48%', '47.54%', '40.34%', '36.74%', '29.73%', '23.86%']
         Reconstructed tracks: 74 of 528 tracks

```
Above output shows the scoring per layer for example 48% with 256 hits matched at first layer, the results are 74 tracks reconstructed of 528 tracks. We also write other info like what kind of coordinate, if we use nearest optimization, how many epochs, batch , model name etc.

The geral accuracy per layer are:

```
---Regression Scores--- 
        R_2 statistics        (R2)  = 0.992
        Mean Square Error     (MSE) = 882.525
        Root Mean Square Error(RMSE) = 29.707
        Mean Absolute Error   (MAE) = 9.858

layer  5
---Regression Scores--- 
        R_2 statistics        (R2)  = 1.0
        Mean Square Error     (MSE) = 6.818
        Root Mean Square Error(RMSE) = 2.611
        Mean Absolute Error   (MAE) = 1.325

layer  6
---Regression Scores--- 
        R_2 statistics        (R2)  = 0.999
        Mean Square Error     (MSE) = 27.603
        Root Mean Square Error(RMSE) = 5.254
        Mean Absolute Error   (MAE) = 2.541

layer  7
---Regression Scores--- 
        R_2 statistics        (R2)  = 0.998
        Mean Square Error     (MSE) = 141.074
        Root Mean Square Error(RMSE) = 11.877
        Mean Absolute Error   (MAE) = 5.285
```

## Vizualization
Open the plot_prediction.ipynb file at notebooks directory to see the plots for results.

The next plot shows all hits.
<p align="center">
         <img width="600" height="450" src="./imgs/all_hits_per_layer.png"></img>
</p>

The next plot is the prediction of all hits.
<p align="center">
         <img width="600" height="450" src="./imgs/all_tracks_pred.png"></img>
</p>

This plot is 10 tracks reconstructed.
<p align="center">
         <img width="600" height="450" src="./imgs/internal_reconstruction_cartesian_zy.png"></img>
</p>
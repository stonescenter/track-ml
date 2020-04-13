# Track Particle
This project is the first version of Track Particles problem and it is part of [SPRACE](https://sprace.org.br/) sponsored by [Serrapilheira](https://serrapilheira.org/). 
<p align="center">
	 <img width="450" height="400" src="./prediction-img.png"></img>
 </p>

## Run
To run:
1. Clone our repository
2. Configure your conda envirotment with env.yml file
3. There are many scripts to test the problem with diferents models(MLP, CNN, LSTM, RNN and others). For example: ` python main_train.py --config config_lstm.json `. It will create a model with a LSTM architecture. You can inference the data test with ` python_inference.py --config config_lstm.json `.

## Accuracy of Algorithm
We are using regressions metrics for tracking reconstruction. Our accuracy is showed for forecast of 10 hit with . 
m
```
[Output] ---Regression Scores--- 
	R_2 statistics        (R2)  = 0.81
	Root Mean Square Error(RMSE) = 0.17
	Mean Absolute Error   (MAE) = 0.085

RMSE:		[0.170] 
RMSE features: 	[0.15, 0.23, 0.09] 
R^2  features:	[0.97, 0.94, 0.99] 
```
the last two metrics are vectors of the RMSE and R^2 average.

## Vizualization
Open the plot_prediction.ipynb file to see the N hits predicted.
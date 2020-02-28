# Track Particle
This project is the first version of Track Particles problem and it is part of [SPRACE](https://sprace.org.br/) sponsored by [Serrapilheira](https://serrapilheira.org/). 



## Run
To run:
1. Clone our repository
2. Configure your conda envirotment 
3. Run ` python main.py --config config.json `. It will create  a first model with LSTM architecture. If you don't have acces to a gpu, you could use the `test_1_lstm_paralel.ipynb` file to run into Google Colab.

## Accuracy of Algorithm
We are using regressions metrics for each hit. Our accuracy is showed for forecas of 5 hit. 

```
[Output] ---Regression Scores--- 
	R_2 statistics        (R2)  = 0.97
	Root Mean Square Error(RMSE) = 0.17
	Mean Absolute Error   (MAE) = 0.085

RMSE:		[0.170] 
RMSE features: 	[0.15, 0.23, 0.09] 
R^2  features:	[0.97, 0.94, 0.99] 
```

## Vizualization
Open the plot_prediction.ipynb file to see the 5 hit predicted.
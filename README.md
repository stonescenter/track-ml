# Track Particle
This project investigate Machine Learning techniques to Particle Track reconstruction problems to HEP, it is part of [SPRACE](https://sprace.org.br/) sponsored by [Serrapilheira](https://serrapilheira.org/). This is a work flow of proposal.

![](imgs/work_flow.png)

We use different Machine Learning tecnhiques to resolve this big problem in the physics community. If you want to reproduce our results, we have written some general steps. You are welcome, if you have some ideas our suggestations, please let us know.

# Setup
## Environment
You need to install [miniconda](https://docs.conda.io/en/latest/miniconda.html) before on a linux system
1. Configure your conda environtment with env.yml file.
```sh
$ conda env create -f env.yml
$ conda activate trackml
```

## Intallation
To run:
1. Clone the repository
```sh
$ git clone https://github.com/SPRACE/track-ml.git
```
2. go to `track-ml` directory created

you will need to have a GPU or some descent CPU. 


# Dataset
We transformed the detector into three kinematical regions to train our models with different datasets. 
- The first region is formed by the internal barrel with `$\eta$` coordinate from `($-1.0$, $1.0$)`. 
- The second region is the intermediary barrel, (overlap) with `$\eta$` coordinate from `($-2.0$ to $-1.0$)` or `($2.0$ to $1.0$)`. 
- The last region is external with `$\eta$` values between `($-3.0$ to $-2.0$)` or `($3.0$ to $2.0$)`.

Considering the mentioned regions and the symmetry of the detector, each dataset was filtered to contain only high energy particles `pT > 1.0 GeV` with `$\phi$` values between `($-0.5$, $0.5$)`, in order to obtain tracks with larger curvature radius, facilitating initial training. 

<p align="center">
        <img width="600" height="450" src="./imgs/cartesian_last.png"></img>
        <img width="600" height="450" src="./imgs/polar_last.png"></img>
</p>

Short datasets are in `dataset`  directory. We split a region in three parts (this is too big to be here).  

# Running
## Training
There are some predefined config files to train diferents models (MLP, CNN, LSTM, CNN-parallel and others). If you need to change the parameters then change the config_*.json file. We used internal barrel as dataset, this dataset is previously transformed and linked in json file:
```sh
$ python main_train.py --config configs/config_lstm_parallel_internal.json
```
There are other configurations for example a CNN model:

```sh
$ python main_train.py --config configs/config_cnn_parallel_internal.json
```

If you want to use a gaussian model 

```sh
$ python main_train.py --config configs/config_lstm_gaussian.json
```

If you want to see the training process when ajust any parameters of .json file. Run the notebook:

```sh
$ main_train.ipynb
```

## Inference

You can inference the previous compiled model with the same config file:
```sh
$ python main_inference.py --config configs/config_lstm_parallel_internal.json --load True 
```
This will produce an report at `results/encrypt_name/results-test.txt` file. With the number of reconstructed tracks and quantity of hits correctly predicted by layer.

If you want to inference with a short dataset:
```sh
$ python main_inference.py --config configs/config_lstm_parallel_internal.json --load True --samples 500
```
or 
```sh
$ python main_inference.py --config configs/config_lstm_gaussian.json --load True
```
## Auxiliary Scripts

You can automatizate trainings and testings with scripts. there are some default configurations:

```sh
$ ./run_trains.sh
$ ./run_tests.sh
```

# Performance
## Accuracy of Algorithm

We are using regressions metrics for accuracy of models. We show 2 groups of metrics.

- The principal metrics is a scoring. Scoring counts how many correct hits were found per layer and comparates with original truth hits. Finally we count the quantity of tracks reconstructed.

- The other metrics are regression metrics, we measure the error between real and predicted hits per layer. 

For example, to see the accuracy of training algorithm, go to `results/encrypt_name/results-train.txt` file and the scoring of correct and tracks reconstructed go to `results/encrypt_name/results-test.txt` file. 

Output test file:
```
[Output] Results 
---Parameters--- 
         Model Name    :  gaussian-lstm
         Dataset       :  eta_n0.5-0.5_phi_n0.5-0.5_20200518171238_tracks.csv
         Tracks        :  3000
         Model saved   :  compiled/model-gaussian-lstm-MEnGHa5DCmJGE9C9BpAg4i-coord-cylin-normalise-true-epochs-30-batch-30.h5
         Test date     :  02/10/2020 20:23:55
         Coordenates   :  cylin
         Coordenate 3D :  True
         Model Scaled   :  True
         Model Optimizer :  adam
         Prediction Opt  :  gaussian
         Distance metric :  ['polar', 'euclidean', 'mahalanobis', 'cosine']
         Remove hit      :  False
         Correct hits per layer (polar) ['191.0(6.37%)', '135.0(4.5%)', '182.0(6.07%)', '167.0(5.57%)', '121.0(4.03%)', '127.0(4.23%)'] of 3000 tracks: 
         Correct hits per layer (euclidean) ['61.0(2.03%)', '54.0(1.8%)', '64.0(2.13%)', '62.0(2.07%)', '49.0(1.63%)', '36.0(1.2%)'] of 3000 tracks: 
         Correct hits per layer (mahalanobis) ['191.0(6.37%)', '193.0(6.43%)', '178.0(5.93%)', '193.0(6.43%)', '165.0(5.5%)', '133.0(4.43%)'] of 3000 tracks: 
         Correct hits per layer (cosine) ['2323.0(77.43%)', '1980.0(66.0%)', '1792.0(59.73%)', '1702.0(56.73%)', '1487.0(49.57%)', '1351.0(45.03%)'] of 3000 tracks: 
         Reconstructed tracks: 887 of 3000 tracks (29.566666666666666)


```
Above output shows scoring per layer for example 48% with 256 hits were matched at the first layer, results are 74 tracks reconstructed of 528 tracks(it is a short dataset just). We also write other info like what kind of coordinate, if we use the nearest optimization, epochs, batchs, optimazer used, model name etc.

Regression metrics per layer are:

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
The last output shows one geral metric for all hits and four (R^2, MSE, RMSE, MAE) metrics per layer.


## Plots
If you want to see the results with plots, go to the plot_prediction.ipynb file at `notebooks` directory. This example plots cartesian coordinates. 

<table><tr>
<td align="center"> <img src="./imgs/internal_reconstruction_cartesian_zy.png" alt="Drawing" style="width: 250px;"/> </td>
</tr></table>

The next plot shows all hits.
<table><tr>
<td align="center"> <img src="./imgs/internal_reconstruction_3d.png" alt="Drawing" style="width: 250px;"/> </td>
</tr></table>

### Plot Distributions
There is other notebook for plot the gaussian distributions plot_distributions.ipynb

<table><tr>
<td align="center"> <img src="./imgs/plot_polar.png" alt="Drawing" style="width: 250px;"/> </td>
</tr></table>

The pull of distributions for $etha$ and $phi$.

<table><tr>
<td> <img src="./imgs/plot_eta.png" alt="Drawing" style="width: 250px;"/> </td>
<td> <img src="./imgs/plot_phi.png" alt="Drawing" style="width: 250px;"/> </td>
</tr></table>



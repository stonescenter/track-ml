__author__ = "Steve Ataucuri"
__copyright__ = "Sprace.org.br"
__version__ = "1.0.0"

import sys
import os
import argparse
import json

from sklearn.model_selection import train_test_split

from core.data.data_loader import *
from core.models.lstm import ModelLSTM, ModelLSTMParalel, ModelLSTMCuDnnParalel
from core.models.cnn import ModelCNN
from core.models.mlp import ModelMLP
from core.models.rnn import ModelRNN

from core.utils.metrics import *
from core.utils.utils import *

import numpy as np

def parse_args():
    """Parse arguments."""
    # Parameters settings
    parser = argparse.ArgumentParser(description="LSTM implementation ")

    # Dataset setting
    parser.add_argument('--config', type=str, default="config.json", help='Configuration file')
    parser.add_argument('--dataset', type=str, help='Path to dataset')
    parser.add_argument('--cylindrical', type=str, help='Type of Coordenates system')
    # parse the arguments
    args = parser.parse_args()

    return args

def manage_models(config):
    
    type_model = config['model']['name']
    model = None

    if type_model == 'lstm': #simple LSTM
        model = ModelLSTM(config)
    elif type_model == 'lstm-paralel':
        model = ModelLSTMParalel(config)
    elif type_model == 'cnn':
        model = ModelCNN(config)
    elif type_model == 'mlp':
        model = ModelMLP(config)
    elif type_model == 'rnn':
        model = ModelRNN(config)        

    return model

def main():

    args = parse_args()       

    # load configurations of model and others
    configs = json.load(open(args.config, 'r'))

    # create defaults dirs
    output_bin = configs['paths']['bin_dir']
    output_path = configs['paths']['save_dir']
    output_logs = configs['paths']['log_dir']
    data_file = configs['data']['filename']

    if os.path.isdir(output_bin) == False:
        os.mkdir(output_bin)

    if os.path.isdir(output_path) == False:
        os.mkdir(output_path)

    if os.path.isdir(output_logs) == False:
        os.mkdir(output_logs)        
    
    time_steps =  configs['model']['layers'][0]['input_timesteps']  # the number of points or hits
    num_features = configs['model']['layers'][0]['input_features']  # the number of features of each hits

    split = configs['data']['train_split']  # the number of features of each hits
    cylindrical = configs['data']['cylindrical']  # set to polar or cartesian coordenates
    normalise = configs['data']['normalise'] 
    num_hits = configs['data']['num_hits']


    if args.dataset is not None:
        data_file = args.dataset
        configs['data']['filename'] = data_file     
    if args.cylindrical is not None:
        cylindrical = True if args.cylindrical == "True" else False
        configs['data']['cylindrical'] = cylindrical    

    # prepare data set
    data = Dataset(data_file, split, cylindrical, num_hits, KindNormalization.Zscore)

    X_train, y_train = data.get_training_data(n_hit_in=time_steps, n_hit_out=1,
                                 n_features=num_features, normalise=normalise)

    print('[Data] shape supervised: X%s y%s :' % (X_train.shape, y_train.shape))
    
    X_train = data.reshape3d(X_train, time_steps, num_features)

    print('[Data] shape data X_train.shape:', X_train.shape)
    print('[Data] shape data y_train.shape:', y_train.shape)

    model = manage_models(configs)

    if model is None:
        print('Please instance model')
        return

    loadModel = configs['training']['load_model']
    show_metrics = configs['training']['show_metrics']
    report = ""

    if loadModel == False:
        # if exist, please used the compiled model!
        if model.exist_model(model.save_fnameh5):
            print("[Warning] Please there is a previous model compiled (%s) for %s file." 
                % (model.save_fnameh5,data_file))
            return

        model.build_model()

        # in-memory training
        history = model.train(
            x=X_train,
            y=y_train,
            epochs = configs['training']['epochs'],
            batch_size = configs['training']['batch_size']
        )
        #if show_metrics:
        report = evaluate_training(history, output_path)

    elif loadModel == True:       
        if not model.load_model():
            print ('[Error] please change the config file : load_model')
            return
    
    if cylindrical:
        coord = 'cylin'
    else:
        coord = 'xyz'

    # save results in a file    
    orig_stdout = sys.stdout
    f = open('results/results-train.txt', 'a')
    sys.stdout = f        

    print("[Output] Train results ")
    print("---Parameters--- ")
    print("\t Model Name        : ", model.name)
    print("\t Dataset           : ", model.orig_ds_name)
    print("\t Total tracks      : ", len(X_train))
    print("\t Path saved        : ", model.save_fnameh5) 
    print("\t Coordenate type   : ", coord) 
    print("\t Model scaled      : ", model.normalise)
    print("\t Accuracy          : ", report) 
    
    sys.stdout = orig_stdout
    f.close()    
    
    print('[Output] All results saved at %s directory and results.txt file. Please use notebooks/plot_prediction.ipynb' % output_path)    


if __name__=='__main__':
    main()
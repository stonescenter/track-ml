__author__ = "Steve Ataucuri"
__copyright__ = "Sprace.org.br"
__version__ = "1.0.0"

import sys
import os
import argparse
import json

from sklearn.model_selection import train_test_split
import datetime as dt

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
    parser.add_argument('--load', type=str, help='this param load model')
        
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
    time_steps =  configs['model']['layers'][0]['input_timesteps']  # the number of points or hits
    num_features = configs['model']['layers'][0]['input_features']  # the number of features of each hits

    split = configs['data']['train_split']  # the number of features of each hits
    cylindrical = configs['data']['cylindrical']  # set to polar or cartesian coordenates
    normalise = configs['data']['normalise'] 
    num_hits = configs['data']['num_hits']
    model_name = configs['model']['name']
    optim = configs['model']['optimizer']
    neurons = configs['model']['layers'][0]['neurons']
    loadModel = configs['training']['load_model']
    validation_split = configs['training']['validation']
    epochs = configs['training']['epochs']
    batch = configs['training']['batch_size']

    if args.dataset is not None:
        data_file = args.dataset
        configs['data']['filename'] = data_file     
    if args.cylindrical is not None:
        cylindrical = True if args.cylindrical == "True" else False
        configs['data']['cylindrical'] = cylindrical    
    if args.load is not None:
        loadModel = True if args.load == "True" else False
        configs['training']['load_model'] = loadModel 

    #create a encryp name for dataset
    path_to, filename = os.path.split(data_file)

    orig_ds_name = filename

    encryp_ds_name = get_unique_name(orig_ds_name)
    decryp_ds_name = get_decryp_name(encryp_ds_name)

    output_encry = os.path.join(output_path, encryp_ds_name)  
    if os.path.isdir(output_bin) == False:
        os.mkdir(output_bin)

    if os.path.isdir(output_path) == False: 
        os.mkdir(output_path)
         
    if os.path.isdir(output_encry) == False: 
        os.mkdir(output_encry)

    if os.path.isdir(output_logs) == False:
        os.mkdir(output_logs)        
    
    # prepare data set
    data = Dataset(data_file, split, cylindrical, num_hits, KindNormalization.Zscore)

    X_train, y_train = data.get_training_data(n_hit_in=time_steps, n_hit_out=1,
                                 n_features=num_features, normalise=normalise)

    if normalise:
        data.save_scale_param(output_encry)

    print('[Data] shape supervised: X%s y%s :' % (X_train.shape, y_train.shape))
    
    X_train = data.reshape3d(X_train, time_steps, num_features)

    print('[Data] shape data X_train.shape:', X_train.shape)
    print('[Data] shape data y_train.shape:', y_train.shape)

    model = manage_models(configs)

    if model is None:
        print('Please instance model')
        return


    show_metrics = configs['training']['show_metrics']
    report = ""
    
    if cylindrical:
        coord = 'cylin'
    else:
        coord = 'xyz'

    ident_name = model.name + "_" + coord 
        
    if loadModel == False:
        # if exist, please used the compiled model!
        if model.exist_model(model.save_fnameh5):
            print("[Warning] Please there is a previous model compiled (%s) for %s file." 
                % (model.save_fnameh5,data_file))
            return

        model.build_model()
        save_fname = os.path.join(output_encry, 'architecture_%s.png' % ident_name)
        model.save_architecture(save_fname) 
        # in-memory training
        history = model.train(
            x=X_train,
            y=y_train,
            validation = validation_split,
            epochs = epochs,
            batch_size = batch
        )
        #if show_metrics:
        report = evaluate_training(history, output_encry, ident_name)

    elif loadModel == True:       
        if not model.load_model():
            print ('[Error] please change the config file : load_model')
            return
    
    # save results in a file    
    orig_stdout = sys.stdout
    f = open(os.path.join(output_encry, 'results-train.txt'), 'a')
    sys.stdout = f        

    now = dt.datetime.now()
    print("[Output] Train results ")
    print("---Parameters--- ")
    print("\t Model Name        : ", model.name)
    print("\t Dataset           : ", model.orig_ds_name)
    print("\t Total tracks      : ", len(X_train))
    print("\t Path saved        : ", model.save_fnameh5) 
    print("\t Coordenate type   : ", coord) 
    print("\t Compiled date     : ", now.strftime("%d/%m/%Y %H:%M:%S"))    
    print("\t Model scaled      : ", model.normalise)
    print("\t Model Optimizer   : ", optim)
    print("\t Model batch_size  : ", batch)
    print("\t Model epochs      : ", epochs)
    print("\t Accuracy          : ", report) 
    
    sys.stdout = orig_stdout
    f.close()    
    
    print('[Output] All results saved at %s directory it results-train.txt file. Please use notebooks/plot_prediction.ipynb' % output_encry)    


if __name__=='__main__':
    main()
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

        evaluate_training(history, output_path)

    elif loadModel == True:       
        if not model.load_model():
            print ('[Error] please change the config file : load_model')
            return

    # prepare data set
    data = Dataset(data_file, split, cylindrical, num_hits, KindNormalization.Zscore)

    X_test, y_test = data.get_testing_data(n_hit_in=time_steps, n_hit_out=1,
                                 n_features=num_features, normalise=normalise)

    print('[Data] shape data X_test.shape:', X_test.shape)
    print('[Data] shape data y_test.shape:', y_test.shape)

    # convertimos a matriz do test em um vetor
    X_test_ = data.reshape3d(X_test, time_steps, num_features) 
    #y_test_ = convert_matrix_to_vec(y_test, num_features)
    #y_test_ = np.array(y_test_)

    print('[Data] Predicting dataset with input ...', X_test_.shape)
    
    seq_len = num_hits - time_steps
    pred_full_res, correct = model.predict_full_sequences_nearest(X_test_, y_test, seq_len)

    predicted_nearest = convert_vector_to_matrix(pred_full_res, num_features, seq_len)
    predicted_nearest = to_frame(predicted_nearest)
    
    # we need to transform to original data
    if normalise:
        y_test_orig = data.inverse_transform_test_y(y_test)
        y_predicted_orig = data.inverse_transform_test_y(predicted_nearest)
    else:
        y_test_orig = y_test
        y_predicted_orig = predicted_nearest

    if cylindrical:
        coord = 'cylin'
    else:
        coord = 'xyz'

    # save results in a file    
    orig_stdout = sys.stdout
    f = open('results/results.txt', 'a')
    sys.stdout = f        

    print("[Output] Results ")
    print("---Parameters--- ")
    print("\t Model Name    : ", model.name)
    print("\t Dataset       : ", model.orig_ds_name)
    print("\t Tracks        : ", len(X_test))
    print("\t Model saved   : ", model.save_fnameh5) 
    print("\t Coordenates   : ", coord) 
    print("\t Model stand   : ", model.normalise) 
    print("\t Total correct : ", correct)
    print("\t Total porcentage correct :", (correct*100)/len(X_test))

    y_test_orig = pd.DataFrame(y_test_orig)
    y_predicted_orig = pd.DataFrame(y_predicted_orig)

    # calculing scores
    result = calc_score(data.reshape2d(y_test_orig, 1),
                        data.reshape2d(y_predicted_orig, 1), report=False)

    r2, rmse, rmses = evaluate_forecast_seq(y_test_orig, y_predicted_orig)
    summarize_scores(r2, rmse, rmses)

    sys.stdout = orig_stdout
    f.close()    

    print('[Data] shape y_test_orig ', y_test_orig.shape)
    print('[Data] shape y_predicted_orig ', y_predicted_orig.shape)

    # call this function againt with normalise False
    x_true, y_true = data.get_testing_data(n_hit_in=time_steps, n_hit_out=1,
                                     n_features=num_features, normalise=False)

    if cylindrical:

        y_test_orig.to_csv(os.path.join(output_path, 'y_true_%s_cylin.csv' % configs['model']['name']),
                    header=False, index=False)
        y_predicted_orig.to_csv(os.path.join(output_path, 'y_pred_%s_cylin.csv' % configs['model']['name']),
                    header=False, index=False)
        x_true.to_csv(os.path.join(output_path, 'x_true_%s_cylin.csv' % configs['model']['name']),
                    header=False, index=False)
    else:

        y_test_orig.to_csv(os.path.join(output_path, 'y_true_%s_xyz.csv' % configs['model']['name']),
                    header=False, index=False)
        y_predicted_orig.to_csv(os.path.join(output_path, 'y_pred_%s_xyz.csv' % configs['model']['name']),
                    header=False, index=False)
        x_true.to_csv(os.path.join(output_path, 'x_true_%s_xyz.csv' % configs['model']['name']),
                    header=False, index=False)

    print('[Output] All results saved at %s directory and results.txt file. Please use notebooks/plot_prediction.ipynb' % output_path)    


if __name__=='__main__':
    main()
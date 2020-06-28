__author__ = "Steve Ataucuri"
__copyright__ = "Sprace.org.br"
__version__ = "1.0.0"

import sys
import os
import argparse
import json

import datetime as dt
from core.data.data_loader import *
from core.models.lstm import ModelLSTM, ModelLSTMParallel, ModelLSTMCuDnnParalel
from core.models.cnn import ModelCNN, ModelCNNParallel
from core.models.mlp import ModelMLP
from core.models.rnn import ModelRNN
from core.models.base import BagOfHits

from core.utils.metrics import *
from core.utils.utils import *

import numpy as np

def parse_args():
    """Parse arguments."""
    # Parameters settings
    parser = argparse.ArgumentParser(description="LSTM implementation ")

    # Dataset setting
    parser.add_argument('--config', type=str, default="default_config.json", help='Configuration file')
    parser.add_argument('--dataset', type=str, help='Path to dataset')
    parser.add_argument('--cylindrical', type=str, help='Type of Coordenates system')
    parser.add_argument('--load', type=str, help='this param load model')
    parser.add_argument('--normalise', type=str, help='normalise input data')
    parser.add_argument('--typeopt', type=str, help='type of optimization of predicted value')
    
    # parse the arguments
    args = parser.parse_args()

    return args

def manage_models(config):
    
    type_model = config['model']['name']
    model = None

    if type_model == 'lstm': #simple LSTM
        model = ModelLSTM(config)
    elif type_model == 'lstm-parallel':
        model = ModelLSTMParallel(config)
    elif type_model == 'cnn':
        model = ModelCNN(config)
    elif type_model == 'cnn-parallel':
        model = ModelCNNParallel(config)        
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
    output_path = configs['paths']['save_dir']
    output_logs = configs['paths']['log_dir']
    data_file = configs['data']['filename']

    time_steps =  configs['model']['layers'][0]['input_timesteps']  # the number of points or hits
    t_features = configs['model']['layers'][0]['input_features']  # the number of features of a tensor
    n_features = configs['data']['features']  # the number of features of data input
    optim = configs['model']['optimizer']
    type_model = configs['model']['name']    
    is_parallel = configs['model']['isparallel']
    
    split = configs['data']['train_split']  # the number of features of each hits
    cylindrical = configs['data']['cylindrical']  # set to polar or cartesian coordenates
    normalise = configs['data']['normalise'] 
    num_hits = configs['data']['num_hits']
    type_opt = configs['testing']['type_optimization']
    tolerance = configs['testing']['tolerance']
    metric = configs['testing']['metric']

    loadModel = configs['training']['load_model']


    # we set preference to params by bash commands
    if args.dataset is not None:
        data_file = args.dataset
        configs['data']['filename'] = data_file     
    if args.cylindrical is not None:
        cylindrical = True if args.cylindrical == "True" else False
        configs['data']['cylindrical'] = cylindrical    
    if args.load is not None:
        loadModel = True if args.load == "True" else False
        configs['training']['load_model'] = loadModel  
    if args.normalise is not None:
        normalise = True if args.normalise == "True" else False
        configs['data']['normalise'] = normalise  
    if args.typeopt is not None:
        type_opt = args.typeopt
        configs['testing']['type_optimization'] = type_opt  
    
    #create a encryp name for dataset
    path_to, filename = os.path.split(data_file)

    orig_ds_name = filename

    encryp_ds_name = get_unique_name(orig_ds_name)
    decryp_ds_name = get_decryp_name(encryp_ds_name)

    output_encry = os.path.join(output_path, encryp_ds_name)  

    if os.path.isdir(output_path) == False: 
        os.mkdir(output_path)
         
    if os.path.isdir(output_encry) == False: 
        os.mkdir(output_encry)

    if os.path.isdir(output_logs) == False:
        os.mkdir(output_logs)        

    model = manage_models(configs)

    if model is None:
        print('Please instance model')
        return
   
    if loadModel:       
        if not model.load_model():
            print('[Error] please change the config file : load_model')
            return
    elif not loadModel:
        print('[Error] this scripts don´t allow train models. Change the load_model parameter to true.')
        return

    # prepare data set
    data = Dataset(data_file, split, cylindrical, num_hits, KindNormalization.Zscore)

    # we need to load a previous distribution of training data. If we have testing stage divided
    # pay attention x_scaler and y_scaler have the same distribution normalized of training stage
    x_scaler, y_scaler = data.load_scale_param(output_encry)

    X_test, y_test = data.get_testing_data(n_hit_in=time_steps, n_hit_out=1,
                                       n_features=n_features, normalise=False,
                                       xscaler=x_scaler, yscaler=y_scaler)

    # a short dataset
    #X_test = X_test.iloc[0:1000,]
    #y_test = y_test[0:1000]

    print('[Data] shape data X_test.shape:', X_test.shape)
    print('[Data] shape data y_test.shape:', y_test.shape)


    if type_model == 'lstm' or type_model == 'cnn':
        if not is_parallel:
            # convertimos a matriz do test em um vetor
            X_test_ = data.reshape3d(X_test, time_steps, n_features)
            y_test_ = data.reshape3d(y_test, 6, n_features)
    elif type_model == 'lstm-parallel' or type_model == 'cnn-parallel':
        X_test_ = data.reshape3d(X_test, time_steps, n_features)
        y_test_ = y_test

    print('[Data] Predicting dataset with input ...', X_test_.shape)
    
    seq_len = num_hits - time_steps
    #pred_full_res = model.predict_full_sequences_nearest(X_test_, y_test_, seq_len)
    #pred_full_res, correct = model.predict_full_sequences_nearest(X_test_, y_test, seq_len)

    correct = [0]
    correct_nearest = [0]
    y_pred = None
    if cylindrical:
        if type_opt == "normal":
            y_pred = model.predict_full_sequences(X_test_, data, num_hits=6, normalise=normalise)
        elif type_opt == "nearest":
            # get data in coord cartesian
            data_tmp = Dataset(data_file, split, False, num_hits, KindNormalization.Zscore)

            # for cylindrical True always we need the data as original values with normalise False
            X_test_aux, y_test_aux = data_tmp.get_testing_data(n_hit_in=time_steps, n_hit_out=1,
                                             n_features=n_features, normalise=False)
            if not is_parallel:       
                y_pred, correct_nearest, correct = model.predict_full_sequences_nearest(X_test_, y_test, data, BagOfHits.Layer, y_test_aux, seq_len, 
                                                                     normalise=normalise, cylindrical=True,
                                                                     verbose=False, tol=tolerance)
            else:
                y_pred, correct_nearest, correct = model.predict_full_sequences_nearest_parallel(X_test_, y_test, data, BagOfHits.Layer, y_test_aux,
                                                                t_steps=time_steps, t_features=t_features, n_features=n_features, num_hits=seq_len,  
                                                                normalise=normalise, cylindrical=True, verbose=False, tol=tolerance, metric=metric)
    else:
        if type_opt == "normal":
            y_pred = model.predict_full_sequences(X_test_, data, num_hits=6, normalise=normalise)
        elif type_opt == "nearest":
            if not is_parallel:          
                y_pred, correct_nearest, correct = model.predict_full_sequences_nearest(X_test_, y_test, data, BagOfHits.Layer, None, seq_len, 
                                                                 normalise=normalise, cylindrical=False,
                                                                 verbose=False, tol=tolerance)
            else:
                y_pred, correct_nearest, correct = model.predict_full_sequences_nearest_parallel(X_test_, y_test, data, BagOfHits.Layer, None,
                                                                t_steps=time_steps, t_features=t_features, n_features=n_features, num_hits=seq_len,  
                                                                normalise=normalise, cylindrical=False, verbose=False, tol=tolerance, metric=metric)
        else:
            print('no algorithm defined to predict')

    y_predicted = convert_vector_to_matrix(y_pred, n_features, seq_len)
    y_predicted = to_frame(y_predicted)

    print('[Data] shape y_test ', y_test.shape)
    print('[Data] shape y_predicted ', y_predicted.shape)   
    
    # we need to transform to original data
    # no more supported
    '''
    if normalise:
        y_test_orig = data.inverse_transform_test_y(y_test)
        y_predicted_orig = data.inverse_transform_test_y(predicted_nearest)
    else:
        y_test_orig = y_test
        y_predicted_orig = predicted_nearest
    '''

    if cylindrical:
        coord = 'cylin'
    else:
        coord = 'xyz'

    ident_name = model.name + "_" + coord 

    # save correct hits
    correct_new = list(correct_nearest)
    correct_new.append(tolerance)
    save_numpy_values(correct_new, output_encry, 'correct_%s.npy' % ident_name)

    # save results in a file    
    orig_stdout = sys.stdout

    f = open(os.path.join(output_encry, 'results-test.txt'), 'a')
    sys.stdout = f        
    now = dt.datetime.now()

    total_tracks = len(X_test)

    print("[Output] Results ")
    print("---Parameters--- ")
    print("\t Model Name    : ", model.name)
    print("\t Dataset       : ", model.orig_ds_name)
    print("\t Tracks        : ", total_tracks)
    print("\t Model saved   : ", model.save_fnameh5)
    print("\t Test date     : ", now.strftime("%d/%m/%Y %H:%M:%S")) 
    print("\t Coordenates   : ", coord) 
    print("\t Model Scaled   : ", model.normalise)
    print("\t Model Optimizer : ", optim)
    print("\t Prediction Opt  : ", type_opt)
    print("\t Distance metric : ", metric)
    print("\t Correct hits per layer Nearest %s of %s tracks tolerance=%s: " % (correct_nearest, total_tracks, tolerance))
    print("\t Porcentage correct hits :", [str(round((t*100)/total_tracks, 2)) +"%" for t in correct_nearest]) 
    print("\t Correct hits per layer with Normal %s of %s tracks tolerance=%s: " % (correct, total_tracks, tolerance))
    print("\t Porcentage correct hits :", [str(round((t*100)/total_tracks, 2)) +"%" for t in correct])     

    # calculate the number of reconstructed tracks
    true_tracks = np.concatenate([X_test, y_test], axis = 1)
    pred_tracks = np.concatenate([X_test, y_predicted], axis = 1)
    true_tracks = pd.DataFrame(true_tracks)
    pred_tracks = pd.DataFrame(pred_tracks)

    tracks = pd.concat([true_tracks, pred_tracks])
    tracks_ = tracks[tracks.duplicated(keep='first')]
    print('\t Reconstructed tracks: %s of %s tracks (%s)' % (tracks_.shape[0], total_tracks, (tracks_.shape[0]*100)/total_tracks))

    # metrics for nearest
    _,_,_,_,result = calc_score(data.reshape2d(y_test, 1),
                        data.reshape2d(y_predicted, 1), report=True)
    print(result)

    calc_score_layer(y_test, y_predicted, n_features=3)

    mses, rmses, r2s = calc_score_layer_axes(y_test, y_predicted)
    summarize_scores_axes(mses, rmses, r2s)

    sys.stdout = orig_stdout
    f.close()

    # call this function againt with normalise False
    #x_true, y_true = data.get_testing_data(n_hit_in=time_steps, n_hit_out=1,
    #                                 n_features=num_features, normalise=False)

    if cylindrical:

        y_test.to_csv(os.path.join(output_encry, 'y_true_%s_cylin_%s.csv' % (configs['model']['name'], type_opt)),
                    header=False, index=False)
        y_predicted.to_csv(os.path.join(output_encry, 'y_pred_%s_cylin_%s.csv' % (configs['model']['name'], type_opt)),
                    header=False, index=False)
        X_test.to_csv(os.path.join(output_encry, 'x_true_%s_cylin_%s.csv' % (configs['model']['name'], type_opt)),
                    header=False, index=False)
    else:

        y_test.to_csv(os.path.join(output_encry, 'y_true_%s_xyz_%s.csv' % (configs['model']['name'],type_opt)),
                    header=False, index=False)
        y_predicted.to_csv(os.path.join(output_encry, 'y_pred_%s_xyz_%s.csv' % (configs['model']['name'], type_opt)),
                    header=False, index=False)
        X_test.to_csv(os.path.join(output_encry, 'x_true_%s_xyz_%s.csv' % (configs['model']['name'], type_opt)),
                    header=False, index=False)

    print('[Output] All results saved at %s directory at results-test.txt file. Please use notebooks/plot_prediction.ipynb' % output_encry)

if __name__=='__main__':
    main()
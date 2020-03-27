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
    parser = argparse.ArgumentParser(description="MLP Implementation")

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
    elif type_model == 'simplernn':
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
    data = Dataset(data_file, KindNormalization.Zscore)
    
    dataset = data.get_training_data(cylindrical=cylindrical, hits=num_hits)
    #dataset = dataset.iloc[0:2640,0:]
    #dataset = dataset.iloc[0:31600,0:]
    print('[Data] new shape :', dataset.shape)

    print("[Data] Converting to supervised ...")
    X, y = data.convert_to_supervised(dataset.values, n_hit_in=time_steps,
                                n_hit_out=1, n_features=num_features, normalise=normalise)

    print('[Data] shape supervised: X%s y%s :' % (X.shape, y.shape))

    # shuffle is True for default value
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1-split, shuffle=False, random_state=42)
    
    print('[Data] shape data X_train.shape:', X_train.shape)
    print('[Data] shape data X_test.shape:', X_test.shape)
    print('[Data] shape data y_train.shape:', y_train.shape)
    print('[Data] shape data y_test.shape:', y_test.shape)

    model = manage_models(configs)

    if model is None:
        print('Please instance model')
        return

    loadModel = configs['training']['load_model']
    
    if loadModel == False:

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
   
    print('[Data] Predicting dataset with input ...', X_test.shape)
    predicted = model.predict_one_hit(X_test)
    print('[Data] shape predicted output ', predicted.shape)
    print('[Data] shape y_test ', y_test.shape)

    y_predicted = np.reshape(predicted, (predicted.shape[0]*predicted.shape[1], 1))
    y_true_ = data.reshape2d(y_test, 1)
    
    print('[Data] new shape y_test, y_true_ ', y_true_.shape)
    print('[Data] new shape y_predicted ', y_predicted.shape)

    # we need to transform to original data
    if normalise:
        y_test_orig = data.inverse_transform_y(y_test)
        y_predicted_orig = data.inverse_transform_y(predicted)
    else:
        y_test_orig = y_test
        y_predicted_orig = predicted

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
    print("\t Tracks        : ", len(dataset))
    print("\t Model saved   : ", model.save_fnameh5) 
    print("\t Coordenates   : ", coord) 
    print("\t Model stand   : ", model.normalise) 

    # calculing scores
    result = calc_score(y_true_, y_predicted, report=True)
    #r2, rmse, rmses = evaluate_forecast(y_test, predicted)
    r2, rmse, rmses = evaluate_forecast(y_test_orig, y_predicted_orig)  
    summarize_scores(r2, rmse,rmses)

    sys.stdout = orig_stdout
    f.close()

    print('[Data] shape y_test ', y_test.shape)
    print('[Data] shape predicted ', predicted.shape)

    print('[Data] shape y_test_orig ', y_test_orig.shape)
    print('[Data] shape y_predicted_orig ', y_predicted_orig.shape)

    #X = data.get_training_data(cylindrical=False, hit=10)
    X_train, X_test_new = train_test_split(dataset, test_size=1-split, shuffle=False, random_state=42)
    #X_train, X_test = data.train_test_split(dataset, y, train_size=split)

    # 18 fields
    y_predicted_orig = pd.DataFrame(y_predicted_orig)
    y_predicted_orig = data.convert_supervised_to_normal(y_predicted_orig.values, n_hit_in=4, n_hit_out=1, hits=10)
    y_predicted_orig = pd.DataFrame(y_predicted_orig)
    print('y_predicted_orig shape ', y_predicted_orig.shape)

    y_true_orig = pd.DataFrame(y_test_orig)
    y_true_orig = data.convert_supervised_to_normal(y_true_orig.values, n_hit_in=4, n_hit_out=1, hits=10)
    y_true_orig = pd.DataFrame(y_true_orig)
    print('y_true_orig shape ', y_true_orig.shape)

    x_true = X_test_new.iloc[:,0:time_steps*num_features]
    y_true = X_test_new.iloc[:,time_steps*num_features:]

    if cylindrical:

        y_true_orig.to_csv(os.path.join(output_path, 'y_true_%s_cylin.csv' % configs['model']['name']),
                    header=False, index=False)
        y_predicted_orig.to_csv(os.path.join(output_path, 'y_pred_%s_cylin.csv' % configs['model']['name']),
                    header=False, index=False)
        x_true.to_csv(os.path.join(output_path, 'x_true_%s_cylin.csv' % configs['model']['name']),
                    header=False, index=False)
    else:

        y_true_orig.to_csv(os.path.join(output_path, 'y_true_%s_xyz.csv' % configs['model']['name']),
                    header=False, index=False)
        y_predicted_orig.to_csv(os.path.join(output_path, 'y_pred_%s_xyz.csv' % configs['model']['name']),
                    header=False, index=False)
        x_true.to_csv(os.path.join(output_path, 'x_true_%s_xyz.csv' % configs['model']['name']),
                    header=False, index=False)

    print('[Output] Results saved in files %', output_path)

if __name__=='__main__':
    main()

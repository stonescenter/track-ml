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

from core.utils.metrics import *
from core.utils.utils import *

import numpy as np

def parse_args():
    """Parse arguments."""
    # Parameters settings
    parser = argparse.ArgumentParser(description="LSTM implementation ")

    # Dataset setting
    parser.add_argument('--config', type=str, default="config.json", help='Configuration file')

    # parse the arguments
    args = parser.parse_args()

    return args

def gpu():
    import tensorflow as tf
    from tensorflow import set_random_seed

    import keras.backend as K
    from keras.backend.tensorflow_backend import set_session
    
    #configure  gpu_options.allow_growth = True in order to CuDNNLSTM layer work on RTX
    config = tf.ConfigProto(device_count = {'GPU': 0})
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    set_session(sess)

def no_gpu():
    import os
    os.environ["CUDA_VISIBLE_DEVICES"]="-1"
    import tensorflow as tf

    config=tf.ConfigProto(log_device_placement=True)
    sess = tf.Session(config=config)
    set_session(sess)

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

    return model

def main():

    args = parse_args()       

    # load configurations of model and others
    configs = json.load(open(args.config, 'r'))

    # create defaults dirs
    output_path = configs['paths']['save_dir']
    output_logs = configs['paths']['log_dir']
    data_dir = configs['data']['filename']

    if os.path.isdir(output_path) == False:
        os.mkdir(output_path)

    if os.path.isdir(output_logs) == False:
        os.mkdir(output_logs)        

    save_fname = os.path.join(output_path, 'architecture-%s.png' % configs['model']['name'])
    save_fnameh5 = os.path.join(output_path, 'model-%s.h5' % configs['model']['name'])
    
    time_steps =  configs['model']['layers'][0]['input_timesteps']  # the number of points or hits
    num_features = configs['model']['layers'][0]['input_features']  # the number of features of each hits

    split = configs['data']['train_split']  # the number of features of each hits
    cylindrical = configs['data']['cylindrical']  # set to polar or cartesian coordenates
    normalise = configs['data']['normalise'] 

    # config gpu
    #gpu()
      
    # prepare data set
    data = Dataset(data_dir, KindNormalization.Zscore)

    X, y = data.prepare_training_data(FeatureType.Positions, normalise=normalise,
                                                  cylindrical=cylindrical)

    # reshape data     
    X = data.reshape3d(X, time_steps, num_features)
  
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1-split, random_state=42)
    
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
   
    predicted = model.predict_one_hit(X_test)

    y_predicted = np.reshape(predicted, (predicted.shape[0]*predicted.shape[1], 1))
    y_true_ = data.reshape2d(y_test, 1)

    # calculing scores
    result = calc_score(y_true_, y_predicted, report=True)
    r2, rmse, rmses = evaluate_forecast(y_test, predicted)
    summarize_scores(r2, rmse,rmses)


    # print('[Data] shape y_test ', y_test.shape)
    # print('[Data] shape predicted ', predicted.shape)

    # print('[Data] shape y_test ', y_test.shape)
    # print('[Data] shape predicted ', predicted.shape)

    # print('[Output] Finding shortest points ... ')
    # near_points = get_shortest_points(y_test, predicted)
    # y_near_points = pd.DataFrame(near_points)

    # print('[Data] shape predicted ', y_near_points.shape)

    # we need to transform to original data
    y_test_orig = data.inverse_transform(y_test)
    y_predicted_orig = data.inverse_transform(predicted)
    #y_near_orig = data.inverse_transform(y_near_points)

    print(y_test_orig.shape)
    print(y_predicted_orig.shape)

    # print('[Output] Calculating distances ...')

    # dist0 = calculate_distances_matrix(y_predicted_orig, y_test_orig)
    # dist1 = calculate_distances_matrix(y_predicted_orig, y_near_orig)

    # print('[Output] Saving distances ... ')
    # save_fname = os.path.join(save_dir, 'distances.png' )
    # plot_distances(dist0, dist1, save_fname)

    #Save data to plot
    X, y = data.prepare_training_data(FeatureType.Positions, normalise=False,
                                                      cylindrical=cylindrical)
    X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                        test_size=1-split, random_state=42)

    y_pred = pd.DataFrame(y_predicted_orig)
    y_true = pd.DataFrame(y_test_orig)

    if cylindrical:

        y_true.to_csv(os.path.join(output_path, 'y_true_%s_cylin.csv' % configs['model']['name']),
                    header=False, index=False)
        y_pred.to_csv(os.path.join(output_path, 'y_pred_%s_cylin.csv' % configs['model']['name']),
                    header=False, index=False)
        X_test.to_csv(os.path.join(output_path, 'x_test_%s_cylin.csv' % configs['model']['name']),
                    header=False, index=False)
    else:

        y_true.to_csv(os.path.join(output_path, 'y_true_%s_xyz.csv' % configs['model']['name']),
                    header=False, index=False)
        y_pred.to_csv(os.path.join(output_path, 'y_pred_%s_xyz.csv' % configs['model']['name']),
                    header=False, index=False)
        X_test.to_csv(os.path.join(output_path, 'x_test_%s_xyz.csv' % configs['model']['name']),
                    header=False, index=False)

    print('[Output] Results saved at %', output_path)

if __name__=='__main__':
    main()








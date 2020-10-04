__author__ = "Steve Ataucuri"
__copyright__ = "Sprace.org.br"
__version__ = "1.0.0"

import os
import math

import numpy as np
import datetime as dt
from enum import Enum

from scipy.spatial import distance

import tensorflow as tf
import keras.backend as K
#from keras.backend.tensorflow_backend import set_session
from keras.models import Sequential, load_model
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.utils import plot_model   

from core.utils.utils import *
from core.models.gaussian_loss import gaussian_loss, gaussian_nll
from scipy.stats import norm

class BagOfHits(Enum):
    All=1,
    Track=2,
    Layer=3

class BaseModel():
    def __init__(self, configs):
        self.model = Sequential()
        self.name = configs['model']['name']
        self.normalise = configs['data']['normalise']
        self.cylindrical = configs['data']['cylindrical']
        self.epochs = configs['training']['epochs']
        self.batch_size = configs['training']['batch_size']
        self.validation = configs['training']['validation']
        self.earlystopping = configs['training']['earlystopping']
        self.stopped_epoch = 0

        path_to, filename = os.path.split(configs['data']['filename'])
        #print(get_unique_name(filename))
        #self.orig_ds_name = configs['data']['filename']
        self.orig_ds_name = filename

        self.encryp_ds_name = get_unique_name(self.orig_ds_name)
        self.decryp_ds_name = get_decryp_name(self.encryp_ds_name)

        #print(self.encryp_ds_name)

        if self.cylindrical:
            coord = 'cylin'
        else:
            coord = 'xyz'

        # set unique Id identification
        self.save_fnameh5 = os.path.join(configs['paths']['bin_dir'], 
            'model-%s-%s-coord-%s-normalise-%s-epochs-%s-batch-%s.h5' % (
                self.name, self.encryp_ds_name, coord,
                str(self.normalise).lower(), self.epochs, self.batch_size))
        #print(self.save_fnameh5)

        self.save_fname = os.path.join(configs['paths']['save_dir'], 'architecture-%s.png' % self.name)

        self.save = configs['training']['save_model']

        if configs['training']['use_gpu'] == True:
            #if tf.test.is_gpu_available():
            gpus = tf.config.experimental.list_physical_devices('GPU')

            if gpus:
                try:
                    # Currently, memory growth needs to be the same across GPUs
                    for gpu in gpus:
                        tf.config.experimental.set_memory_growth(gpu, True)
                        print('[Model] Set memory growth for %s to True', gpu)

                    logical_gpus = tf.config.experimental.list_logical_devices('GPU')
                    print("[Model] ", len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
                except RuntimeError as e:
                    # Memory growth must be set before GPUs have been initialized
                    print(e)
        else:
            print('No GPU configured.')
            pass

        # if configs['training']['use_gpu'] == True:
        #     #config = tf.ConfigProto( device_count = {'GPU': 0 , 'CPU': 0} ) 
        #     config = tf.ConfigProto()
        #     config.gpu_options.allow_growth = True
        #     sess = tf.Session(config=config) 
        #     set_session(sess)
        #     tf.device('/gpu:0')
        # else:
        #     config=tf.ConfigProto(log_device_placement=True)
        #     sess = tf.Session(config=config)
        #     set_session(sess)
        
        #set_random_seed(42)
        tf.compat.v1.set_random_seed(0)

    def load_model(self):
        if self.exist_model(self.save_fnameh5):
            print('[Model] Loading model from file %s' % self.save_fnameh5)
            self.model = load_model(self.save_fnameh5, custom_objects={'gaussian_loss': gaussian_loss, 'gaussian_nll': gaussian_nll})
            return True
        else:
            print('[Model] Can not load the model from file %s' % self.save_fnameh5)
        return False
    
    def exist_model(self, filepath):
        if os.path.exists(filepath): 
            return True
        return False

    def save_architecture(self, filepath):

        plot_model(self.model, to_file=filepath, show_shapes=True)
        print('[Model] Model Architecture saved at %s' % filepath)

    def save_model(self, filepath):
        self.model.save(filepath)
        print('[Model] Model for inference saved at %s' % filepath)

    def train(self, x, y, epochs, batch_size, validation, shuffle=False, verbose=False, callbacks=None):
        timer = Timer()
        timer.start()
        print('[Model] Training Started')
        print('[Model] %s epochs, %s batch size' % (epochs, batch_size))
        #print('[Model] Shape of data train: ', x.shape) 
        #save_fname = os.path.join(save_dir, '%s-e%s.h5' % (dt.datetime.now().strftime('%d%m%Y-%H%M%S'), str(epochs)))
        
        if callbacks is None:
            print('DEBUG')
            if self.earlystopping:
                callbacks = [
                    EarlyStopping(monitor='loss', mode='min', verbose=1),
                    ModelCheckpoint(filepath=self.save_fnameh5, monitor='val_loss', mode='min', save_best_only=True)
                ]
            else:
                callbacks = [
                    ModelCheckpoint(filepath=self.save_fnameh5, monitor='val_loss', mode='min', save_best_only=True)
                ]
        else:
            pass

        history = self.model.fit(
            x,
            y,
            verbose=verbose,
            validation_split=validation,
            epochs=epochs,
            batch_size=batch_size,
            shuffle=shuffle,
            callbacks=callbacks
        )

        if self.save == True:
            self.save_model(self.save_fnameh5)             

        # what epocks the algorith stopped
        if self.earlystopping:        
            self.stopped_epoch = callbacks[0].stopped_epoch
            
        print('[Model] Model training stopped at %s epoch' % self.stopped_epoch)
        print('[Model] Training Completed. Model h5 saved as %s' % self.save_fnameh5)
        print('[Model] Model train with structure:', self.model.inputs)
        timer.stop()

        return history
    
    def train_generator(self, data_gen, epochs, batch_size, steps_per_epoch, save_dir):
        timer = Timer()
        timer.start()
        print('[Model] Training Started')
        print('[Model] %s epochs, %s batch size, %s batches per epoch' % (epochs, batch_size, steps_per_epoch))
         
        save_fname = os.path.join(save_dir, '%s-e%s.h5' % (dt.datetime.now().strftime('%d%m%Y-%H%M%S'), str(epochs)))
        callbacks = [
            ModelCheckpoint(filepath=save_fname, monitor='loss', save_best_only=True)
        ]
        self.model.fit_generator(
            data_gen,
            steps_per_epoch=steps_per_epoch,
            epochs=epochs,
            callbacks=callbacks,
            workers=1
        )
         
        print('[Model] Training Completed. Model saved as %s' % save_fname)
        timer.stop()

    def evaluate(self, x, y, batch_size=10):
        results = self.model.evaluate(x, y, batch_size=batch_size, verbose=2)
        print('[Model] Test loss %s accuracy %s :' %(results[0], results[1]))
        
    def predict_one_hit(self, data):
        #Predict each timestep given the last sequence of true data, in effect only predicting 1 step ahead each time

        print('[Model] Predicting Hit-by-Hit...')
        predicted = self.model.predict(data)
        print('[Model] Predicted shape predicted%s size %s' % (predicted.shape, predicted.size))

        #predicted = np.reshape(predicted, (predicted.size, 1))
        return predicted
 
    def predict_sequences_multiple(self, data, window_size, prediction_len):
        #Predict sequence of 50 steps before shifting prediction run forward by 50 steps
        print('[Model] Predicting Sequences Multiple...')
        prediction_seqs = []
        for i in range(int(len(data)/prediction_len)):
            curr_frame = data[i*prediction_len]
            predicted = []
            for j in range(prediction_len):
                predicted.append(self.model.predict(curr_frame[np.newaxis,:,:])[0,0])
                curr_frame = curr_frame[1:]
                curr_frame = np.insert(curr_frame, [window_size-2], predicted[-1], axis=0)
            prediction_seqs.append(predicted)
        return prediction_seqs
    '''
    def predict_full_sequences(self, x_test, y_true, hits_len):

        timer = Timer()
        timer.start() 
        print('[Model] Predicting Sequences Started')

        total = len(x_test)
        
        correct = 0
        incorrect = 0
        pred_sequences = []
        
        for j in range(total):       
            curr_frame = x_test[j]
            predicted = []
            for i in range(hits_len):            
                pred = self.model.predict(curr_frame[np.newaxis,:,:])
                predicted.append(pred)
                curr_frame = curr_frame[1:]
                # inserta um  valor np.insert(array, index, value, axes)
                curr_frame = np.insert(curr_frame, [3], predicted[-1], axis=0)
                #print(curr_frame, predicted[-1])
            
            pred_sequences.append(predicted)
        
        print('[Model] Prediction Finished.')
        timer.stop()

        return pred_sequences
    '''
    def predict_full_sequences(self, x_test, data, num_hits=6, normalise=False, tol=0.1):
        '''
            x_test: input data
            normalise: say input data must be scaled 
        '''
        timer = Timer()
        timer.start() 
        print('[Model] Predicting Sequences Started')

        total = len(x_test)

        pred_sequences = []

        #count_correct = np.zeros(num_hits)

        for j in range(total):       
            curr_frame = x_test[j]
            predicted = []
            for i in range(num_hits):

                if normalise:
                    curr_frame = data.x_scaler.transform(np.reshape(curr_frame,(1,12)))
                    curr_frame_orig = data.inverse_transform_x(pd.DataFrame(curr_frame).values.flatten())
                    curr_frame_orig = np.reshape(curr_frame_orig, (4,3))
                    curr_frame = np.reshape(curr_frame, (4,3))
                else:
                    curr_frame = curr_frame
                    curr_frame_orig = curr_frame
                                
                pred = self.model.predict(curr_frame[np.newaxis,:,:])
                
                pred = np.reshape(pred, (1, 3))
                if normalise:
                    pred = data.inverse_transform_y(pred)
                else:
                    pred = pred

                                    
                pred = np.reshape(pred, (1, 3))
                
                #if np.isclose(curr_hit, near_pred, atol=0.01).all():
                #    count_correct[i]=+1
                
                predicted.append(pred)

                curr_frame = curr_frame_orig[1:]
                # inserta um  valor np.insert(array, index, value, axes)
                curr_frame = np.insert(curr_frame, [3], predicted[-1], axis=0)
                #print(curr_frame, predicted[-1])

            pred_sequences.append(predicted)

        print('[Model] Prediction Finished.')
        timer.stop()

        return pred_sequences

    def predict_full_sequences_nearest(self, x_test, y_test, data, bag_of_hits, y_test_aux=None, num_hits=6, num_features=3, 
                                        num_obs=4, normalise=False, cylindrical=False, verbose=False, tol=0.01):
        
        '''
            This function shift the window by 1 new prediction each time, re-run predictions on new window
            parameters:
            x_test : X test data normaly not scaled (4 hits)
            y_test : y test data, normaly not scaled (6 hits)
            num_hits : how many hits by y_test
            normalise : it param says the input data must be scaled or not
        
        '''
        
        timer = Timer()
        timer.start()

        print('[Model] Predicting Sequences with Nearest Started')
        print('[DEBUG] num_obs %s , num_features %s' % (num_obs, num_features))

        total = len(x_test)

        pred_sequences = []
        pred_sequences_orig = []
               
        # covert to original values
        #y_true = data.inverse_transform_test_y(y_test)
        #y_true = y_true.round(decimals)
        
        #change the dataset by cartesian coordinates
        y_test_cpy = y_test        
        if cylindrical:
            y_test = y_test_aux
        else:
            y_test = y_test
            
        # bag_of_hits  
        bag_of_hits_all = np.array(convert_matrix_to_vec(y_test, num_features))

        count_correct_nearest = np.zeros(num_hits)
        count_correct = np.zeros(num_hits)
        begin_idx, end_idx = 0, 0
        num_boh = 6
        
        for j in range(total):
            curr_frame = x_test[j]
            # bag_of_hit by track
            curr_track = np.array(y_test.iloc[j,0:]).reshape(num_hits, num_features)
            
            if verbose:
                print('curr_track %s , %s:' % (j , curr_track))

            predicted = []
            predicted_orig = []
            begin = 0
            for i in range(num_hits):
                # bag_of_hit by layer
                end = begin+num_features
                curr_layer = np.array(y_test.iloc[0:,begin:end]).reshape(total, num_features)
                curr_layer_polar = np.array(y_test_cpy.iloc[0:,begin:end]).reshape(total, num_features)
                curr_hit = curr_track[i]
                begin = end
                
                if verbose:
                    # primeira esta em originais
                    print('input:\n', curr_frame)
                
                if normalise:
                    curr_frame = data.x_scaler.transform(np.reshape(curr_frame,(1, num_obs*num_features)))
                    curr_frame_orig = data.inverse_transform_x(pd.DataFrame(curr_frame).values)
                    curr_frame_orig = np.reshape(curr_frame_orig, (num_obs,num_features))
                    curr_frame = np.reshape(curr_frame, (num_obs, num_features))
                else:
                    curr_frame = curr_frame
                    curr_frame_orig = curr_frame
                    
                if verbose:
                    print('input:\n', curr_frame)
                    #print('input orig:\n', curr_frame_orig)
                
                #print('input inv :\n', curr_frame_inv.reshape(4,3))
                #print('newaxis ', curr_frame[np.newaxis,:,:])
                # input must be scaled curr_frame
                y_pred = self.model.predict(curr_frame[np.newaxis,:,:], batch_size=self.batch_size)

                
                y_pred = np.reshape(y_pred, (1, num_features))
                if normalise:
                    y_pred_orig = data.inverse_transform_y(y_pred)
                else:
                    y_pred_orig = y_pred

                y_pred_orig = np.reshape(y_pred_orig, (1, num_features))

                if bag_of_hits == BagOfHits.All:
                    hits = bag_of_hits_all
                elif bag_of_hits == BagOfHits.Track:
                    hits = curr_track
                elif bag_of_hits == BagOfHits.Layer:
                    hits = curr_layer
                
                if cylindrical:
                    rho, eta, phi = y_pred_orig[0][0], y_pred_orig[0][1], y_pred_orig[0][2]
                    #print(rho,eta,phi)
                    x, y, z = convert_rhoetaphi_to_xyz(rho, eta, phi)
                    #print(x,y,z)
                    y_pred_orig[0][0] = x
                    y_pred_orig[0][1] = y
                    y_pred_orig[0][2] = z
                    
                    #print(y_pred_orig)
                        
                dist = distance.cdist(hits, y_pred_orig, 'euclidean')
                idx = np.argmin(dist)
                near_pred = hits[idx]

                # if curr_hit is in cartesian coord, near_pred must be in cartesian coord too
                # very small numbers are differents or equals
                if np.isclose(curr_hit, near_pred, atol=tol).all():
                    count_correct_nearest[i]+=1

                # counting with predicted without nearest hit
                if np.isclose(curr_hit, y_pred_orig, atol=tol).all():
                    count_correct[i]+=1 

                if verbose:
                    print('pred:', y_pred)
                    print('inv pred:', y_pred_orig)
                    print('current:', curr_hit)
                    print('nearest:', near_pred)
                
                if cylindrical:
                    x, y, z = near_pred[0], near_pred[1], near_pred[2]
                    #print(x,y,z)
                    rho, eta, phi = convert_xyz_to_rhoetaphi(x, y, z)
                    #print(rho,eta,phi)
                    near_pred[0] = rho
                    near_pred[1] = eta
                    near_pred[2] = phi

                # we change to the original input 
                if cylindrical:
                    near_pred = curr_layer_polar[idx]

                #curr_track = np.delete(curr_track, idx, 0)
                #near_pred_orig, idx = model.nearest_hit(y_pred_orig, bag_of_hits, silent=True) 
                        
                #near adiciona em valores originaeis
                predicted.append(near_pred)
                
                #predicted_orig.append(near_pred_orig)
                #print('-----\n')
                
                curr_frame = curr_frame_orig[1:]
                curr_frame = np.insert(curr_frame, [3], predicted[-1], axis=0)
                
            pred_sequences.append(predicted)
            #pred_sequences_orig.append(predicted_orig)
            
            if verbose:
                if j == 3: break
                    
        print('[Model] Prediction Finished.')
        timer.stop()

        return pred_sequences, count_correct_nearest, count_correct

    def predict_full_sequences_nearest_parallel(self, x_test, y_test, data, bag_of_hits, y_test_aux=None, 
                                                t_steps=4, t_features=1, n_features=3, num_hits=6,                                          
                                                normalise=False, cylindrical=False, verbose=False, tol=0.01, metric = 'euclidean'):
        
        '''
            This function shift the window by 1 new prediction each time, re-run predictions on new window
            parameters:
            x_test : X test data normaly not scaled (4 hits)
            y_test : y test data, normaly not scaled (6 hits)
            num_hits : how many hits by y_test
            normalise : it param says the input data must be scaled or not
        
        '''
        
        timer = Timer()
        timer.start()

        print('[Model] Predicting Sequences with Nearest Started')

        total = len(x_test)

        pred_sequences = []
        pred_sequences_orig = []
               
        # covert to original values
        #y_true = data.inverse_transform_test_y(y_test)
        #y_true = y_true.round(decimals)
        
        #change the dataset by cartesian coordinates
        y_test_cpy = y_test
        if cylindrical:
            #if cylindrical is true we use cartesian coordinates
            y_test = y_test_aux
        else:
            y_test = y_test
            
        # bag_of_hits  
        bag_of_hits_all = np.array(convert_matrix_to_vec(y_test, n_features))

        count_correct = np.zeros(num_hits)
        count_correct_nearest = np.zeros(num_hits)
        begin_idx, end_idx = 0, 0
        num_boh = 6
        
        for j in range(total):
            # curr_frame is the current input of hits for model
            curr_frame = x_test[j]
            # bag_of_hit by track
            curr_track = np.array(y_test.iloc[j,0:]).reshape(num_hits, n_features)
            
            if verbose:
                print('curr_track %s , %s:' % (j , curr_track))

            predicted = []
            predicted_orig = []
            begin = 0
            for i in range(num_hits):
                # bag_of_hit by layer
                end = begin+n_features

                curr_layer = np.array(y_test.iloc[0:,begin:end]).reshape(total, n_features)
                curr_layer_polar = np.array(y_test_cpy.iloc[0:,begin:end]).reshape(total, n_features)

                curr_hit = curr_track[i]
                begin = end
                
                if verbose:
                    # primeira esta em originais
                    print('input:\n', curr_frame)
                    
                if normalise:
                    curr_frame = data.x_scaler.transform(np.reshape(curr_frame,(1,t_steps*n_features)))
                    curr_frame_orig = data.inverse_transform_x(pd.DataFrame(curr_frame).values)
                    curr_frame_orig = np.reshape(curr_frame_orig, (t_steps, n_features))
                    curr_frame = np.reshape(curr_frame, (t_steps, n_features))
                else:
                    curr_frame = curr_frame
                    curr_frame_orig = curr_frame

                x = curr_frame[:, 0].reshape((1, t_steps, t_features))
                y = curr_frame[:, 1].reshape((1, t_steps, t_features))
                z = curr_frame[:, 2].reshape((1, t_steps, t_features))
                
                if verbose:
                    print('input:\n', curr_frame)
                    #print('input orig:\n', curr_frame_orig)
                
                #y_pred = model.model.predict(curr_frame[np.newaxis,:,:])
                y_pred = self.model.predict([x, y, z])

                #y_pred = np.reshape(y_pred, (1, 3))
                if normalise:
                    y_pred_orig = data.inverse_transform_y(y_pred)
                else:
                    y_pred_orig = y_pred

                y_pred_orig = np.reshape(y_pred_orig, (1, 3))

                if bag_of_hits == BagOfHits.All:
                    hits = bag_of_hits_all
                elif bag_of_hits == BagOfHits.Track:
                    hits = curr_track
                elif bag_of_hits == BagOfHits.Layer:                    
                    hits = curr_layer
                
                # if output of predict is cylindrical, we convert to cartesian temporaly to calculate distance
                if cylindrical:
                    rho, eta, phi = y_pred_orig[0][0], y_pred_orig[0][1], y_pred_orig[0][2]
                    #print(rho,eta,phi)
                    x, y, z = convert_rhoetaphi_to_xyz(rho, eta, phi)
                    #print(x,y,z)
                    y_pred_orig[0][0] = x
                    y_pred_orig[0][1] = y
                    y_pred_orig[0][2] = z
                    
                # the distances is calculate with always with cartesian
                dist = distance.cdist(hits, y_pred_orig, metric)
                idx = np.argmin(dist)
                near_pred = hits[idx]

                # if curr_hit is in cartesian coord, near_pred must be in cartesian coord too
                if np.isclose(curr_hit, near_pred, atol=tol).all():
                    count_correct_nearest[i]+=1
                    
                # counting with predicted without nearest hit
                if np.isclose(curr_hit, y_pred_orig, atol=tol).all():
                    count_correct[i]+=1 
                    
                if verbose:
                    print('pred:', y_pred)
                    print('inv pred:', y_pred_orig)
                    print('current:', curr_hit)
                    print('nearest:', near_pred)
                '''
                if cylindrical:
                    x, y, z = near_pred[0], near_pred[1], near_pred[2]
                    #print(x,y,z)
                    rho, eta, phi = convert_xyz_to_rhoetaphi(x, y, z)
                    #print(rho,eta,phi)
                    near_pred[0] = rho
                    near_pred[1] = eta
                    near_pred[2] = phi
                '''         
                # we change to the original input 
                if cylindrical:
                    near_pred = curr_layer_polar[idx]

                #near adiciona em valores originaeis
                predicted.append(near_pred)
                           
                curr_frame = curr_frame_orig[1:]
                curr_frame = np.insert(curr_frame, [3], predicted[-1], axis=0)
                           
            pred_sequences.append(predicted)
            #pred_sequences_orig.append(predicted_orig)
            
            if verbose:
                if j == 3: break
                    
        print('[Model] Prediction Finished.')
        timer.stop()

        return pred_sequences, count_correct_nearest, count_correct

    def predict_prob(self, x, bs):
        """Make predictions given model and 2d data
        """
        
        ypred = self.model.predict(x, verbose=0, batch_size=bs)
        n_outs = int(ypred.shape[1] / 2)
        mean = ypred[:, 0:n_outs]
        sigma = np.exp(ypred[:, n_outs:])

        return mean, sigma

    def predict_prob_full_sequences_nearest(self, x_test, y_test, data, bag_of_hits, y_test_xyz=None, 
                                       t_steps=4, t_features=1, n_features=3, num_hits=6, 
                                       normalise=False, cylindrical=False, verbose=False, tol=0.01, 
                                       remove_hit=False, points_3d=True, metrics = 'euclidean'):

        '''
            This func predict the mean and variance shift the window by 1 new prediction each time, 
            re-run predictions on new window

            cylindrical : it says the training model was fed with cylindrical data if it is true
            x_test and y_test are original data coordinates and not normalised
            normalise : say if the model was compiled with normalised data
        '''

        timer = Timer()
        timer.start()

        print('[Model] Predicting Sequences with Nearest Started')

        total = len(y_test)

        pred_tracks = []
        pred_sequences_orig = []

        # if cylindrical is true then we change to cartesian coordinates  
        '''
        if cylindrical: 
            y_test = y_test_aux
        else:
            y_test = y_test
        '''
        # bag_of_hits  
        #bag_of_hits_all = np.array(convert_matrix_to_vec(y_test, n_features))

        layers, layers_xyz = [] , []
        begin = 0
        for i in range(num_hits):
            end = begin+n_features
            layer_orig = np.array(y_test.iloc[:,begin:end]).reshape(total, n_features)
            layer_xyz = np.array(y_test_xyz.iloc[:,begin:end]).reshape(total, n_features)
            begin = end
            layers.append(layer_orig)
            layers_xyz.append(layer_xyz)
                          
        corrects_by_layer = np.zeros((len(metrics), num_hits))
        begin_idx, end_idx = 0, 0
        num_boh = 6

        for j in range(total):
            # in original coordinates, if was trained with polar then it must be polar coordinate
            curr_frame = x_test[j]
            # bag_of_hit by track
            curr_track = np.array(y_test.iloc[j,0:]).reshape(num_hits, n_features)

            if verbose:
                print('curr_track %s , %s:' % (j , curr_track))

            predicted = []
            predicted_orig = []
            #begin = 0
            for i in range(num_hits):

                # this current hit is in original coordinates
                curr_hit = curr_track[i]
                
                if verbose:
                    # primeira esta em originais
                    print('input:\n', curr_frame)

                # if the model was normalized, we  need to transform
                if normalise:
                    curr_frame = data.x_scaler.transform(np.reshape(curr_frame,(1, t_steps*n_features)))
                    curr_frame_orig = data.inverse_transform_x(pd.DataFrame(curr_frame).values)
                    curr_frame_orig = np.reshape(curr_frame_orig, (t_steps, n_features))
                    curr_frame = np.reshape(curr_frame, (t_steps, n_features))
                else:
                    curr_frame = curr_frame
                    curr_frame_orig = curr_frame

                if verbose:
                    print('input:\n', curr_frame)
                    #print('input orig:\n', curr_frame_orig)

                #print('input inv :\n', curr_frame_inv.reshape(4,3))
                #print('newaxis ', curr_frame[np.newaxis,:,:])
                # input must be scaled curr_frame
                # the input and output must be in original coordinates that was compiled
                mean, sigma = self.predict_prob(curr_frame[np.newaxis,:,:], bs=1)

                if points_3d:
                    rho_mean, rho_sigma = mean[:,0],  sigma[:,0]
                    eta_mean, eta_sigma = mean[:,1],  sigma[:,1]
                    phi_mean, phi_sigma = mean[:,2],  sigma[:,2]
                    
                    rho_pred = norm.median(rho_mean, rho_sigma)
                    eta_pred = norm.median(eta_mean, eta_sigma)
                    phi_pred = norm.median(phi_mean, phi_sigma)

                    y_pred = np.reshape([rho_pred, eta_pred, phi_pred], (1, n_features))

                else:
                    eta_mean, eta_sigma = mean[:,0],  sigma[:,0]
                    phi_mean, phi_sigma = mean[:,1],  sigma[:,1]
                    
                    eta_pred = norm.median(eta_mean, eta_sigma)
                    phi_pred = norm.median(phi_mean, phi_sigma)

                    y_pred = np.reshape([eta_pred, phi_pred], (1, n_features))

                if normalise:
                    y_pred_orig = data.inverse_transform_y(y_pred)
                else:
                    y_pred_orig = y_pred


                # convert cylindrical to xyz for calculate the distance in the euclidean space
                '''
                if cylindrical:
                    rho, eta, phi = y_pred_orig[0][0], y_pred_orig[0][1], y_pred_orig[0][2]
                    #print(rho,eta,phi)
                    x, y, z = convert_rhoetaphi_to_xyz(rho, eta, phi)

                    #print(x,y,z)
                    y_pred_orig[0][0] = x
                    y_pred_orig[0][1] = y
                    y_pred_orig[0][2] = z

                    #print(y_pred_orig)
                '''

                curr_layer = layers[i]

                for m, metric in enumerate(metrics):
                    # calculate the nearest hit with a polar distance function 
                    if metric == 'polar':
                        if points_3d:
                            dist = distance.cdist(curr_layer, y_pred_orig, distance_cylindrical_3D)
                        else:
                            dist = distance.cdist(curr_layer, y_pred_orig, distance_cylindrical_2D)
                    else: # other distances   
                        dist = distance.cdist(curr_layer, y_pred_orig, metric)

                    # get the hit with lowest distance in orignal coordinates
                    idx = np.argmin(dist)
                    most_nearest = curr_layer[idx]

                    # if curr_hit is in cartesian coord, near_pred must be in cartesian coord too
                    # very small numbers are differents or equals
                    # we count how many times we get the correct hit by layer
                    if np.isclose(curr_hit, most_nearest, atol=tol).all():
                        corrects_by_layer[m][i]+=1

                # removing the hit from layer, mayority of times is not good
                if remove_hit:
                    layers[i] = np.delete(layers[i], idx, 0)

                if verbose:
                    print('pred:', y_pred)
                    print('inv pred:', y_pred_orig)
                    print('current:', curr_hit)
                    print('nearest:', most_nearest)

                # i think it is not necessary
                '''
                if cylindrical:
                    y, z = near_pred[0], near_pred[1]
                    #print(x,y,z)
                    _, eta, phi = convert_xyz_to_rhoetaphi(1, y, z)
                    #print(rho,eta,phi)
                    #near_pred[0] = rho
                    near_pred[0] = eta
                    near_pred[1] = phi
                '''
                # we change to the original input at the same postion
                #if cylindrical:
                #    near_pred = curr_layer_[idx]

                #near adiciona em valores originaeis
                predicted.append(most_nearest)

                curr_frame = curr_frame_orig[1:]
                curr_frame = np.insert(curr_frame, [3], predicted[-1], axis=0)


            pred_tracks.append(predicted)

            if verbose:
                if j == 10: break

        print('[Model] Prediction Finished.')
        timer.stop()

        return pred_tracks, corrects_by_layer

    def nearest_hit(self, hit, hits,
                             silent = True,
                             dist_hit = False,
                             metric = 'euclidean'):
     
        if silent is False:
            timer = Timer()
            timer.start()

        dist = distance.cdist(hits, hit, metric)

        # get the index of minimum distance
        target_hit_index = np.argmin(dist)
        
        if silent is False:    
            print("--- N_hits: %s" % len(hits))
            print("--- Hit index: %s" % target_hit_index)
            print("--- " + str(metric) + " distance: " + str(dist[target_hit_index]))
            print("--- time: %s seconds" % timer.stop())

        # get hits coordinates 
        real_hit = hits[target_hit_index, :]
        real_hit = np.array(real_hit)

        # removing the hit from bag
        #hits = np.delete(hits, target_hit_index, 0)
        
        if dist_hit is False:
            return real_hit, target_hit_index
        else:
            return real_hit, target_hit_index, np.min(dist)

    def rmse(y_true, y_pred):
        return K.sqrt(K.mean(K.square(y_pred - y_true), axis=-1)) 




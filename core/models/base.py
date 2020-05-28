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
from keras.backend.tensorflow_backend import set_session
from keras.models import Sequential, load_model
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.utils import plot_model   

from core.utils.utils import *


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
        print(self.save_fnameh5)

        self.save_fname = os.path.join(configs['paths']['save_dir'], 'architecture-%s.png' % self.name)

        self.save = configs['training']['save_model']
        
        if configs['training']['use_gpu'] == True:
            #config = tf.ConfigProto( device_count = {'GPU': 0 , 'CPU': 0} ) 
            config = tf.ConfigProto()
            config.gpu_options.allow_growth = True
            sess = tf.Session(config=config) 
            set_session(sess)
            tf.device('/gpu:0')
        else:
            config=tf.ConfigProto(log_device_placement=True)
            sess = tf.Session(config=config)
            set_session(sess)
        
        #set_random_seed(42)
        tf.compat.v1.set_random_seed(0)

    def load_model(self):
        if self.exist_model(self.save_fnameh5):
            print('[Model] Loading model from file %s' % self.save_fnameh5)
            self.model = load_model(self.save_fnameh5)
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

    def train(self, x, y, epochs, batch_size, validation, shuffle=False):
        timer = Timer()
        timer.start()
        print('[Model] Training Started')
        print('[Model] %s epochs, %s batch size' % (epochs, batch_size))
        #print('[Model] Shape of data train: ', x.shape) 
        #save_fname = os.path.join(save_dir, '%s-e%s.h5' % (dt.datetime.now().strftime('%d%m%Y-%H%M%S'), str(epochs)))
        callbacks = [
            EarlyStopping(monitor='loss', mode='min', verbose=1),
            ModelCheckpoint(filepath=self.save_fnameh5, monitor='val_loss', mode='min', save_best_only=True)
        ]
        history = self.model.fit(
            x,
            y,
            validation_split=validation,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            shuffle=shuffle
        )

        if self.save == True:
            self.save_model(self.save_fnameh5)             
       
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
                                        normalise=False, cylindrical=False, verbose=False, tol=0.01):
        
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
        
        decimals = 2
       
        # covert to original values
        #y_true = data.inverse_transform_test_y(y_test)
        #y_true = y_true.round(decimals)
        
        #change the dataset by cartesian coordinates
        if cylindrical:
            y_test = y_test_aux
        else:
            y_test = y_test
            
        # bag_of_hits  
        bag_of_hits_all = np.array(convert_matrix_to_vec(y_test, num_features))

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
                curr_hit = curr_track[i]
                begin = end
                
                if verbose:
                    # primeira esta em originais
                    print('input:\n', curr_frame)
                
                if normalise:
                    curr_frame = data.x_scaler.transform(np.reshape(curr_frame,(1,12)))
                    curr_frame_orig = data.inverse_transform_x(pd.DataFrame(curr_frame).values.flatten())
                    curr_frame_orig = np.reshape(curr_frame_orig, (4,3))
                    curr_frame = np.reshape(curr_frame, (4,3))
                else:
                    curr_frame = curr_frame
                    curr_frame_orig = curr_frame
                    
                if verbose:
                    print('input:\n', curr_frame)
                    #print('input orig:\n', curr_frame_orig)
                
                #print('input inv :\n', curr_frame_inv.reshape(4,3))
                #print('newaxis ', curr_frame[np.newaxis,:,:])
                # input must be scaled curr_frame
                y_pred = self.model.predict(curr_frame[np.newaxis,:,:])

                
                y_pred = np.reshape(y_pred, (1, 3))
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

        return pred_sequences, count_correct

 
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




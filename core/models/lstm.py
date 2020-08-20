__author__ = "Steve Ataucuri"
__copyright__ = "Sprace.org.br"
__version__ = "1.0.0"

from keras.layers import Input, Dense, Activation, Dropout, LSTM, Lambda
from keras.layers import TimeDistributed, RepeatVector
#from keras.layers import concatenate, CuDNNLSTM
from keras.layers import concatenate
from keras.models import Model
from keras.optimizers import Adam, SGD, RMSprop
from core.models.base import BaseModel
from core.utils.utils import Timer
from core.models.gaussian_loss import gaussian_layer_2d, gaussian_nll

class ModelLSTM(BaseModel):
    """A class for an building and inferencing an lstm model"""
 
    def __init__(self, configs):
        super().__init__(configs)
        self.c = configs

    def build_model(self):
        timer = Timer()
        timer.start()
 
        print('[Model] Creating model..')   
        
        configs = self.c     

        for layer in configs['model']['layers']:

            neurons = layer['neurons'] if 'neurons' in layer else None
            dropout_rate = layer['rate'] if 'rate' in layer else None
            activation = layer['activation'] if 'activation' in layer else None
            return_seq = layer['return_seq'] if 'return_seq' in layer else None
            input_timesteps = layer['input_timesteps'] if 'input_timesteps' in layer else None
            input_features = layer['input_features'] if 'input_features' in layer else None
            dropout = layer['dropout'] if 'dropout' in layer else None
            stateful = layer['stateful'] if 'stateful' in layer else None

            #print('input_features %s input_timesteps %s ' % ( input_features, input_timesteps))
            #print('batch_size: ', self.batch_size)
            if layer['type'] == 'lstm':
                if dropout is None:
                    if stateful:
                        #inp = Input(batch_shape= (batch_size, input_timesteps, input_features), name="input")
                        # if stateful is True the shuffle parameter must be False
                        self.stateful = stateful
                        self.model.add(LSTM(neurons, batch_input_shape=(self.batch_size, input_timesteps, input_features),
                            return_sequences=return_seq, stateful=stateful))
                    else:
                        self.model.add(LSTM(neurons, input_shape=(input_timesteps, input_features), return_sequences=return_seq))                    
                else:
                    # Dropout can be applied to the input connection within the LSTM nodes.
                    #self.model.add(LSTM(neurons, batch_input_shape=(self.batch_size, input_timesteps, input_features),
                    #    return_sequences=return_seq, stateful=True, dropout=dropout))
                    
                    # applied to input signal of lstm units 
                    self.model.add(LSTM(neurons, batch_input_shape=(self.batch_size, input_timesteps, input_features),
                        return_sequences=return_seq, stateful=True, recurrent_dropout=dropout))

            if layer['type'] == 'dense':
                self.model.add(Dense(neurons, activation=activation))
            if layer['type'] == 'dropout':
                self.model.add(Dropout(dropout_rate))
            if layer['type'] == 'repeatvector':
                self.model.add(RepeatVector(neurons))
            if layer['type'] == 'timedistributed':                
                self.model.add(TimeDistributed(Dense(neurons)))
            if layer['type'] == 'activation':
                self.model.add(Activation('linear'))
        
        if configs['model']['optimizer'] == 'adam':
            opt = Adam(lr=configs['model']['learningrate'])
        elif configs['model']['optimizer'] == 'rmsprop':
            opt = RMSprop(lr=configs['model']['learningrate'])

        self.model.compile(loss=configs['model']['loss'], optimizer=opt, metrics=configs['model']['metrics'])
        print(self.model.summary())
        print('[Model] Model Compiled with structure:', self.model.inputs)
        #self.save_architecture(self.save_fname)     
        timer.stop()

class ModelLSTMParallel(BaseModel):
    """A class for an building and inferencing an lstm model with cuDnnlstm"""
 
    def __init__(self, configs):
        super().__init__(configs)
        self.c = configs
 
    def build_model(self):
        timer = Timer()
        timer.start()
        
        print('[Model] Creating model..')    

        # this model is not sequencial
        self.model = None
        configs = self.c

        for layer in configs['model']['layers']:

            neurons = layer['neurons'] if 'neurons' in layer else None
            dropout_rate = layer['rate'] if 'rate' in layer else None
            activation = layer['activation'] if 'activation' in layer else None
            return_seq = layer['return_seq'] if 'return_seq' in layer else None
            input_timesteps = layer['input_timesteps'] if 'input_timesteps' in layer else None
            input_features = layer['input_features'] if 'input_features' in layer else None
            dropout = layer['dropout'] if 'dropout' in layer else None
            stateful = layer['stateful'] if 'stateful' in layer else None

            #create Neural Network
            if layer['type'] == 'lstm':

                first_input = Input(shape=(input_timesteps, input_features))        
                first_output = LSTM(neurons, return_sequences=return_seq, return_state=False)(first_input)

                second_input = Input(shape=(input_timesteps, input_features)) # without number of features, just with input_timesteps
                second_output = LSTM(neurons, return_sequences=return_seq, return_state=False)(second_input)

                third_input = Input(shape=(input_timesteps, input_features)) # without number of features, just with input_timesteps
                third_output = LSTM(neurons, return_sequences=return_seq, return_state=False)(third_input)

                output = concatenate([first_output, second_output, third_output])

            if layer['type'] == 'dense':
                output = Dense(neurons, activation = activation)(output)
            if layer['type'] == 'dropout':
                output = Dropout(dropout_rate)(output)

        if configs['model']['optimizer'] == 'adam':
            opt = Adam(lr=configs['model']['learningrate'])
        elif configs['model']['optimizer'] == 'rmsprop':
            opt = RMSprop(lr=configs['model']['learningrate'])

        self.model = Model(inputs=[first_input, second_input, third_input], outputs=output)
        self.model.compile(loss=configs['model']['loss'], optimizer=opt, metrics=configs['model']['metrics'])
        print(self.model.summary())
        print('[Model] Model Compiled with structure:', self.model.inputs)
        
        timer.stop()


class GaussianLSTM(BaseModel):
    """A class for an building and inferencing an lstm model"""
 
    def __init__(self, configs):
        super().__init__(configs)
        self.c = configs

    def build_model(self):
        timer = Timer()
        timer.start()
 
        print('[Model] Creating model..')   
        
        configs = self.c     
        self.model = None
        for layer in configs['model']['layers']:

            neurons = layer['neurons'] if 'neurons' in layer else None
            dropout_rate = layer['rate'] if 'rate' in layer else None
            activation = layer['activation'] if 'activation' in layer else None
            return_seq = layer['return_seq'] if 'return_seq' in layer else None
            input_timesteps = layer['input_timesteps'] if 'input_timesteps' in layer else None
            input_features = layer['input_features'] if 'input_features' in layer else None
            dropout = layer['dropout'] if 'dropout' in layer else None
            stateful = layer['stateful'] if 'stateful' in layer else None
            
        inputs = Input(shape=(input_timesteps, input_features))        
        outputs = LSTM(neurons, return_sequences=False, activation='tanh')(inputs)
        #outputs = LSTM(neurons, return_sequences=False, activation='tanh')(outputs)
        outputs = Dense(2, activation='linear')(outputs)

        distributions = Lambda(gaussian_layer_2d)(outputs)

        ####################
        self.model = Model(inputs=inputs, outputs=distributions)

        if configs['model']['optimizer'] == 'adam':
            opt = Adam(lr=configs['model']['learningrate'])
        elif configs['model']['optimizer'] == 'rmsprop':
            opt = RMSprop(lr=configs['model']['learningrate']) 

        print(self.model.summary())
        self.model.compile(loss=gaussian_nll, optimizer=opt, metrics=['accuracy'])       
        print('[Model] Model Compiled with structure:', self.model.inputs)
        self.save_architecture(self.save_fname)     
        timer.stop()
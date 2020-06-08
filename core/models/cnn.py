__author__ = "Steve Ataucuri"
__copyright__ = "Sprace.org.br"
__version__ = "1.0.0"

from keras.layers import Input, Dense, Activation, Dropout, Flatten
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
from keras.layers import concatenate

from keras.models import Model

from .base import BaseModel
from .timer import Timer

class ModelCNN(BaseModel):
    """A class for an building and inferencing a base line cnn model"""
 
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
            input_timesteps = layer['input_timesteps'] if 'input_timesteps' in layer else None
            input_features = layer['input_features'] if 'input_features' in layer else None
            filters = layer['filters'] if 'filters' in layer else None
            kernel_size = layer['kernel_size'] if 'kernel_size' in layer else None
            pool_size = layer['pool_size'] if 'pool_size' in layer else None

            #print('input_features %s input_timesteps %s ' % ( input_features, input_timesteps))

            if layer['type'] == 'cnn':
                self.model.add(Conv1D(filters=filters, kernel_size=kernel_size, activation=activation, 
                    input_shape=(input_timesteps, input_features)))
            if layer['type'] == 'dense':
                self.model.add(Dense(neurons, activation=activation))
            if layer['type'] == 'dropout':
                self.model.add(Dropout(dropout_rate))
            if layer['type'] == 'maxpooling':
                self.model.add(MaxPooling1D(pool_size=pool_size))
            if layer['type'] == 'flatten':                
                self.model.add(Flatten())
            if layer['type'] == 'activation':
                self.model.add(Activation('linear'))
        

        self.model.compile(loss=configs['model']['loss'], optimizer=configs['model']['optimizer'], metrics=configs['model']['metrics'])       
        print(self.model.summary())
        print('[Model] Model Compiled with structure:', self.model.inputs)
        #self.save_architecture(self.save_fname)     
        timer.stop()

class ModelCNNParallel(BaseModel):
    """A class for an building and inferencing a base line cnn model"""
 
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
            input_timesteps = layer['input_timesteps'] if 'input_timesteps' in layer else None
            input_features = layer['input_features'] if 'input_features' in layer else None
            filters = layer['filters'] if 'filters' in layer else None
            kernel_size = layer['kernel_size'] if 'kernel_size' in layer else None
            pool_size = layer['pool_size'] if 'pool_size' in layer else None

            #print('input_features %s input_timesteps %s ' % ( input_features, input_timesteps))

            if layer['type'] == 'cnn':
                first_input = Input(shape=(input_timesteps, input_features)) 
                first_output = Conv1D(filters=filters, kernel_size=kernel_size, activation=activation, 
                    input_shape=(input_timesteps, input_features))(first_input)
                
                second_input = Input(shape=(input_timesteps, input_features))
                second_output = Conv1D(filters=filters, kernel_size=kernel_size, activation=activation, 
                    input_shape=(input_timesteps, input_features))(second_input)

                third_input = Input(shape=(input_timesteps, input_features))
                third_output = Conv1D(filters=filters, kernel_size=kernel_size, activation=activation, 
                    input_shape=(input_timesteps, input_features))(third_input)
          
                output = concatenate([first_output, second_output, third_output])
     
            if layer['type'] == 'dense':
                output = Dense(neurons, activation=activation)(output)
            if layer['type'] == 'dropout':
                output = Dropout(dropout_rate)(output)
            if layer['type'] == 'maxpooling':
                output = MaxPooling1D(pool_size=pool_size)(output)
            if layer['type'] == 'flatten':                
                output = Flatten()(output)
            if layer['type'] == 'activation':
                output = Activation('linear')(output)
        
        self.model = Model(inputs=[first_input, second_input, third_input], outputs=output)
        self.model.compile(loss=configs['model']['loss'], optimizer=configs['model']['optimizer'], metrics=configs['model']['metrics'])       
        print(self.model.summary())
        print('[Model] Model Compiled with structure:', self.model.inputs)        
        timer.stop()


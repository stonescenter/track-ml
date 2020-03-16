__author__ = "Steve Ataucuri"
__copyright__ = "Sprace.org.br"
__version__ = "1.0.0"

from keras.layers import Input, Dense, Activation, Dropout, SimpleRNN
from keras.models import Model

from core.models.base import BaseModel
from core.utils.utils import Timer

class ModelRNN(BaseModel):
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
            input_timesteps = layer['input_timesteps'] if 'input_timesteps' in layer else None
            input_features = layer['input_features'] if 'input_features' in layer else None
            
            #print('input_features %s input_timesteps %s ' % ( input_features, input_timesteps))

            if layer['type'] == 'rnn':
                # batch_size 'e o input_timesteps
                # a 2D input with shape `(batch_size, input_dim)`.
                self.model.add(SimpleRNN(neurons, input_dim=input_features*input_timesteps,
                    kernel_initializer='normal', activation=activation))
            if layer['type'] == 'dense':
                self.model.add(Dense(neurons, kernel_initializer='normal', activation=activation))            
            if layer['type'] == 'dropout':
                self.model.add(Dropout(dropout_rate))
            if layer['type'] == 'activation':
                self.model.add(Activation(activation))
        
        print(self.model.summary())
        self.model.compile(loss=configs['model']['loss'], optimizer=configs['model']['optimizer'], metrics=['accuracy'])       
        print('[Model] Model Compiled with structure:', self.model.inputs)
        self.save_architecture(self.save_fname)     
        timer.stop()
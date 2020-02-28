__author__ = "Steve Ataucuri"
__copyright__ = "Sprace.org.br"
__version__ = "1.0.0"

from keras.layers import Input, Dense, Activation, Dropout, LSTM
from keras.layers import TimeDistributed, RepeatVector
from keras.layers import concatenate, CuDNNLSTM
from keras.models import Model

from core.models.base import BaseModel
from core.utils.utils import Timer

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
            
            print('input_features %s input_timesteps %s ' % ( input_features, input_timesteps))

            if layer['type'] == 'lstm':
                self.model.add(LSTM(neurons, input_shape=(input_timesteps, input_features), return_sequences=return_seq))
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
        
        print(self.model.summary())
        self.model.compile(loss=configs['model']['loss'], optimizer=configs['model']['optimizer'], metrics=['accuracy'])       
        print('[Model] Model Compiled with structure:', self.model.inputs)
        self.save_architecture(self.save_fname)     
        timer.stop()


class ModelLSTMCuDnnParalel(BaseModel):
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

            #create Neural Network
            if layer['type'] == 'lstm-paralel':

                first_input = Input(shape=(input_timesteps, input_features))        
                first_output = CuDNNLSTM(neurons, return_sequences=return_seq, return_state=False)(first_input)

                second_input = Input(shape=(input_timesteps, 1)) # without number of features, just with input_timesteps
                second_output = CuDNNLSTM(neurons, return_sequences=return_seq, return_state=False)(second_input)

                output = concatenate([first_output, second_output], axis=-1)

            if layer['type'] == 'dense':
                output = Dense(neurons, activation = activation)(output)
            if layer['type'] == 'dropout':
                output = Dropout(0.5)(output)

        self.model = Model(inputs=[first_input, second_input], outputs=output)
        self.model.compile(loss=configs['model']['loss'], optimizer=configs['model']['optimizer'], metrics=['accuracy'])
        print(self.model.summary())
        print('[Model] Model Compiled with structure:', self.model.inputs)      
        self.save_architecture(self.save_fname) 
        timer.stop()

class ModelLSTMParalel(BaseModel):
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

            #create Neural Network
            if layer['type'] == 'lstm-paralel':

                first_input = Input(shape=(input_timesteps, input_features))        
                first_output = LSTM(neurons, return_sequences=return_seq, return_state=False)(first_input)

                second_input = Input(shape=(input_timesteps, 1)) # without number of features, just with input_timesteps
                second_output = LSTM(neurons, return_sequences=return_seq, return_state=False)(second_input)

                output = concatenate([first_output, second_output], axis=-1)

            if layer['type'] == 'dense':
                output = Dense(neurons, activation = activation)(output)
            if layer['type'] == 'dropout':
                output = Dropout(dropout_rate)(output)

        self.model = Model(inputs=[first_input, second_input], outputs=output)
        self.model.compile(loss=configs['model']['loss'], optimizer=configs['model']['optimizer'], metrics=['accuracy'])
        print(self.model.summary())
        print('[Model] Model Compiled with structure:', self.model.inputs)
        self.save_architecture(self.save_fname) 

        timer.stop()

def create_Neural_Network(neurons):
    #create Neural Network
    xshape=Input(shape=(4,3))
    #yshape=Input(shape=(3,4))
    yshape=Input(shape=(4,1))

    admi=CuDNNLSTM(neurons,return_sequences=False,return_state=False)(xshape)
    pla=CuDNNLSTM(neurons,return_sequences=False,return_state=False)(yshape)

    #admi=LSTM(neurons,return_sequences=False)(xshape)
    #pla=LSTM(neurons,return_sequences=False)(yshape)
    out=concatenate([admi,pla],axis=-1)

    #output=Dense(3, activation=activation, kernel_initializer=init_mode)(out)

    output2=Dense(neurons, activation = 'relu')(out)
    output3=Dropout(0.5)(output2)
    output4=Dense(neurons, activation = 'relu')(output3)

    output=Dense(3)(output4)
    #output=Dense(3)(out)

    #output=Dense(3, activation = 'relu')(output4)
    #output=Dense(3, activation = 'relu')(out)

    model = Model(inputs=[xshape, yshape], outputs=output)

    model.compile(optimizer = 'RMSprop', loss='mean_absolute_error', metrics =['accuracy'])
    #model.compile(optimizer = optf,loss=lossf, metrics =['mse'])
    #model.compile(optimizer = optf,loss=lossf, metrics =['mse', 'mae', 'mape', 'cosine',mean_pred])
    #model.compile(optimizer = optf,loss=lossf, metrics =['accuracy' , B_accuracy])
    #model.compile(optimizer = optf,loss=lossf, metrics={'output_a': 'accuracy'})

    
    return model
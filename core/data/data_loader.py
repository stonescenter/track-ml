__author__ = "Steve Ataucuri"
__copyright__ = "Sprace.org.br"
__version__ = "1.0.0"

import numpy as np
import pandas as pd

#from torch.utils.data import Dataset, DataLoader

from sklearn.preprocessing import MinMaxScaler, StandardScaler
from enum import Enum

class FeatureType(Enum):
	Divided = 1, # indica as caracteristicas estao divididas em posiciones e outras informacoes
	Mixed = 2, # indica que todas as caracteristicas estao juntas
	Positions = 3 # indica que so tem posicoes dos hits

class KindNormalization(Enum):
	Scaling = 1,
	Zscore = 2,
	Polar = 3,
	Nothing = 4

class Dataset():
	def __init__(self, input_path, kind_normalization):

		#np.set_printoptions(suppress=True)

		# com index_col ja não inclui a coluna index 
		dataframe = pd.read_csv(input_path, header=0, engine='python')
		
		self.kind = kind_normalization
		if self.kind == KindNormalization.Scaling:
			self.x_scaler = MinMaxScaler(feature_range=(0, 1))
			self.y_scaler = MinMaxScaler(feature_range=(0, 1)) 			

		elif kind_normalization == KindNormalization.Zscore:
			self.x_scaler = StandardScaler() # mean and standart desviation
			self.y_scaler = StandardScaler() # mean and standart desviation
			
		'''
				if normalise:            
				    data = self.scaler.fit_transform(dataframe.values)
				    data = pd.DataFrame(data, columns=columns)
				else:
				    data = pd.DataFrame(dataframe.values, columns=columns)
		'''
		self.start_hits = 9
		self.interval = 11
		self.cylindrical = False
		self.data = dataframe.iloc[:, self.start_hits:]
		self.self = 0
		print("[Data] data loaded from ", input_path)

	def prepare_training_data(self, feature_type, normalise=True, cylindrical=False):

		if not isinstance(feature_type, FeatureType):
			raise TypeError('direction must be an instance of FeatureType Enum')
	

		self.cylindrical = cylindrical

		interval = self.interval

		# x, y, z coordinates
		if cylindrical == False:
			bp=1
			ep=4
			bpC=10
			epC=11
		   
		# cilyndrical coordinates    
		elif cylindrical == True:
			bp=4
			ep=7
			bpC=10
			epC=11  

		df_hits_values = None
		df_hits_positions = None

		if feature_type==FeatureType.Divided:
			# get hits positions p1(X1,Y1,Z1) p2(X2,Y2,Z2) p3(X3,Y3,Z3) p4(X4,Y4,Z4)
			df_hits_positions = self.data.iloc[:, np.r_[
							bp:ep,
							bp+(interval*1):ep+(interval*1),
							bp+(interval*2):ep+(interval*2),
							bp+(interval*3):ep+(interval*3)]]
			# get hits values p1(V1,V2,V3,V4)
			df_hits_values = self.data.iloc[:, np.r_[
							bpC:epC,
							bpC+(interval*1):epC+(interval*1),
							bpC+(interval*2):epC+(interval*2),
							bpC+(interval*3):epC+(interval*3)]]

			frames = [df_hits_positions, df_hits_values]
			df_hits_positions = pd.concat(frames, axis=1)

		if feature_type==FeatureType.Mixed:			                               

			df_hits_positions = self.data.iloc[:, np.r_[
							bp:ep,
							bpC:epC,
							bp+(interval*1):ep+(interval*1), bpC+(interval*1):epC+(interval*1),
							bp+(interval*2):ep+(interval*2),  bpC+(interval*2):epC+(interval*2),
							bp+(interval*3):ep+(interval*3),  bpC+(interval*3):epC+(interval*3)]]

		elif feature_type==FeatureType.Positions:			                               

			df_hits_positions = self.data.iloc[:, np.r_[
							bp:ep,
							bp+(interval*1):ep+(interval*1),
							bp+(interval*2):ep+(interval*2),
							bp+(interval*3):ep+(interval*3)]]

		self.x_data = df_hits_positions	          
		self.y_data = self.data.iloc[:, np.r_[bp+(interval*4):(bp+(interval*4)+3)]]

		self.len = len(self.data) 

		xcolumns = self.x_data.columns
		ycolumns = self.y_data.columns

		# normalization just of features.
		if normalise:
			xscaled = self.x_scaler.fit_transform(self.x_data.values)
			self.x_data = pd.DataFrame(xscaled, columns=xcolumns)			

			yscaled = self.y_scaler.fit_transform(self.y_data.values)
			self.y_data = pd.DataFrame(yscaled, columns=ycolumns)	

		print("[Data] shape datas X: ", self.x_data.shape)
		print("[Data] shape data y: ", self.y_data.shape)
		print('[Data] len data total:', self.len)

		#y_hit_info = self.getitem_by_hit(hit_id)
		
		if feature_type==FeatureType.Divided:
			# return x_data, y_data normalizated with data splited

			return (self.x_data.iloc[:,0:12], self.x_data.iloc[:,-4:], self.y_data)
		else:
			# return x_data, y_data normalizated with no data splited
			return (self.x_data, self.y_data)

	def get_training_data(self, cylindrical=False, hits=5):
		'''
			This function return a dataset with the numbers of hits with
			cartesian or cylindrical positions.
		'''

		self.cylindrical = cylindrical

		begin_coord = 0
		end_coord = 0
		begin_val = 10
		end_val = 11

		if self.cylindrical == False:
			begin_coord = 1
			end_coord = 4
		# cilyndrical coordinates    
		elif self.cylindrical == True:
			begin_coord = 4
			end_coord = 7

		begin_cols = [begin_coord+(self.interval*hit) for hit in range(0, hits)]
		end_cols = [end_coord+(self.interval*hit) for hit in range(0, hits)]

		new_df = pd.DataFrame()
		for c in range(0,len(begin_cols)):
		    frame = self.data.iloc[:,np.r_[begin_cols[c]:end_cols[c]]]
		    new_df = pd.concat([new_df, frame], axis=1)

		self.x_data = new_df	          
		#self.y_data = self.x_data.iloc[:, -3:] # last hit

		# normalization just of features.
		#if normalise:
		#	xscaled = self.x_scaler.fit_transform(self.x_data.values)
		#	self.x_data = pd.DataFrame(xscaled)			

		self.len = len(self.x_data) 

		print("[Data] Data Set original with %s hits and shape%s" % (hits, self.x_data.shape))

		return self.x_data

	def convert_to_supervised(self, sequences, n_hit_in, n_hit_out, n_features, normalise=False):
		'''
			n_hit_in : 4 number of hits
			n_hit_out: 1 number of future hits
			n_features 3
		'''
		X , Y = [],[]

		rows = sequences.shape[0]
		cols = sequences.shape[1]

		for i in range(0, rows):
			end_idx = 0
			out_end_idx = 0
			for j in range(0, cols, n_features):
				end_ix = j + n_hit_in*n_features
				out_end_idx = end_ix + n_hit_out*n_features

				if out_end_idx > cols+1:
					#print('corta ', out_end_idx)
					break
				#if i < 5:	
				#    print('[%s,%s:%s][%s,%s:%s]' % (i, j, end_ix, i, end_ix, out_end_idx))

				seq_x, seq_y = sequences[i, j:end_ix], sequences[i, end_ix:out_end_idx]

				X.append(seq_x)
				Y.append(seq_y)
					
		#return np.array(X) , np.array(Y)
		#xcolumns = self.x_data.columns
		#ycolumns = self.y_data.columns

		# normalization just of features.
		if normalise:
			xscaled = self.x_scaler.fit_transform(X)
			self.x_data = pd.DataFrame(xscaled)			

			yscaled = self.y_scaler.fit_transform(Y)
			self.y_data = pd.DataFrame(yscaled)	
		else:
			self.x_data = pd.DataFrame(X)			
			self.y_data = pd.DataFrame(Y)				

		return pd.DataFrame(self.x_data) , pd.DataFrame(self.y_data)

	def convert_supervised_to_normal(self, sequences, n_hit_in, n_hit_out, hits):
		'''
			This function convert the predicted sequences to a vector
			n_hit_in : 4 number of hits
			n_hit_out: 1 number of future hits
			hits	 : 10
		'''

		Y = []

		len_pred_seq = hits - n_hit_in
		len_seq = len(sequences)
		end_ix = 0

		for x in range(0, len_seq, len_pred_seq):
			cols = []
			end_ix = x + len_pred_seq
			pred_seq = sequences[x:end_ix,:]

			#print('matrix: ', pred_seq)
					
			pred_seq = np.matrix(pred_seq).flatten().tolist()			
			#result = m_pred_seq.flatten().tolist()
			#print('result: ', pred_seq)
			#print('result 0: ', pred_seq[0])
			#for i in range(0, len(pred_seq)):		
			#	print('\t row added:', pred_seq[i,:])
			#	ocls.append(pred_seq[i,:])
			#print(lst.to_matrix())

			Y.append(pred_seq[0])

		return Y

	def train_test_split(self, X, y, train_size=0):

		#self.data = self.data.values
		assert len(X) == len(y), 'Invalid len size of dataset!.'
		assert train_size >= 0 and train_size < 1, 'Invalid train_size.'

		i_split = round(len(X) * train_size)

		print("[Data] Splitting data at %d with %s" %(i_split, train_size))

		if i_split > 0:
			x_train = X.iloc[0:i_split,0:]
			y_train = y.iloc[0:i_split,0:]

			x_test = X.iloc[i_split:,0:]
			y_test = y.iloc[i_split:,0:]

			return (x_train, x_test, y_train, y_test)
		elif i_split == 0:
			x_train = X.iloc[0:,0:]
			y_train = y.iloc[0:,0:]

			return (x_train, y_train)

	def reshape3d(self, x, time_steps, num_features):
		len_x = x.shape[0]
		#return np.reshape(x.values.flatten(), (len_x, time_steps, num_features))
		return np.reshape(x.values, (len_x, time_steps, num_features))

	def reshape2d(self, x, num_features):
		#len_x = x.shape[0]
		return np.reshape(x.values.flatten(), (x.shape[0]*x.shape[1], num_features))
		#return np.reshape(x, (x.size, num_features))

	def getitem_by_hit(self, hit):
		'''
			Get information of one hit
			paramenters:
				hit : number of hit
	
		'''
		# hit_id_0,x_0,y_0,z_0,rho_0,eta_0,phi_0,volume_id_0,layer_id_0,module_id_0,value_0,		
		#i_split = int(len(self.data) * train_split)

		begin_hit = 'hit_id_'
		end_hit = 'value_'
		begin = begin_hit+str(hit)
		end = end_hit+str(hit)

		ds_hit = self.data.loc[:, begin:end]

		return ds_hit

	def __getitem__(self, index):
		
		x = self.x_data.iloc[index,0:].values.astype(np.float).reshape(1,self.x_data.shape[1])
		y  = self.y_data.iloc[index,0]

		return	x, y

	def __len__(self):
		return self.len

	def inverse_transform_y(self, data):
		return self.y_scaler.inverse_transform(data)

	def inverse_transform_x(self, data):
		return self.x_scaler.inverse_transform(data)
from ModuleFCNN import FCNN
import pandas as pd
import torch
import os
import gc
import DataProcessing as DP
from torch.utils.data import TensorDataset,DataLoader
import time
from sklearn.metrics import r2_score as R2

class AutoEncoder(FCNN):
	
	def forward(self, X):
		X = self.Encode(X, mode='train')
		X = self.Decode(X, mode='train')
		return X
	
	def Encode(self, X: torch.Tensor, mode='eval'):
		if mode.lower() == 'eval':
			self.SetMode('eval')
			if self.normalizetype == 'minmax':
				X = (X - self.tensorInputMin) / (self.tensorInputMax - self.tensorInputMin)
			elif self.normalizetype == 'gaussian':
				X = (X - self.tensorInputMean) / self.tensorInputStd
			X=torch.nan_to_num(X,posinf=0,neginf=0,nan=0)
		
		for i in range(len(self.encoderdim) - 1):
			X = self.EncoderLayersModule[i](X)
			if self.batchnorm:
				X = self.EncoderBatchNormLayersModule[i](X)
			if not self.activationtype == 'none':
				X = self.Activation(X)
			if self.dropout:
				X = self.DropoutLayer(X)
		return X
	
	def Decode(self, X: torch.Tensor, mode='eval'):
		if mode.lower() == 'eval':
			self.SetMode('eval')
		
		for i in range(len(self.decoderdim) - 1):
			X = self.DecoderLayersModule[i](X)
			if (i + 1) == len(self.decoderdim) - 1:  # if Output Layer:
				break
			if self.batchnorm:
				X = self.DecoderBatchNormLayersModule[i](X)
			if not self.activationtype == 'none':
				X = self.Activation(X)
			if self.dropout:
				X = self.DropoutLayer(X)
		
		if mode.lower() == 'eval':
			if self.normalizetype == 'minmax':
				X = X * (self.tensorOutputMax - self.tensorOutputMin) + self.tensorOutputMin
			elif self.normalizetype == 'gaussian':
				X = X * self.tensorOutputStd + self.tensorOutputMean
			X = torch.nan_to_num(X, posinf=0, neginf=0, nan=0)
		
		return X
	
	
	def SetHyperParams(self, epochs=0, encoderdim: list = None, decoderdim: list = None,
					   batchsize: int = 0, lr: float = 1e-3, bias: bool = True,
					   activation: str = 'relu', loss: str = 'mse', optimizer: str = 'adam',
					   adamWdecay: float = 0, l2_lambda:float=0, dropout: float = 0, batchnorm: bool = False,
					   weight_init: bool = False, bias_init: bool = False):
		"""
		:param epochs: Epochs to add.
		:param nodes: Number of nodes in fully-connected hidden layers.
		:param hiddenlayers: Number of hidden layers.
		:param batchsize: Number of train data batch size.
		:param lr: Learning rate.
		:param bias: Whether to use bias term in all layers.
		:param activation: Activation function type.
		:param loss: Loss function type.
		:param optimizer: Optimizer type.
		:param adamWdecay: L2 regularization rate for ADAM optimizer.
		:param dropout: Dropout probability in hidden layers.
		:param batchnorm: Whether to use torch.BatchNorm1d() between hidden layers.
		:param weight_init: Whether to initialize all weights by Xavier uniform initialization method.
		:param bias_init: Whether to initialize all bias to torch.zeros.
		:return: None
		"""
		# Define Hyperparams
		# self.Numnodes = nodes
		# self.Numhiddenlayers = hiddenlayers
		self.encoderdim = encoderdim
		if decoderdim == None:
			self.decoderdim = list(reversed(encoderdim))
		else:
			self.decoderdim = decoderdim
		
		self.Numbatchsize = batchsize
		self.learningrate = lr
		self.bias = bias
		self.dropout = dropout
		self.batchnorm = batchnorm
		self.weight_init = weight_init
		self.bias_init = bias_init
		self.activationtype = activation.lower()
		self.lossfunctype = loss.lower()
		self.optimizertype = optimizer.lower()
		self.adamWdecay = adamWdecay
		self.l2_lambda=l2_lambda
		
		# Nonlinear activation
		if self.activationtype == 'relu':
			self.Activation = torch.nn.ReLU()
		if self.activationtype == 'leakyrelu':
			self.Activation = torch.nn.LeakyReLU()
		
		if self.activationtype == 'tanh':
			self.Activation = torch.nn.Tanh()
		if self.activationtype == 'sigmoid':
			self.Activation = torch.nn.Sigmoid()
		if self.activationtype == 'none':
			self.Activation = None
		if self.activationtype == 'selu':
			self.Activation = torch.nn.SELU()
		if self.activationtype == 'elu':
			self.Activation = torch.nn.ELU()
		if self.activationtype == 'celu':
			self.Activation = torch.nn.CELU()
		if self.activationtype == 'gelu':
			self.Activation = torch.nn.GELU()
		
		# Loss
		if self.lossfunctype == 'mse':
			self.LossFunction = torch.nn.MSELoss()
		if self.lossfunctype == 'l1':
			self.LossFunction = torch.nn.L1Loss()
		
		# Layers
		if self.Numepochs == 0 or hasattr(self, 'ModelPath'):  # Only at initial training
			# Define Layers at DEVICE
			self.EncoderLayersModule = torch.nn.ModuleList()
			self.DecoderLayersModule = torch.nn.ModuleList()
			for i in range(len(self.encoderdim) - 1):
				self.EncoderLayersModule.append(torch.nn.Linear(self.encoderdim[i], self.encoderdim[i + 1], bias=self.bias, device=self.device, dtype=self.tensordtype))
			for i in range(len(self.decoderdim) - 1):  # Output Layer 제외
				self.DecoderLayersModule.append(torch.nn.Linear(self.decoderdim[i], self.decoderdim[i + 1], bias=self.bias, device=self.device, dtype=self.tensordtype))
			
			if self.batchnorm:
				self.EncoderBatchNormLayersModule = torch.nn.ModuleList()
				self.DecoderBatchNormLayersModule = torch.nn.ModuleList()
				for i in range(len(self.encoderdim) - 1):
					self.EncoderBatchNormLayersModule.append(torch.nn.BatchNorm1d(num_features=self.encoderdim[i + 1], device=self.device, dtype=self.tensordtype))
				for i in range(len(self.decoderdim) - 2):
					self.DecoderBatchNormLayersModule.append(torch.nn.BatchNorm1d(num_features=self.decoderdim[i + 1], device=self.device, dtype=self.tensordtype))
			
			if self.dropout:
				self.DropoutLayer = torch.nn.Dropout(p=self.dropout)
			
			if self.weight_init:
				for i in range(len(self.EncoderLayersModule)):
					torch.nn.init.xavier_uniform_(self.EncoderLayersModule[i].weight)
				for i in range(len(self.DecoderLayersModule)):
					torch.nn.init.xavier_uniform_(self.DecoderLayersModule[i].weight)
			
			if self.bias_init:
				for i in range(len(self.EncoderLayersModule)):
					self.EncoderLayersModule[i].bias.datapath.fill_(0)
				for i in range(len(self.DecoderLayersModule)):
					self.DecoderLayersModule[i].bias.datapath.fill_(0)
		
		self.Numepochs += epochs
		self.AddedNumepochs = epochs
		
		# Optimizer
		if self.optimizertype == 'adam':
			self.Optimizer = torch.optim.Adam(params=self.parameters(), lr=self.learningrate,
											  weight_decay=self.adamWdecay)
		# weight_decay(regularization)
		if self.optimizertype == 'sgd':
			self.Optimizer = torch.optim.SGD(params=self.parameters(), lr=self.learningrate)
			
		if self.optimizertype == 'radam':
			self.Optimizer = torch.optim.RAdam(params=self.parameters(), lr=self.learningrate)
		# momentum
		
		print("Hyperparameters and layer modules are set.")
	
	
	def BaseDataset(self, FullDataset: pd.DataFrame, inputcols: list, outputcols: list, tensordtype: torch.FloatType = torch.float32):
		"""
		Saves MinMax/Gaussian scaling parameters in FCNN() instance.
		:param FullDataset: pd.DataFrame of full dataset.
		:param inputcols: Network input column list.
		:param outputcols: Network output column list.
		:param tensordtype: Decides data type.
		:return: None
		"""
		self.tensordtype = tensordtype
		# Save Raw
		self.DFlength = FullDataset.shape[0]
		self.InputColumns = inputcols
		self.OutputColumns = outputcols
		self.NumInfeatures = len(self.InputColumns)
		self.NumOutfeatures = len(self.OutputColumns)
		
		self.dfMax = FullDataset.max()
		self.dfMin = FullDataset.min()
		self.dfStd = FullDataset.std()
		self.dfMean = FullDataset.mean()
		
		FullDatasetInput = torch.tensor(FullDataset[self.InputColumns].to_numpy(), dtype=self.tensordtype, device='cpu')
		FullDatasetOutput = torch.tensor(FullDataset[self.OutputColumns].to_numpy(), dtype=self.tensordtype,
										 device='cpu')
		
		del FullDataset
		gc.collect()
		
		self.tensorInputMax = FullDatasetInput.max(dim=0).values
		self.tensorInputMin = FullDatasetInput.min(dim=0).values
		self.tensorInputStd = FullDatasetInput.std(dim=0)
		self.tensorInputMean = FullDatasetInput.mean(dim=0)
		
		del FullDatasetInput
		gc.collect()
		
		self.tensorOutputMax = FullDatasetOutput.max(dim=0).values
		self.tensorOutputMin = FullDatasetOutput.min(dim=0).values
		self.tensorOutputStd = FullDatasetOutput.std(dim=0)
		self.tensorOutputMean = FullDatasetOutput.mean(dim=0)
		
		del FullDatasetOutput
		gc.collect()
		
		self.cudaInputMax = self.tensorInputMax.to(self.device)
		self.cudaInputMin = self.tensorInputMin.to(self.device)
		self.cudaInputStd = self.tensorInputStd.to(self.device)
		self.cudaInputMean = self.tensorInputMean.to(self.device)
		self.cudaOutputMax = self.tensorOutputMax.to(self.device)
		self.cudaOutputMin = self.tensorOutputMin.to(self.device)
		self.cudaOutputStd = self.tensorOutputStd.to(self.device)
		self.cudaOutputMean = self.tensorOutputMean.to(self.device)
		
		print(f"Full Dataset Info: Length {self.DFlength} with {self.NumInfeatures} In-features & {self.NumOutfeatures} Out-features.")
	
	def SetDataloader(self, normalizetype: str = 'minmax', batchsize: int = 1, traindata=pd.DataFrame(), validdata=pd.DataFrame(), testdata=pd.DataFrame(), shuffle: bool = True, pin_memory: bool = False):
		"""
		Sets train, validation, test dataloader class.
		:param normalizetype: Normalization method. 'gaussian' or 'minmax'.
		:param batchsize: Number of train data batch size.
		:param traindata: Train dataset.
		:param validdata: Validation dataset.
		:param testdata: Test dataset.
		:param shuffle: Whether to shuffle dataloader. If pd.DataFrame is shuffled, no need to use.
		:return: None
		"""
		self.normalizetype = normalizetype
		self.Numbatchsize = batchsize
		TrainSet = traindata
		ValidSet = validdata
		TestSet = testdata
		Shuffle = shuffle
		PinMemory = pin_memory
		del traindata, validdata, testdata
		gc.collect()
		
		# Normalize
		if self.normalizetype == 'minmax':
			if not TrainSet.empty:
				TrainSet = (TrainSet - self.dfMin) / (self.dfMax - self.dfMin)
			if not ValidSet.empty:
				ValidSet = (ValidSet - self.dfMin) / (self.dfMax - self.dfMin)
			if not TestSet.empty:
				TestSet = (TestSet - self.dfMin) / (self.dfMax - self.dfMin)
		elif self.normalizetype == 'gaussian':
			if not TrainSet.empty:
				TrainSet = (TrainSet - self.dfMean) / self.dfStd
			if not ValidSet.empty:
				ValidSet = (ValidSet - self.dfMean) / self.dfStd
			if not TestSet.empty:
				TestSet = (TestSet - self.dfMean) / self.dfStd
		
		if not TrainSet.empty:
			TrainSet = TrainSet.fillna(0)  # Fill 0
			TrainSet = TensorDataset(
					torch.tensor(TrainSet[self.InputColumns].values, dtype=self.tensordtype,
								 device='cpu'),
					torch.tensor(TrainSet[self.OutputColumns].values, dtype=self.tensordtype,
								 device='cpu'))
			self.TrainSet = DataLoader(TrainSet, batch_size=self.Numbatchsize, shuffle=Shuffle, pin_memory=PinMemory)  # DO NOT PIN MEMORY FOR PYCHARM PARALLEL RUN
		if not ValidSet.empty:
			ValidSet = ValidSet.fillna(0)  # Fill 0
			ValidSet = TensorDataset(
					torch.tensor(ValidSet[self.InputColumns].values, dtype=self.tensordtype, device='cpu'),
					torch.tensor(ValidSet[self.OutputColumns].values, dtype=self.tensordtype, device='cpu'))
			self.ValidSet = DataLoader(ValidSet, batch_size=10000, shuffle=Shuffle, pin_memory=PinMemory)
		if not TestSet.empty:
			TestSet = TestSet.fillna(0)  # Fill 0
			TestSet = TensorDataset(
					torch.tensor(TestSet[self.InputColumns].values, dtype=self.tensordtype, device='cpu'),
					torch.tensor(TestSet[self.OutputColumns].values, dtype=self.tensordtype, device='cpu'))
			self.TestSet = DataLoader(TestSet, batch_size=10000, shuffle=Shuffle, pin_memory=PinMemory)
		
		print("Dataloader ready.")
	
	def SaveModel(self, path: str):
		"""
		Save model file to path.
		:param path: dir1\dir2..\name.pt
		:return: None
		"""
		DP.CreateDir(os.path.dirname(path))
		Dict = {
			# Hyperparms
			'model_state_dict': self.state_dict(),
			'optimizer_state_dict': self.Optimizer.state_dict(),
			'inputcols': self.InputColumns,
			'outputcols': self.OutputColumns,
			'epochs': self.Numepochs,
			# 'nodes': self.Numnodes,
			# 'hiddenlayers': self.Numhiddenlayers,
			'encoderdim': self.encoderdim,
			'decoderdim': self.decoderdim,
			'batchsize': self.Numbatchsize,
			'learningrate': self.learningrate,
			'bias': self.bias,
			'activation': self.activationtype,
			'loss': self.lossfunctype,
			'optimizer': self.optimizertype,
			'adamWdecay': self.adamWdecay,
			'dropout': self.dropout,
			'normalizetype': self.normalizetype,
			'tensordtype': self.tensordtype,
			'batchnorm': self.batchnorm,
			'weight_init': self.weight_init,
			'bias_init': self.bias_init,
			
			# Loss log
			'TrainLossArray': self.TrainLossArray,
			'ValidLossArray': self.ValidLossArray,
			'TrainTime': self.TrainTime,
			
			# Data info
			'DFmax': self.dfMax,
			'DFmin': self.dfMin,
			'DFmean': self.dfMean,
			'DFstd': self.dfStd,
			'tensorInputMax': self.tensorInputMax,
			'tensorInputMin': self.tensorInputMin,
			'tensorInputStd': self.tensorInputStd,
			'tensorInputMean': self.tensorInputMean,
			'tensorOutputMax': self.tensorOutputMax,
			'tensorOutputMin': self.tensorOutputMin,
			'tensorOutputStd': self.tensorOutputStd,
			'tensorOutputMean': self.tensorOutputMean,
			'cudaInputMax': self.cudaInputMax,
			'cudaInputMin': self.cudaInputMin,
			'cudaInputStd': self.cudaInputStd,
			'cudaInputMean': self.cudaInputMean,
			'cudaOutputMax': self.cudaOutputMax,
			'cudaOutputMin': self.cudaOutputMin,
			'cudaOutputStd': self.cudaOutputStd,
			'cudaOutputMean': self.cudaOutputMean
		}
		torch.save(Dict, path)
		print(f"Model saved: {path}\n")
	
	
	def LoadModel(self, path: str, traindata: pd.DataFrame = pd.DataFrame(), validdata: pd.DataFrame = pd.DataFrame(), testdata: pd.DataFrame = pd.DataFrame()):
		"""
		Load model file from path.
		Do not use self.SetDataloader(name.pt) after LoadModel(testdata=TestData). It will overwrite BaseDataset() info.
		Instead, use LoadModel(name.pt, testdata=TestData)
		:param path: dir1\dir2..\name.pt
		:param traindata: :pd.DataFrame
		:param validdata: :pd.DataFrame
		:param testdata: :pd.DataFrame
		:return: None
		"""
		SavePoint = torch.load(path)
		self.ModelPath = path
		
		# Hyperparams
		self.InputColumns = SavePoint['inputcols']
		self.OutputColumns = SavePoint['outputcols']
		self.NumInfeatures = len(self.InputColumns)
		self.NumOutfeatures = len(self.OutputColumns)
		self.Numepochs = SavePoint['epochs']
		# self.Numnodes = SavePoint['nodes']
		# self.Numhiddenlayers = SavePoint['hiddenlayers']
		self.encoderdim = SavePoint['encoderdim']
		self.decoderdim = SavePoint['decoderdim']
		
		self.Numbatchsize = SavePoint['batchsize']
		self.learningrate = SavePoint['learningrate']
		self.bias = SavePoint['bias']
		self.activationtype = SavePoint['activation']
		self.lossfunctype = SavePoint['loss']
		self.adamWdecay = SavePoint['adamWdecay'] if 'adamWdecay' in SavePoint.keys() else 0
		self.dropout = SavePoint['dropout'] if 'dropout' in SavePoint.keys() else 0
		self.normalizetype = SavePoint['normalizetype']
		self.tensordtype = SavePoint['tensordtype'] if 'tensordtype' in SavePoint.keys() else torch.float32
		self.batchnorm = SavePoint['batchnorm'] if 'batchnorm' in SavePoint.keys() else False
		self.weight_init = SavePoint['weight_init'] if 'weight_init' in SavePoint.keys() else False
		self.bias_init = SavePoint['bias_init'] if 'bias_init' in SavePoint.keys() else False
		
		self.TrainLossArray = SavePoint['TrainLossArray']
		self.ValidLossArray = SavePoint['ValidLossArray']
		self.TrainTime = SavePoint['TrainTime'] if 'TrainTime' in SavePoint.keys() else None
		self.optimizertype = SavePoint['optimizer']
		
		# Data Info
		self.dfMax = SavePoint['DFmax']
		self.dfMin = SavePoint['DFmin']
		self.dfMean = SavePoint['DFmean']
		self.dfStd = SavePoint['DFstd']
		
		self.tensorInputMax = SavePoint['tensorInputMax'] if 'tensorInputMax' in SavePoint.keys() else None
		self.tensorInputMin = SavePoint['tensorInputMin'] if 'tensorInputMin' in SavePoint.keys() else None
		self.tensorInputStd = SavePoint['tensorInputStd'] if 'tensorInputStd' in SavePoint.keys() else None
		self.tensorInputMean = SavePoint['tensorInputMean'] if 'tensorInputMean' in SavePoint.keys() else None
		
		self.tensorOutputMax = SavePoint['tensorOutputMax'] if 'tensorOutputMax' in SavePoint.keys() else None
		self.tensorOutputMin = SavePoint['tensorOutputMin'] if 'tensorOutputMin' in SavePoint.keys() else None
		self.tensorOutputStd = SavePoint['tensorOutputStd'] if 'tensorOutputStd' in SavePoint.keys() else None
		self.tensorOutputMean = SavePoint['tensorOutputMean'] if 'tensorOutputMean' in SavePoint.keys() else None
		
		self.cudaInputMax = SavePoint['cudaInputMax'] if 'cudaInputMax' in SavePoint.keys() else None
		self.cudaInputMin = SavePoint['cudaInputMin'] if 'cudaInputMin' in SavePoint.keys() else None
		self.cudaInputStd = SavePoint['cudaInputStd'] if 'cudaInputStd' in SavePoint.keys() else None
		self.cudaInputMean = SavePoint['cudaInputMean'] if 'cudaInputMean' in SavePoint.keys() else None
		self.cudaOutputMax = SavePoint['cudaOutputMax'] if 'cudaOutputMax' in SavePoint.keys() else None
		self.cudaOutputMin = SavePoint['cudaOutputMin'] if 'cudaOutputMin' in SavePoint.keys() else None
		self.cudaOutputStd = SavePoint['cudaOutputStd'] if 'cudaOutputStd' in SavePoint.keys() else None
		self.cudaOutputMean = SavePoint['cudaOutputMean'] if 'cudaOutputMean' in SavePoint.keys() else None
		
		# Reproduce Modules
		self.SetHyperParams(epochs=self.Numepochs, encoderdim=self.encoderdim, decoderdim=self.decoderdim,
							# nodes=self.Numnodes, hiddenlayers=self.Numhiddenlayers,
							batchsize=self.Numbatchsize, lr=self.learningrate, bias=self.bias, dropout=self.dropout,
							activation=self.activationtype, loss=self.lossfunctype, optimizer=self.optimizertype, adamWdecay=self.adamWdecay,
							batchnorm=self.batchnorm, weight_init=self.weight_init, bias_init=self.bias_init)
		self.load_state_dict(SavePoint['model_state_dict'])
		self.Optimizer.load_state_dict(SavePoint['optimizer_state_dict'])
		print(f"Model loaded: {path}")
		
		# Reproduce dataloaders
		self.SetDataloader(normalizetype=self.normalizetype, traindata=traindata, validdata=validdata, testdata=testdata)

if __name__ == '__main__':
	pass

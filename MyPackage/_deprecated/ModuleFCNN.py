import matplotlib.offsetbox
import numpy as np
import pandas as pd
import torch
import time
import os
from torch.utils.data import TensorDataset, DataLoader
import gc
from sklearn.metrics import r2_score as R2
import matplotlib.pyplot as plt
import DataProcessing as DP



class FCNN(torch.nn.Module):
	"""
	For model training:
		Net=FCNN()
		Net.BaseDataset()
		Net.SetHyperParams()
		Net.SetDataloader()
		Net.Train()
		Net.Save()
			FOR ADDITIONAL TRAINING...USE
			Net.SetHyperParams()
			Net.Train()
			Net.Save()
		
	For model prediction:
		Net=FCNN()
		Net.LoadModel(file, testdata=TestData)
		optional: Net.CPU()
		Net.EvaluateTestDataPrediction()
		Net.Predict()
	
	"""
	device = 'cuda' if torch.cuda.is_available() else 'cpu'
	
	def __init__(self):
		super().__init__()
		self.Numepochs = 0
		self.TrainLossArray = []
		self.ValidLossArray = []
		self.ValidR2Array = []
	
	def forward(self, X):
		# Input
		X = self.InputLayer(X)
		if self.batchnorm:
			X = self.BatchNormLayersModule[0](X)
		if self.activationtype != 'none':
			X = self.Activation(X)
		if self.dropout:
			X = self.DropoutLayer(X)
		
		# Hidden
		for idx, hiddenlayer in enumerate(self.FCHiddenLayersModule):
			X = hiddenlayer(X)
			if self.batchnorm:
				X = self.BatchNormLayersModule[idx + 1](X)
			if self.activationtype != 'none':
				X = self.Activation(X)
			if self.dropout:
				X = self.DropoutLayer(X)
		
		# Output
		X = self.OutputLayer(X)
		return X
	
	def FixSeed(self, seednum=777):
		"""
		Fixes all possible random/stochastic processes to specific seed number. This makes network training reproducible.
		:param seednum: Seed number to use.
		:return: None
		"""
		self.seednum = seednum
		np.random.seed(seednum)
		torch.manual_seed(seednum)
		torch.cuda.manual_seed(seednum)
		torch.Generator().manual_seed(seednum)  # Data split
		torch.backends.cudnn.deterministic = True  # BackPropagation
		print(f"Fixed all stochastic seeds to {seednum}.")
	
	def Initializer(self, layer):
		if isinstance(layer, (torch.nn.Linear)):
			torch.nn.init.xavier_normal_(layer.weight)
			torch.nn.init.constant_(layer.bias, 0)
	
	def SetHyperParams(self, epochs: int = 1, nodes: int = 1, hiddenlayers: int = 1, lr: float = 1e-3, bias: bool = True,
					   activation: str = 'relu', loss: str = 'mse', optimizer: str = 'adam',
					   adamWdecay: float = 0, l2_lambda: float = 0, dropout: float = 0, batchnorm: bool = False,
					   param_init: bool = False):
		"""
		:param epochs: Epochs to add.
		:param nodes: Number of nodes in fully-connected hidden layers.
		:param hiddenlayers: Number of hidden layers.
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
		
		# Layers
		# Only at initial training
		# Only at initial training
		if self.Numepochs == 0 or hasattr(self, 'ModelPath'):  # Only at initial training
			
			self.Numnodes = nodes
			self.Numhiddenlayers = hiddenlayers
			self.learningrate = lr
			self.bias = bias
			self.dropout = dropout
			self.batchnorm = batchnorm
			self.param_init = param_init
			self.activationtype = activation.lower()
			self.lossfunctype = loss.lower()
			self.optimizertype = optimizer.lower()
			self.adamWdecay = adamWdecay
			self.l2_lambda = l2_lambda
			
			# Define Layers at DEVICE
			self.InputLayer = torch.nn.Linear(self.NumInfeatures, self.Numnodes, bias=self.bias, device=self.device,
											  dtype=self.tensordtype)
			self.FCHiddenLayersModule = torch.nn.ModuleList()
			self.BatchNormLayersModule = torch.nn.ModuleList()
			for i in range(self.Numhiddenlayers):
				self.FCHiddenLayersModule.add_module(f"HiddenLayer{i + 1:d}",
													 torch.nn.Linear(self.Numnodes, self.Numnodes, bias=self.bias,
																	 device=self.device, dtype=self.tensordtype))
			if self.batchnorm:
				for i in range(self.Numhiddenlayers + 1):
					self.BatchNormLayersModule.append(torch.nn.BatchNorm1d(num_features=self.Numnodes,
																		   device=self.device, dtype=self.tensordtype))
			
			self.OutputLayer = torch.nn.Linear(self.Numnodes, self.NumOutfeatures, device=self.device,
											   dtype=self.tensordtype)
			self.DropoutLayer = torch.nn.Dropout(p=self.dropout)
			
			if self.param_init:
				self.apply(self.Initializer)
			
			# Nonlinear activation
			if self.activationtype == 'none':
				self.Activation = None
			if self.activationtype == 'relu':
				self.Activation = torch.nn.ReLU()
			if self.activationtype == 'leakyrelu':
				self.Activation = torch.nn.LeakyReLU()
			if self.activationtype == 'tanh':
				self.Activation = torch.nn.Tanh()
			if self.activationtype == 'sigmoid':
				self.Activation = torch.nn.Sigmoid()
			if self.activationtype == 'selu':
				self.Activation = torch.nn.SELU()
			if self.activationtype == 'elu':
				self.Activation = torch.nn.ELU()
			if self.activationtype == 'celu':
				self.Activation = torch.nn.CELU()
			if self.activationtype == 'gelu':
				self.Activation = torch.nn.GELU()
			if self.activationtype == 'mish':
				self.Activation = torch.nn.Mish()
			if activation.lower() == 'silu':
				self.Activation = torch.nn.SiLU()
			print("Model initialized.")
			
			# Loss
			if self.lossfunctype == 'mse':
				self.LossFunction = torch.nn.MSELoss()
			if self.lossfunctype == 'l1':
				self.LossFunction = torch.nn.L1Loss()
			
			# Optimizer
			if self.optimizertype == 'adam':
				self.Optimizer = torch.optim.Adam(params=self.parameters(), lr=self.learningrate, weight_decay=self.adamWdecay)
			# weight_decay(regularization)
			if self.optimizertype == 'sgd':
				self.Optimizer = torch.optim.SGD(params=self.parameters(), lr=self.learningrate, momentum=0)
			
			if self.optimizertype == 'radam':
				self.Optimizer = torch.optim.RAdam(params=self.parameters(), lr=self.learningrate, weight_decay=self.adamWdecay)
			
			if self.optimizertype == 'nadam':
				self.Optimizer = torch.optim.NAdam(params=self.parameters(), lr=self.learningrate, weight_decay=self.adamWdecay)
			
			if self.optimizertype == 'adamax':
				self.Optimizer = torch.optim.Adamax(params=self.parameters(), lr=self.learningrate, weight_decay=self.adamWdecay)
			
			if self.optimizertype == 'adamw':
				self.Optimizer = torch.optim.AdamW(params=self.parameters(), lr=self.learningrate, weight_decay=self.adamWdecay)
			
			if self.optimizertype == 'lbfgs':
				self.Optimizer = torch.optim.LBFGS(params=self.parameters(), lr=self.learningrate)
		
		if hasattr(self, 'Optimizer'):
			old_learningrate = self.learningrate
			self.learningrate = lr
			for optimparam in self.Optimizer.param_groups:
				optimparam['lr'] = self.learningrate
			print(f"Learning rate changed from {old_learningrate:.2e} to {self.learningrate:.2e}")
		
		self.Numepochs += epochs
		self.AddedNumepochs = epochs
	
	def BaseDataset(self, FullDataset: pd.DataFrame = pd.DataFrame(), inputcols: list = [], outputcols: list = [], tensordtype: torch.FloatType = torch.float32):
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
	
	
	def Train(self, printrate: int = 1, printbatch: bool = False, earlystopR2: float = 999, lr_halflife:int=999):
		"""
		Train FCNN() instance with given dataloader and hyperparameters.
		:param printrate: if epoch % printrate == 0: print(Loss, R2(if self.ValidSet exists))
		:param printbatch: Prints batch training process.
		:param printgrad: if True, prints network parameters and backward gradients
		:return:
		"""
		self.SetMode('train')
		printbatchrate = int(len(self.TrainSet) / 3)
		if hasattr(self, 'seednum'):
			self.FixSeed(self.seednum)
		if self.device == 'cuda':
			self.CUDA()
		elif self.device == 'cpu':
			self.CPU()
		
		print(f'Starting model training with device "{self.device.upper()}"')
		
		StartTime = time.time()
		for epoch in range(self.AddedNumepochs):
			TrainBatchesLabel = []
			TrainBatchesPrediction = []
			for idx, batch in enumerate(self.TrainSet):
				trainx, tlabel = batch
				# Send2GPU
				trainx = trainx.to(self.device)
				tlabel = tlabel.to(self.device)
				
				if self.optimizertype == 'lbfgs':
					tprediction = self.forward(trainx)
					Loss = self.LossFunction(tlabel, tprediction)
					
					def Closure():
						self.Optimizer.zero_grad(set_to_none=False)
						tprediction = self.forward(trainx)
						Loss = self.LossFunction(tlabel, tprediction)
						
						if self.l2_lambda:
							l2reg = torch.tensor(0, device=self.device, dtype=self.tensordtype)
							for name,param in self.named_parameters():
								if 'weight' in name:
									l2reg+=torch.sum(torch.square(param))
							Loss += self.l2_lambda * l2reg
						
						Loss.backward()
						return Loss
					self.Optimizer.step(Closure)
				
				else:
					self.Optimizer.zero_grad(set_to_none=False)
					tprediction = self.forward(trainx)
					Loss = self.LossFunction(tlabel, tprediction)
					
					if self.l2_lambda:
						l2reg = 0
						for param in self.parameters():
							l2reg += torch.linalg.norm(param, ord=2)
						l2reg *= self.l2_lambda
						Loss += l2reg
					
					Loss.backward()
					self.Optimizer.step()
				
				TrainBatchesLabel.append(tlabel.cpu())
				TrainBatchesPrediction.append(tprediction.cpu())
				
				# Delete from gpu memory
				del trainx, tlabel, tprediction, batch
				torch.cuda.empty_cache()
				torch.cuda.synchronize()
				gc.collect()
				torch.cuda.synchronize()
				
				# if idx == 0 and printbatch:
				#     print(f"\nEpoch {i + 1} batch training")
				if printbatch and (idx + 1) % printbatchrate == 0:
					print(f"Batch {idx + 1}/{len(self.TrainSet)} in process")
			
			# Calculate Entire Train Loss
			TrainBatchesLabel = torch.concat(TrainBatchesLabel, dim=0)
			TrainBatchesPrediction = torch.concat(TrainBatchesPrediction, dim=0)
			tLoss = self.LossFunction(TrainBatchesLabel, TrainBatchesPrediction)
			self.TrainLossArray.append(tLoss.detach().item())  # DETACH GRADIENT WHEN APPENDING!!
			
			if (epoch + 1) % printrate == 0:
				print(f"Epoch {epoch + 1} Train Loss={tLoss:.10f}")
			
			if hasattr(self, 'ValidSet'):
				with torch.no_grad():
					self.SetMode('eval')
					ValidBatchesLabel = []
					ValidBatchesPrediction = []
					for batch in self.ValidSet:
						validx, vlabel = batch
						
						# Send2GPU
						validx = validx.to(self.device)
						vlabel = vlabel.to(self.device)
						
						vprediction = self.forward(validx)
						vLoss = self.LossFunction(vlabel, vprediction)
						
						# for R2 Calc, to cpu
						ValidBatchesLabel.append(vlabel.detach().cpu())
						ValidBatchesPrediction.append(vprediction.detach().cpu())
						
						# Delete from gpu memory
						del validx, vlabel, vprediction, vLoss, batch
						torch.cuda.empty_cache()
						gc.collect()
					
					# Calculate Entire Validation Loss
					ValidBatchesLabel = torch.cat(ValidBatchesLabel, dim=0)
					ValidBatchesPrediction = torch.cat(ValidBatchesPrediction, dim=0)
					vLoss = self.LossFunction(ValidBatchesLabel, ValidBatchesPrediction)
					self.ValidLossArray.append(vLoss.detach().item())
					
					# Record Validation R2
					if not len(ValidBatchesLabel) <= 1:  # Numbatch<=1: R2 calculation err
						R2score = R2(ValidBatchesLabel, ValidBatchesPrediction, multioutput='raw_values')
						self.ValidR2Array.append(R2score)
					else:
						pass
					
					# Print Validation Loss, R2
					if (epoch + 1) % printrate == 0:
						print(f"Validation Loss: {vLoss:.10f}")
						if self.l2_lambda:
							print(f"Regularizer: {l2reg:.6f}")
						if not len(ValidBatchesLabel) <= 1:  # Numbatch<=1: R2 calculation err
							for idx, outputcolumn in enumerate(self.OutputColumns):
								print(f"R2({outputcolumn}) = {self.ValidR2Array[-1][idx]:.6f}")
							
							print()
						else:
							print()
					
					del ValidBatchesLabel, ValidBatchesPrediction, vLoss
					torch.cuda.empty_cache()
					torch.cuda.synchronize()
					gc.collect()
					self.SetMode('train')
			
			earlystop_checker = []
			for r2value in self.ValidR2Array[-1]:
				if r2value >= earlystopR2:
					earlystop_checker.append('pass')
				else:
					earlystop_checker.append('fail')
			if not 'fail' in earlystop_checker:
				self.Numepochs = self.Numepochs - (self.AddedNumepochs - (epoch + 1))
				break
			if (epoch+1)%lr_halflife==0:
				self.Optimizer.param_groups[0]['lr']/=2
		
		EndTime = time.time()
		if not hasattr(self, 'TrainTime'):
			self.TrainTime = EndTime - StartTime
		else:
			self.TrainTime += EndTime - StartTime
		
		dummyTrainTime = DP.Sec2Time(self.TrainTime)
		
		print(f"{epoch + 1} Epochs done.")
		for idx, finalr2value in enumerate(self.ValidR2Array[-1]):
			print(f"R2({self.OutputColumns[idx]}) = {finalr2value:.6f}")
		print(f"Model training finished in {dummyTrainTime[0]}hr {dummyTrainTime[1]}min {int(dummyTrainTime[2])}sec.")
		print()
	
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
			'nodes': self.Numnodes,
			'hiddenlayers': self.Numhiddenlayers,
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
			'param_init': self.param_init,
			'l2_lambda': self.l2_lambda,
			
			# Train log
			'TrainLossArray': self.TrainLossArray,
			'ValidLossArray': self.ValidLossArray,
			'TrainTime': self.TrainTime if hasattr(self, 'TrainTime') else 0,
			'ValidR2Array': self.ValidR2Array,
			
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
		self.Numnodes = SavePoint['nodes']
		self.Numhiddenlayers = SavePoint['hiddenlayers']
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
		self.param_init = SavePoint['param_init'] if 'param_init' in SavePoint.keys() else False
		self.l2_lambda = SavePoint['l2_lambda'] if 'l2_lambda' in SavePoint.keys() else 0
		
		self.TrainLossArray = SavePoint['TrainLossArray']
		self.ValidLossArray = SavePoint['ValidLossArray']
		self.TrainTime = SavePoint['TrainTime'] if 'TrainTime' in SavePoint.keys() else None
		self.ValidR2Array = SavePoint['ValidR2Array'] if 'ValidR2Array' in SavePoint.keys() else None
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
		# if epochs=self.Numepochs:
		# 		self.Numepochs = self.Numepochs + self.Numepochs....
		self.SetHyperParams(epochs=0, nodes=self.Numnodes, hiddenlayers=self.Numhiddenlayers,
							lr=self.learningrate, bias=self.bias, dropout=self.dropout,
							activation=self.activationtype, loss=self.lossfunctype, optimizer=self.optimizertype, adamWdecay=self.adamWdecay,
							batchnorm=self.batchnorm, param_init=self.param_init)
		self.load_state_dict(SavePoint['model_state_dict'])
		self.Optimizer.load_state_dict(SavePoint['optimizer_state_dict'])
		
		print(f"Model loaded: {path}")
		
		# Reproduce dataloaders
		self.SetDataloader(normalizetype=self.normalizetype, batchsize=self.Numbatchsize,
						   traindata=traindata, validdata=validdata, testdata=testdata)
	
	def Describe(self):
		"""
		Describes network hyperparameters and properties.
		:return: None
		"""
		print("==========================================================")
		print("==========================================================")
		print(f"{self.NumInfeatures} Input Columns: {self.InputColumns}")
		print(f"{self.NumOutfeatures} Output Columns: {self.OutputColumns}")
		print(f"Epochs: {self.Numepochs}")
		print(f"Nodes: {self.Numnodes}")
		print(f"FC-hidden layers: {self.Numhiddenlayers}")
		print(f"Train batch size: {self.Numbatchsize}")
		print(f"Learning rate: {self.learningrate:.1e}")
		print(f"Use bias in layers: {self.bias}")
		print(f"Activation function: {self.activationtype.upper()}")
		print(f"Loss function: {self.lossfunctype.upper()}")
		print(f"Optimizer type: {self.optimizertype.upper()}")
		print(f"ADAM weight decay: {self.adamWdecay}")
		print(f"L2 regularizer lambda: {self.l2_lambda}")
		print(f"Dropout probability in hidden layers: {self.dropout}")
		print(f"Apply batch normalization: {self.batchnorm}")
		print(f"Apply parameter initialization: {self.param_init}")
		print(f"Normalization type: {self.normalizetype.upper()}")
		print("==========================================================")
		print("==========================================================")
	
	def SetMode(self, mode: str):
		"""
		Set network mode 'eval' or 'train'.
		There's no need to call this method before Train(), Predict(), and EvaluateTestDataPrediction() manually...
		They automatically call this method at initial.
		:param mode: 'eval' or 'train'.
		:return:
		"""
		if mode.lower() == 'train':
			self.train()
			self.mode = 'train'
			self.requiresgrad = True
			for name, param in self.named_parameters():
				param.requires_grad = self.requiresgrad
		elif mode.lower() == 'eval':
			self.eval()
			self.mode = 'eval'
			self.dropout = 0
			self.requiresgrad = False
			for name, param in self.named_parameters():
				param.requires_grad = self.requiresgrad
	
	def EvaluateTestDataPredictions(self, inverse: bool = True):
		"""
		Evaluate R2 and RMSE based on the self.TestSet(torch.utils.data.DataLoader class).
		:param inverse: If True, output prediction is inverse-normalized. Inverse normalized RMSE will be calculated. R2 will not be affected.
		:return: None
		"""
		
		self.SetMode('eval')
		with torch.no_grad():
			self.SetMode('eval')
			TestX = []
			TestLabel = []
			TestPrediction = []
			for batch in self.TestSet:
				testx, testlabel = batch
				
				testx = testx.to(self.device)
				testlabel = testlabel.to(self.device)
				testprediction = self.forward(testx)
				
				TestX.append(testx)
				TestLabel.append(testlabel)
				TestPrediction.append(testprediction)
				
				del testx, testlabel, testprediction
				torch.cuda.empty_cache()
				torch.cuda.synchronize()
				gc.collect()
			
			TestX = torch.concat(TestX, dim=0)
			TestLabel = torch.concat(TestLabel, dim=0)
			TestPrediction = torch.concat(TestPrediction, dim=0)
			
			TestLabel = TestLabel.to('cpu').numpy()
			TestLabel = pd.DataFrame(TestLabel, columns=self.OutputColumns)
			
			TestPrediction = TestPrediction.to('cpu').numpy()
			TestPrediction = pd.DataFrame(TestPrediction, columns=self.OutputColumns)
			
			if inverse:
				if self.normalizetype == 'minmax':
					TestLabel = TestLabel * (self.dfMax[self.OutputColumns] - self.dfMin[self.OutputColumns]) + self.dfMin[self.OutputColumns]
					TestPrediction = TestPrediction * (self.dfMax[self.OutputColumns] - self.dfMin[self.OutputColumns]) + self.dfMin[self.OutputColumns]
				elif self.normalizetype == 'gaussian':
					TestLabel = TestLabel * self.dfStd[self.OutputColumns] + self.dfMean[self.OutputColumns]
					TestPrediction = TestPrediction * self.dfStd[self.OutputColumns] + self.dfMean[self.OutputColumns]
		
		print(self.ModelPath)
		DP.PrintR2(TestLabel, TestPrediction)
		DP.PrintRMSE(TestLabel, TestPrediction)
	
	def Predict(self, Input, cuda: bool = True, inverse_input: bool = True, inverse_output: bool = True, return_df: bool = True):
		"""
		Returns network prediction(output) of Input data.
		:param Input: pd.DataFrame or torch.Tensor, input shape should match network input dimension
		:param inverse_input: If False, input normalization is passed.
		:param inverse_output: If False, output normalization is passed.
		:param return_df: If True, returns pd.DataFrame output. If False, returns torch.Tensor output.
		:param cuda: If True, assumes network weights and bias are on GPU device. Before setting this param to False, implement self.CPU().
		:return: pd.DataFrame or torch.Tensor
		"""
		self.SetMode('eval')
		PredTime = 0
		NumPredictions = Input.shape[0]
		InverseInput = inverse_input
		InverseOutput = inverse_output
		ReturnDF = return_df
		CUDA = cuda
		if not CUDA:
			self.CPU()
		if CUDA:
			self.CUDA()
		
		if type(Input) == pd.DataFrame:
			Input = Input.reset_index(drop=True)
			if InverseInput:
				# Normalize
				if self.normalizetype == 'minmax':
					Input = (Input - self.dfMin[self.InputColumns]) / (
							self.dfMax[self.InputColumns] - self.dfMin[self.InputColumns])
				elif self.normalizetype == 'gaussian':
					Input = (Input - self.dfMean[self.InputColumns]) / self.dfStd[self.InputColumns]
			# Casting
			Input = torch.tensor(Input.values, device='cpu', dtype=self.tensordtype)
			Input = TensorDataset(Input)
			Input = DataLoader(Input, batch_size=100000)
		
		elif type(Input) == torch.Tensor:
			if InverseInput:
				# Normalize
				if self.normalizetype == 'minmax':
					Input = (Input - self.tensorInputMin) / (self.tensorInputMax - self.tensorInputMin)
				elif self.normalizetype == 'gaussian':
					Input = (Input - self.tensorInputMean) / self.tensorInputStd
			
			Input = TensorDataset(Input)
			Input = DataLoader(Input, batch_size=100000)
		
		if ReturnDF:
			# Predict
			DFlist = []
			with torch.no_grad():
				for batch in Input:
					inputbatch = batch[0]  # unpack
					if CUDA:
						inputbatch = inputbatch.to(self.device)
					StartTime = time.time()
					Output = self.forward(inputbatch)
					EndTime = time.time()
					PredTime += EndTime - StartTime
					if CUDA:
						Output = Output.to('cpu')
					Output = pd.DataFrame(Output.numpy(), columns=self.OutputColumns)
					DFlist.append(Output)
					del inputbatch
					if CUDA:
						torch.cuda.empty_cache()
						torch.cuda.synchronize()
			Output = pd.concat(DFlist, axis=0, ignore_index=True)
		
		if not ReturnDF:
			# Predict
			tensorList = []
			with torch.no_grad():
				for batch in Input:
					inputbatch = batch[0]  # unpack
					if CUDA:
						inputbatch = inputbatch.to(self.device)
					StartTime = time.time()
					Output = self.forward(inputbatch)
					EndTime = time.time()
					PredTime += EndTime - StartTime
					if CUDA:
						Output = Output.to('cpu')
					tensorList.append(Output)
					del inputbatch
					if CUDA:
						torch.cuda.empty_cache()
						torch.cuda.synchronize()
			Output = torch.concat(tensorList, dim=0)
		
		if InverseOutput:
			if type(Output) == pd.DataFrame:
				# Inverse Normalize
				if self.normalizetype == 'minmax':
					Output = Output * (self.dfMax[self.OutputColumns] - self.dfMin[self.OutputColumns]) + self.dfMin[
						self.OutputColumns]
				elif self.normalizetype == 'gaussian':
					Output = Output * self.dfStd[self.OutputColumns] + self.dfMean[self.OutputColumns]
			elif type(Output) == torch.Tensor:
				# Inverse Normalize
				if self.normalizetype == 'minmax':
					Output = Output * (self.tensorOutputMax - self.tensorOutputMin) + self.tensorOutputMin
				elif self.normalizetype == 'gaussian':
					Output = Output * self.tensorOutputStd + self.tensorOutputMean
		
		print(f"{NumPredictions} predictions took {PredTime:.4f}sec.\n")
		return Output
	
	def CPU(self):
		"""
		Moves all network parameters to CPU.
		:return: None
		"""
		self.ParamLoc = 'cpu'
		self.cpu()
		Counter = 0
		for param in self.parameters():
			Counter += 1
		torch.cuda.empty_cache()
		print(f"{Counter} model parameter tensors moved to \"CPU\".")
	
	def CUDA(self):
		"""
		Moves all network parameters to CUDA.
		:return: None
		"""
		self.ParamLoc = 'cuda'
		self.cuda(device=self.device)
		Counter = 0
		for param in self.parameters():
			Counter += 1
		print(f"{Counter} model parameter tensors moved to \"{self.device.upper()}\".")










if __name__ == '__main__':
	pass

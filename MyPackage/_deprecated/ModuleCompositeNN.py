import torch


class LearnableParams(torch.nn.Module):
	
	def __init__(self):
		super(LearnableParams, self).__init__()
	
	def RegisterAlpha(self, init_value: float = 0, size: tuple = (1, 1)):
		sample = torch.empty(size)
		self.alpha = torch.full_like(sample, init_value, dtype=torch.float32, requires_grad=True)
		self.alpha = torch.nn.Parameter(self.alpha)
		self.register_parameter('alpha', self.alpha)
	
	def RegisterBeta(self, init_value: float = 0, size: tuple = (1, 1)):
		sample = torch.empty(size)
		self.beta = torch.full_like(sample, init_value, dtype=torch.float32, requires_grad=True)
		self.beta = torch.nn.Parameter(self.beta)
		self.register_parameter('beta', self.beta)
	
	def RegisterGamma(self, init_value: float = 0, size: tuple = (1, 1)):
		sample = torch.empty(size)
		self.gamma = torch.full_like(sample, init_value, dtype=torch.float32, requires_grad=True)
		self.gamma = torch.nn.Parameter(self.gamma)
		self.register_parameter('gamma', self.gamma)


class Unit(torch.nn.Module):
	def __init__(self, in_N: int, out_N: int, activation: str = 'tanh', batchnorm: bool = False, dropout: float = 0):
		super(Unit, self).__init__()
		self.in_N = in_N
		self.out_N = out_N
		self.activation = activation
		self.Layer = torch.nn.Linear(self.in_N, self.out_N, bias=True)
		self.batchnorm = batchnorm
		self.dropout = dropout
		
		if activation.lower() == 'tanh':
			self.Activation = torch.nn.Tanh()
		if activation.lower() == 'relu':
			self.Activation = torch.nn.ReLU()
		if activation.lower() == 'gelu':
			self.Activation = torch.nn.GELU()
		if activation.lower() == 'silu':
			self.Activation = torch.nn.SiLU()
		if activation.lower() == 'mish':
			self.Activation = torch.nn.Mish()
		if activation.lower() == 'none':
			self.Activation = None
		if activation.lower() == 'elu':
			self.Activation = torch.nn.ELU()
		
		if self.batchnorm:
			self.BatchNorm = torch.nn.BatchNorm1d(num_features=self.out_N)
		if self.dropout:
			self.Dropout = torch.nn.Dropout(p=self.dropout)
	
	def forward(self, x):
		x = self.Layer(x)
		if self.batchnorm:
			x = self.BatchNorm(x)
		if self.Activation:
			x = self.Activation(x)
		if self.dropout:
			x = self.Dropout(x)
		return x

class StackedNN(torch.nn.Module):
	def __init__(self, in_N: int, depth: int, width: int, out_N: int, activation: str = 'gelu', param_init: str = '',
				 batchnorm: bool = True, dropout: float = 0):
		super(StackedNN, self).__init__()
		self.in_N = in_N
		self.width = width
		self.depth = depth
		self.out_N = out_N
		self.activation = activation
		self.param_init = param_init
		self.batchnorm = batchnorm
		self.dropout = dropout
		
		self.Layers = torch.nn.ModuleList()
		
		self.Layers.append(Unit(self.in_N, self.width, activation=self.activation, batchnorm=self.batchnorm, dropout=self.dropout))
		for i in range(self.depth):
			self.Layers.append(Unit(width, width, activation=self.activation, batchnorm=self.batchnorm, dropout=self.dropout))
		self.Layers.append(torch.nn.Linear(self.width, self.out_N))
		
		if self.param_init:
			self.apply(self.initializer)
	
	def initializer(self, layer):
		if isinstance(layer, (torch.nn.Linear)):
			# Xavier init
			# torch.nn.init.xavier_normal_(layer.weight)
			
			# He init
			torch.nn.init.kaiming_normal_(layer.weight, nonlinearity='relu')
			torch.nn.init.constant_(layer.bias, 0)
	
	def NumParams(self):
		num_params=0
		for param in self.parameters():
			if len(param.data.shape)==1:
				num_params+=param.data.shape[0]
			if len(param.data.shape)==2:
				num_params+=param.data.shape[0]*param.data.shape[1]
		print(f"Number of trainable parameters: {num_params}")
	
	def forward(self, x):
		for layer in self.Layers:
			x = layer.forward(x)
		return x

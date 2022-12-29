import pandas as pd
from _deprecated.ModuleDNN import DNN
import torch
import numpy as np
from sklearn.metrics import r2_score
from torch.utils.data import TensorDataset, DataLoader
from CommonVariables import *
import time, os
from _deprecated import DataProcessing as DP
import warnings

class Module_PINN(torch.nn.Module):
    def __init__(self, in_N: int = 1, depth: int = 2, width: int = 2, out_N: int = 1,
                 activation: str = 'gelu', param_init: str = 'pinn',
                 batchnorm: bool = False, dropout: float = 0):
        ################## model info ##################
        vars = locals()
        if 'self' in vars.keys(): del vars['self']
        if '__class__' in vars.keys(): del vars['__class__']
        self.model_info = vars
        self.model_info.update(vars)
        self.model_info = {key: value for key, value in sorted(self.model_info.items())}
        ################## model info ##################
        
        super(Module_PINN, self).__init__()
        
        self.PINN = DNN(in_N, depth, width, out_N,
                        activation=activation, param_init=param_init,
                        batchnorm=batchnorm, dropout=dropout)
        
        # self.model_info = dict(in_N=in_N, depth=depth, width=width, out_N=out_N,
        #                        activation=activation, param_init=param_init,
        #                        batchnorm=batchnorm, dropout=dropout)
    
    def MSE(self, x):
        return torch.mean(torch.square(x))

    def RMS(self, label, prediction):
        rms_label = torch.sqrt(torch.mean(torch.square(label)))
        rms_prediction = torch.sqrt(torch.mean(torch.square(prediction)))
        rms_error = torch.abs(rms_label - rms_prediction) / rms_label
        return rms_error
    
    def MAE(self, x):
        return torch.mean(torch.abs(x))
    
    def count_params(self):
        num_params = 0
        for param in self.parameters():
            if len(param.data.shape) == 1:
                num_params += param.data.shape[0]
            if len(param.data.shape) == 2:
                num_params += param.data.shape[0] * param.data.shape[1]
        print(f"Number of trainable parameters: {num_params}")
        return num_params
    
    def forward(self, x):
        yDDot = self.PINN.forward(x)
        return yDDot
    
    def setup_dataloader(self, TrainData: pd.DataFrame, ValidData: pd.DataFrame,
                         batch_size: int, metamodeling: bool = False,
                         pin_memory: bool = False, num_workers: int = 0):
        if num_workers:
            persistent_workers = True
        else:
            persistent_workers = False
        ################## model info ##################
        vars = locals()
        if 'self' in vars.keys(): del vars['self']
        if '__class__' in vars.keys(): del vars['__class__']
        if 'TrainData' in vars.keys(): del vars['TrainData']
        if 'ValidData' in vars.keys(): del vars['ValidData']
        self.model_info.update(vars)
        self.model_info = {key: value for key, value in sorted(self.model_info.items())}
        ################## model info ##################
        
        Train_x = torch.FloatTensor(TrainData[t].values)
        Train_y = torch.FloatTensor(TrainData[y].values)
        Train_yDot = torch.FloatTensor(TrainData[yDot].values)
        Train_yDDot = torch.FloatTensor(TrainData[yDDot].values)
        
        Valid_x = torch.FloatTensor(ValidData[t].values)
        Valid_y = torch.FloatTensor(ValidData[y].values)
        Valid_yDot = torch.FloatTensor(ValidData[yDot].values)
        Valid_yDDot = torch.FloatTensor(ValidData[yDDot].values)
        
        self.mean_x = Train_x.mean(dim=0).cuda()
        self.mean_y = Train_y.mean(dim=0).cuda()
        self.mean_yDot = Train_yDot.mean(dim=0).cuda()
        self.mean_yDDot = Train_yDDot.mean(dim=0).cuda()
        
        self.std_x = Train_x.std(dim=0).cuda()
        self.std_y = Train_y.std(dim=0).cuda()
        self.std_yDot = Train_yDot.std(dim=0).cuda()
        self.std_yDDot = Train_yDDot.std(dim=0).cuda()
        
        ################## model info ##################
        self.model_info['mean_x'] = self.mean_x
        self.model_info['mean_y'] = self.mean_y
        self.model_info['mean_yDot'] = self.mean_yDot
        self.model_info['mean_yDDot'] = self.mean_yDDot
        self.model_info['std_x'] = self.std_x
        self.model_info['std_y'] = self.std_y
        self.model_info['std_yDot'] = self.std_yDot
        self.model_info['std_yDDot'] = self.std_yDDot
        
        if metamodeling:
            Train_params = torch.FloatTensor(TrainData[params].values)
            Valid_params = torch.FloatTensor(ValidData[params].values)
            self.mean_params = Train_params.mean(dim=0).cuda()
            self.std_params = Train_params.std(dim=0).cuda()
            self.model_info['mean_params'] = self.mean_params
            self.model_info['std_params'] = self.std_params
            
            train_dataloader = TensorDataset(Train_x, Train_params, Train_y, Train_yDot, Train_yDDot)
            valid_dataloader = TensorDataset(Valid_x, Valid_params, Valid_y, Valid_yDot, Valid_yDDot)
        else:
            train_dataloader = TensorDataset(Train_x, Train_y, Train_yDot, Train_yDDot)
            valid_dataloader = TensorDataset(Valid_x, Valid_y, Valid_yDot, Valid_yDDot)
        ################## model info ##################
        
        self.train_dataloader = DataLoader(train_dataloader, batch_size=batch_size,
                                           shuffle=True, pin_memory=pin_memory, num_workers=num_workers,
                                           persistent_workers=persistent_workers)
        self.valid_dataloader = DataLoader(valid_dataloader, batch_size=batch_size,
                                           shuffle=False, pin_memory=pin_memory, num_workers=num_workers,
                                           persistent_workers=persistent_workers)
    
    def setup_optimizer(self, initial_lr: float = 1e-3, optimizer: str = 'radam'):
        optimizer = optimizer.lower()
        ################## model info ##################
        vars = locals()
        if 'self' in vars.keys(): del vars['self']
        if '__class__' in vars.keys(): del vars['__class__']
        self.model_info.update(vars)
        self.model_info = {key: value for key, value in sorted(self.model_info.items())}
        ################## model info ##################
        
        dict1 = {'params': self.PINN.parameters(), 'lr': initial_lr}
        if optimizer == 'radam':
            self.optimizer1 = torch.optim.RAdam([dict1])
        elif optimizer == 'lbfgs':
            self.optimizer1 = torch.optim.LBFGS([dict1])
    
    def setup_loss_fn(self, loss_fn: str = 'mse'):
        loss_fn = loss_fn.lower()
        ################## model info ##################
        vars = locals()
        if 'self' in vars.keys(): del vars['self']
        if '__class__' in vars.keys(): del vars['__class__']
        self.model_info.update(vars)
        self.model_info = {key: value for key, value in sorted(self.model_info.items())}
        ################## model info ##################
        
        loss_fns = {'mse': torch.nn.MSELoss(),
                    'mae': torch.nn.L1Loss(),
                    'smoothl1': torch.nn.SmoothL1Loss(beta=1),
                    'huber': torch.nn.HuberLoss(delta=1)}
        loss_fn = loss_fn.lower()
        self.loss_fn = loss_fns[loss_fn]
    
    def fit(self, epochs, lr_halflife, filename, save: bool = True, save_every: int = 1, reg_lambda: float = 0,
            initial_lr: float = 1e-3, loss_fn: str = 'mse', optimizer: str = 'radam', print_every: int = 10, valid_measure: str = 'mse'):
        ################## model info ##################
        vars = locals()
        if 'self' in vars.keys(): del vars['self']
        if '__class__' in vars.keys(): del vars['__class__']
        self.model_info.update(vars)
        self.model_info = {key: value for key, value in sorted(self.model_info.items())}
        ################## model info ##################
        self.setup_optimizer(initial_lr=initial_lr, optimizer=optimizer)
        self.setup_loss_fn(loss_fn)
        
        for key, value in self.model_info.items():
            print(f"{key.upper()}: {value}")
        print()
        
        for module in self.children():
            print(module)
        print()
        
        tLoss_yDDot_history = np.empty(epochs)
        vLoss_yDDot_history = np.empty(epochs)
        model_parameter_group = np.empty(epochs, dtype=object)
        model_validation_loss = np.full(epochs, 1e30)
        TrainingStartTime = time.time()
        
        for epoch in range(self.model_info['epochs']):
            
            # Training step
            tLabel = []
            tPrediction = []
            epoch_computation_time = 0
            for trainbatch_idx, trainbatch in enumerate(self.train_dataloader):
                batch_start = time.time()
                # Forward
                def closure():
                    if self.model_info['metamodeling']:
                        train_x, train_params, train_y, train_yDot, train_yDDot = trainbatch
                        train_x = train_x.cuda()
                        train_params = train_params.cuda()
                        train_y = train_y.cuda()
                        train_yDot = train_yDot.cuda()
                        train_yDDot = train_yDDot.cuda()
                        
                        with torch.no_grad():
                            train_x = (train_x - self.mean_x) / self.std_x
                            train_params = (train_params - self.mean_params) / self.std_params
                            train_y = (train_y - self.mean_y) / self.std_y
                            train_yDot = (train_yDot) * (self.std_x / self.std_y)
                            train_yDDot = (train_yDDot) * (self.std_x ** 2 / self.std_y)
                        
                        train_x.requires_grad_(True)
                        pred_y = self.forward(torch.cat((train_x, train_params), dim=1))
                        pred_yDot = []
                        pred_yDDot = []
                        for k in range(len(y)):
                            pred_yDot.append(torch.autograd.grad(outputs=pred_y[:, k], inputs=train_x, grad_outputs=torch.ones_like(pred_y[:, k]),
                                                                 create_graph=True, retain_graph=True)[0])
                            pred_yDDot.append(torch.autograd.grad(outputs=pred_yDot[k], inputs=train_x, grad_outputs=torch.ones_like(pred_yDot[k]),
                                                                  create_graph=True, retain_graph=True)[0])
                        pred_yDot = torch.cat(pred_yDot, dim=1)
                        pred_yDDot = torch.cat(pred_yDDot, dim=1)
                        train_x.requires_grad_(False)
                    
                    else:
                        train_x, train_y, train_yDot, train_yDDot = trainbatch
                        train_x = train_x.cuda()
                        train_y = train_y.cuda()
                        train_yDot = train_yDot.cuda()
                        train_yDDot = train_yDDot.cuda()
                        
                        with torch.no_grad():
                            train_x = (train_x - self.mean_x) / self.std_x
                            train_y = (train_y - self.mean_y) / self.std_y
                            train_yDot = (train_yDot) * (self.std_x / self.std_y)
                            train_yDDot = (train_yDDot) * (self.std_x ** 2 / self.std_y)
                        
                        train_x.requires_grad_(True)
                        pred_y = self.forward(train_x)
                        pred_yDot = []
                        pred_yDDot = []
                        for k in range(len(y)):
                            pred_yDot.append(torch.autograd.grad(outputs=pred_y[:, k], inputs=train_x, grad_outputs=torch.ones_like(pred_y[:, k]),
                                                                 create_graph=True, retain_graph=True)[0])
                            pred_yDDot.append(torch.autograd.grad(outputs=pred_yDot[k], inputs=train_x, grad_outputs=torch.ones_like(pred_yDot[k]),
                                                                  create_graph=True, retain_graph=True)[0])
                        pred_yDot = torch.cat(pred_yDot, dim=1)
                        pred_yDDot = torch.cat(pred_yDDot, dim=1)
                        train_x.requires_grad_(False)
                    
                    # Loss
                    loss_y = self.loss_fn(pred_y, train_y)
                    loss_yDot = self.loss_fn(pred_yDot, train_yDot)
                    loss_yDDot = self.loss_fn(pred_yDDot, train_yDDot)
                    
                    # Regularizer
                    regularizer = 0
                    if reg_lambda:
                        for name, param in self.named_parameters():
                            if 'weight' in name:
                                regularizer += torch.sum(torch.square(param))
                        regularizer *= reg_lambda
                    
                    # Backward
                    for param in self.parameters():
                        param.grad = None
                    (loss_y + loss_yDot + loss_yDDot + regularizer).backward()
                    
                    # Save
                    batch_labels = torch.cat((train_y, train_yDot, train_yDDot), dim=1).detach()
                    batch_preds = torch.cat((pred_y, pred_yDot, pred_yDDot), dim=1).detach()
                    tLabel.append(batch_labels.cpu())
                    tPrediction.append(batch_preds.cpu())
                    # Print
                    if (epoch + 1) % print_every == 0 and int(len(self.train_dataloader) / 3) != 0 and (trainbatch_idx + 1) % int(len(self.train_dataloader) / 3) == 0:
                        print(f"Batch {trainbatch_idx + 1}/{len(self.train_dataloader)} loss: {loss_y:.6f}, {loss_yDot:.6f}, {loss_yDDot:.6f}")
                    
                    return (loss_y + loss_yDot + loss_yDDot + regularizer)
                
                if self.model_info['optimizer'] == 'lbfgs':
                    self.optimizer1.step(closure)
                    batch_end = time.time()
                    batch_computation_time = batch_end - batch_start
                    epoch_computation_time += batch_computation_time
                else:
                    if self.model_info['metamodeling']:
                        train_x, train_params, train_y, train_yDot, train_yDDot = trainbatch
                        train_x = train_x.cuda()
                        train_params = train_params.cuda()
                        train_y = train_y.cuda()
                        train_yDot = train_yDot.cuda()
                        train_yDDot = train_yDDot.cuda()
                        
                        with torch.no_grad():
                            train_x = (train_x - self.mean_x) / self.std_x
                            train_params = (train_params - self.mean_params) / self.std_params
                            train_y = (train_y - self.mean_y) / self.std_y
                            train_yDot = (train_yDot) * (self.std_x / self.std_y)
                            train_yDDot = (train_yDDot) * (self.std_x ** 2 / self.std_y)
                        
                        train_x.requires_grad_(True)
                        pred_y = self.forward(torch.cat((train_x, train_params), dim=1))
                        pred_yDot = []
                        pred_yDDot = []
                        for k in range(len(y)):
                            pred_yDot.append(torch.autograd.grad(outputs=pred_y[:, k], inputs=train_x, grad_outputs=torch.ones_like(pred_y[:, k]),
                                                                 create_graph=True, retain_graph=True)[0])
                            pred_yDDot.append(torch.autograd.grad(outputs=pred_yDot[k], inputs=train_x, grad_outputs=torch.ones_like(pred_yDot[k]),
                                                                  create_graph=True, retain_graph=True)[0])
                        pred_yDot = torch.cat(pred_yDot, dim=1)
                        pred_yDDot = torch.cat(pred_yDDot, dim=1)
                        train_x.requires_grad_(False)
                    
                    else:
                        train_x, train_y, train_yDot, train_yDDot = trainbatch
                        train_x = train_x.cuda()
                        train_y = train_y.cuda()
                        train_yDot = train_yDot.cuda()
                        train_yDDot = train_yDDot.cuda()
                        
                        with torch.no_grad():
                            train_x = (train_x - self.mean_x) / self.std_x
                            train_y = (train_y - self.mean_y) / self.std_y
                            train_yDot = (train_yDot) * (self.std_x / self.std_y)
                            train_yDDot = (train_yDDot) * (self.std_x ** 2 / self.std_y)
                        
                        train_x.requires_grad_(True)
                        pred_y = self.forward(train_x)
                        pred_yDot = []
                        pred_yDDot = []
                        for k in range(len(y)):
                            pred_yDot.append(torch.autograd.grad(outputs=pred_y[:, k], inputs=train_x, grad_outputs=torch.ones_like(pred_y[:, k]),
                                                                 create_graph=True, retain_graph=True)[0])
                            pred_yDDot.append(torch.autograd.grad(outputs=pred_yDot[k], inputs=train_x, grad_outputs=torch.ones_like(pred_yDot[k]),
                                                                  create_graph=True, retain_graph=True)[0])
                        pred_yDot = torch.cat(pred_yDot, dim=1)
                        pred_yDDot = torch.cat(pred_yDDot, dim=1)
                        train_x.requires_grad_(False)
                    
                    # Loss
                    loss_y = self.loss_fn(pred_y, train_y)
                    loss_yDot = self.loss_fn(pred_yDot, train_yDot)
                    loss_yDDot = self.loss_fn(pred_yDDot, train_yDDot)
                    
                    # Regularizer
                    regularizer = 0
                    if reg_lambda:
                        for name, param in self.named_parameters():
                            if 'weight' in name:
                                regularizer += torch.square(param)
                        regularizer *= reg_lambda
                    
                    # Backward
                    for param in self.parameters():
                        param.grad = None
                    (loss_y + loss_yDot + loss_yDDot + regularizer).backward()
                    self.optimizer1.step()
                    batch_end = time.time()
                    batch_computation_time = batch_end - batch_start
                    epoch_computation_time += batch_computation_time
                    
                    # Save
                    batch_labels = torch.cat((train_y, train_yDot, train_yDDot), dim=1).detach()
                    batch_preds = torch.cat((pred_y, pred_yDot, pred_yDDot), dim=1).detach()
                    tLabel.append(batch_labels.cpu())
                    tPrediction.append(batch_preds.cpu())
                    # del train_x, train_y, train_yDot, train_yDDot, trainbatch
                    # del pred_y, pred_yDot, pred_yDDot
                    # del batch_preds, batch_labels
                    # torch.cuda.empty_cache()
                    
                    # Print
                    if (epoch + 1) % print_every == 0 and int(len(self.train_dataloader) / 3) != 0 and (trainbatch_idx + 1) % int(len(self.train_dataloader) / 3) == 0:
                        print(f"Batch {trainbatch_idx + 1}/{len(self.train_dataloader)} loss: {loss_y:.6f}, {loss_yDot:.6f}, {loss_yDDot:.6f}")
            
            tLabel = torch.cat(tLabel, dim=0)
            tPrediction = torch.cat(tPrediction, dim=0)
            tLoss_y = self.MSE((tLabel - tPrediction)
                               [:, :len(y)])
            tLoss_yDot = self.MSE((tLabel - tPrediction)
                                  [:, 1 * len(y):2 * len(y)])
            tLoss_yDDot = self.MSE((tLabel - tPrediction)
                                   [:, 2 * len(y):3 * len(y)])
            tLoss_yDDot_ = self.MSE((tLabel - tPrediction)
                                    [:, 2 * len(y):3 * len(y)]
                                    * (self.std_y.cpu() / self.std_x.cpu() ** 2))
            
            # Validation step
            vLabel = []
            vPrediction = []
            self.eval()
            for validbatch in self.valid_dataloader:
                if self.model_info['metamodeling']:
                    valid_x, valid_params, valid_y, valid_yDot, valid_yDDot = validbatch
                    valid_x = valid_x.cuda()
                    valid_params = valid_params.cuda()
                    valid_y = valid_y.cuda()
                    valid_yDot = valid_yDot.cuda()
                    valid_yDDot = valid_yDDot.cuda()
                    
                    with torch.no_grad():
                        valid_x = (valid_x - self.mean_x) / self.std_x
                        valid_params = (valid_params - self.mean_params) / self.std_params
                        valid_y = (valid_y - self.mean_y) / self.std_y
                        valid_yDot = (valid_yDot) * (self.std_x / self.std_y)
                        valid_yDDot = (valid_yDDot) * (self.std_x ** 2 / self.std_y)
                    
                    valid_x.requires_grad_(True)
                    pred_y = self.forward(torch.cat((valid_x, valid_params), dim=1))
                    pred_yDot = []
                    pred_yDDot = []
                    for k in range(len(y)):
                        pred_yDot.append(torch.autograd.grad(outputs=pred_y[:, k], inputs=valid_x, grad_outputs=torch.ones_like(pred_y[:, k]),
                                                             create_graph=True, retain_graph=True)[0])
                        pred_yDDot.append(torch.autograd.grad(outputs=pred_yDot[k], inputs=valid_x, grad_outputs=torch.ones_like(pred_yDot[k]),
                                                              create_graph=True, retain_graph=True)[0])
                    pred_yDot = torch.cat(pred_yDot, dim=1)
                    pred_yDDot = torch.cat(pred_yDDot, dim=1)
                    valid_x.requires_grad_(False)
                
                else:
                    valid_x, valid_y, valid_yDot, valid_yDDot = validbatch
                    valid_x = valid_x.cuda()
                    valid_y = valid_y.cuda()
                    valid_yDot = valid_yDot.cuda()
                    valid_yDDot = valid_yDDot.cuda()
                    
                    with torch.no_grad():
                        valid_x = (valid_x - self.mean_x) / self.std_x
                        valid_y = (valid_y - self.mean_y) / self.std_y
                        valid_yDot = (valid_yDot) * (self.std_x / self.std_y)
                        valid_yDDot = (valid_yDDot) * (self.std_x ** 2 / self.std_y)
                    
                    valid_x.requires_grad_(True)
                    pred_y = self.forward(valid_x)
                    pred_yDot = []
                    pred_yDDot = []
                    for k in range(len(y)):
                        pred_yDot.append(torch.autograd.grad(outputs=pred_y[:, k], inputs=valid_x, grad_outputs=torch.ones_like(pred_y[:, k]),
                                                             create_graph=True, retain_graph=True)[0])
                        pred_yDDot.append(torch.autograd.grad(outputs=pred_yDot[k], inputs=valid_x, grad_outputs=torch.ones_like(pred_yDot[k]),
                                                              create_graph=True, retain_graph=True)[0])
                    pred_yDot = torch.cat(pred_yDot, dim=1)
                    pred_yDDot = torch.cat(pred_yDDot, dim=1)
                    valid_x.requires_grad_(False)
                
                # Save
                batch_labels = torch.cat((valid_y, valid_yDot, valid_yDDot), dim=1).detach()
                batch_preds = torch.cat((pred_y, pred_yDot, pred_yDDot), dim=1).detach()
                vLabel.append(batch_labels.cpu())
                vPrediction.append(batch_preds.cpu())
                # del valid_x, valid_y, valid_yDot, valid_yDDot, validbatch
                # del pred_y, pred_yDot, pred_yDDot
                # del batch_preds, batch_labels
                # torch.cuda.empty_cache()
            self.train()
            
            vLabel = torch.cat(vLabel, dim=0)
            vPrediction = torch.cat(vPrediction, dim=0)
            vLoss_y = self.MSE((vLabel - vPrediction)[:, :len(y)])
            vLoss_yDot = self.MSE((vLabel - vPrediction)[:, len(y):2 * len(y)])
            vLoss_yDDot = self.MSE((vLabel - vPrediction)[:, 2 * len(y):3 * len(y)])
            vLoss_yDDot_ = self.MSE((vLabel - vPrediction)
                                    [:, 2 * len(y):3 * len(y)]
                                    * (self.std_y.cpu() / self.std_x.cpu() ** 2))
            if valid_measure == 'mse':
                vLoss = self.MSE(vLabel - vPrediction)
            elif valid_measure == 'rms':
                vLoss = self.RMS(vLabel, vPrediction)
            try:
                R2value = r2_score(vLabel, vPrediction, multioutput='raw_values')
            except ValueError:
                R2value = torch.zeros(vLabel.shape[1])
                warnings.warn(f"R2 calculation encountered NaN")
            
            tLoss_yDDot_history[epoch] = tLoss_yDDot_.item()
            vLoss_yDDot_history[epoch] = vLoss_yDDot_.item()
            if (epoch + 1) % print_every == 0:
                print(f"Epoch {epoch + 1} (done in {epoch_computation_time:.4f} sec)")
                print(f"Learning rate={self.optimizer1.param_groups[0]['lr']:.3e}")
                print(f"Normalized loss1 = Train:{tLoss_y.item():.6f}, Valid:{vLoss_y.item():.6f}")
                print(f"Normalized loss2 = Train:{tLoss_yDot.item():.6f}, Valid:{vLoss_yDot.item():.6f}")
                print(f"Normalized loss3 = Train:{tLoss_yDDot.item():.6f}, Valid:{vLoss_yDDot.item():.6f}")
                print(f"Validation measure({valid_measure.upper()}) = {vLoss.item():.6f}")
                for outidx, r2value in enumerate(R2value):
                    print(f"R2({(y + yDot + yDDot)[outidx]})={r2value:.6f}")
                print()
            
            # Save state dict
            if (epoch + 1) % save_every == 0:
                model_parameter = self.state_dict()
                for name, param in model_parameter.items():
                    model_parameter[name] = param.cpu()
                model_validation_loss[epoch] = vLoss.item()
                model_parameter_group[epoch] = model_parameter
            
            # LR decay
            if (epoch + 1) % lr_halflife == 0:
                old_LR = self.optimizer1.param_groups[0]['lr']
                self.optimizer1.param_groups[0]['lr'] /= 2
                new_LR = self.optimizer1.param_groups[0]['lr']
                print(f"LR decayed, {old_LR:.3e} -> {new_LR:.3e}")
                print()
        
        # End of training epoch
        TrainingEndTime = time.time()
        Hr, Min, Sec = DP.Sec2Time(TrainingEndTime - TrainingStartTime)
        print(f"Training finished in {Hr}hr {Min}min {Sec}sec.")
        self.count_params()
        if save:
            idx_argmin = np.argmin(model_validation_loss)
            print(f"Saving the best model:")
            print(f"Validation loss({valid_measure.upper()}) {model_validation_loss[idx_argmin]:.6f} at Epoch {idx_argmin + 1} ")
            
            self.model_info['training_time'] = (TrainingEndTime - TrainingStartTime)
            self.model_info['yddot_training_loss_history'] = tLoss_yDDot_history
            self.model_info['yddot_validation_loss_history'] = vLoss_yDDot_history
            self.model_info['model_state_dict'] = model_parameter_group[idx_argmin]
            self.model_info = {key: value for key, value in sorted(self.model_info.items())}  # arrange
            
            if not os.path.exists('Models'):
                os.mkdir("Models")
            torch.save(self.model_info, f'Models/{filename}')
            # Dictionary txt
            f = open(f"Models/{filename.split('.')[0]}.txt", "w")
            lines = []
            exceptions = ["model_state_dict", ]
            for k in self.model_info.keys():
                if k in exceptions:
                    continue
                else:
                    lines.append(f"{k} : {self.model_info[k]}\n")
            f.writelines(lines)
            f.close()
            
            if model_parameter_group[idx_argmin] == None:
                print(f"WARNING: Saved model parameter is None")
    
    def load_model_info(self, model_info: dict):
        self.model_info = model_info
        self.__dict__.update(self.model_info)
        
        self.mean_x = model_info['mean_x']
        self.mean_y = model_info['mean_y']
        self.mean_yDDot = model_info['mean_yDDot']
        self.mean_yDot = model_info['mean_yDot']
        self.std_x = model_info['std_x']
        self.std_y = model_info['std_y']
        self.std_yDDot = model_info['std_yDDot']
        self.std_yDot = model_info['std_yDot']
        
        if self.metamodeling:
            self.mean_params = model_info['mean_params']
            self.std_params = model_info['std_params']
        
        self.__init__(self.in_N, self.depth, self.width, self.out_N, self.activation,
                      self.param_init, self.batchnorm, self.dropout)
        self.load_state_dict(self.model_state_dict)
        self.cuda()
    
    def forward_with_normalization(self, x):
        self.eval()
        x = x.cuda()
        if self.metamodeling:
            x = (x - torch.cat((self.mean_x, self.mean_params))) / torch.cat((self.std_x, self.std_params))
            t = x[:, 0].view(-1, 1)
            t.requires_grad_(True)
            pred_y = self.forward(torch.cat((t, x[:, 1:]), dim=1))
            pred_yDot = []
            pred_yDDot = []
            for k in range(len(y)):
                pred_yDot.append(torch.autograd.grad(outputs=pred_y[:, k], inputs=t, grad_outputs=torch.ones_like(pred_y[:, k]),
                                                     create_graph=True, retain_graph=True)[0])
                pred_yDDot.append(torch.autograd.grad(outputs=pred_yDot[k], inputs=t, grad_outputs=torch.ones_like(pred_yDot[k]),
                                                      create_graph=True, retain_graph=True)[0])
            pred_yDot = torch.cat(pred_yDot, dim=1)
            pred_yDDot = torch.cat(pred_yDDot, dim=1)
            t.requires_grad_(False)
        else:
            x = (x - self.mean_x) / self.std_x
            t = x
            t.requires_grad_(True)
            pred_y = self.forward(t)
            pred_yDot = []
            pred_yDDot = []
            for k in range(len(y)):
                pred_yDot.append(torch.autograd.grad(outputs=pred_y[:, k], inputs=t, grad_outputs=torch.ones_like(pred_y[:, k]),
                                                     create_graph=True, retain_graph=True)[0])
                pred_yDDot.append(torch.autograd.grad(outputs=pred_yDot[k], inputs=t, grad_outputs=torch.ones_like(pred_yDot[k]),
                                                      create_graph=True, retain_graph=True)[0])
            pred_yDot = torch.cat(pred_yDot, dim=1)
            pred_yDDot = torch.cat(pred_yDDot, dim=1)
            t.requires_grad_(False)
        
        pred_y = (pred_y * self.std_y + self.mean_y).detach().cpu()
        pred_yDot = (pred_yDot * (self.std_y / self.std_x)).detach().cpu()
        pred_yDDot = (pred_yDDot * (self.std_y / self.std_x ** 2)).detach().cpu()
        
        return pred_y, pred_yDot, pred_yDDot
    
    
    def forward_without_normalization(self, x):
        self.eval()
        x = x.cuda()
        if self.metamodeling:
            x = (x - torch.cat((self.mean_x, self.mean_params))) / torch.cat((self.std_x, self.std_params))
            t = x[:, 0].view(-1, 1)
            t.requires_grad_(True)
            pred_y = self.forward(torch.cat((t, x[:, 1:]), dim=1))
            pred_yDot = []
            pred_yDDot = []
            for k in range(len(y)):
                pred_yDot.append(torch.autograd.grad(outputs=pred_y[:, k], inputs=t, grad_outputs=torch.ones_like(pred_y[:, k]),
                                                     create_graph=True, retain_graph=True)[0])
                pred_yDDot.append(torch.autograd.grad(outputs=pred_yDot[k], inputs=t, grad_outputs=torch.ones_like(pred_yDot[k]),
                                                      create_graph=True, retain_graph=True)[0])
            pred_yDot = torch.cat(pred_yDot, dim=1)
            pred_yDDot = torch.cat(pred_yDDot, dim=1)
            t.requires_grad_(False)
        else:
            x = (x - self.mean_x) / self.std_x
            t = x
            t.requires_grad_(True)
            pred_y = self.forward(t)
            pred_yDot = []
            pred_yDDot = []
            for k in range(len(y)):
                pred_yDot.append(torch.autograd.grad(outputs=pred_y[:, k], inputs=t, grad_outputs=torch.ones_like(pred_y[:, k]),
                                                     create_graph=True, retain_graph=True)[0])
                pred_yDDot.append(torch.autograd.grad(outputs=pred_yDot[k], inputs=t, grad_outputs=torch.ones_like(pred_yDot[k]),
                                                      create_graph=True, retain_graph=True)[0])
            pred_yDot = torch.cat(pred_yDot, dim=1)
            pred_yDDot = torch.cat(pred_yDDot, dim=1)
            t.requires_grad_(False)
        
        pred_y = (pred_y).detach().cpu()
        pred_yDot = ((pred_yDot * (self.std_y / self.std_x) - self.mean_yDot) / self.std_yDot).detach().cpu()
        pred_yDDot = ((pred_yDDot * (self.std_y / self.std_x ** 2) - self.mean_yDDot) / self.std_yDDot).detach().cpu()
        
        return pred_y, pred_yDot, pred_yDDot

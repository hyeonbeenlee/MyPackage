import pandas as pd
import torch
from torch.utils.data import TensorDataset, DataLoader
from MyPackage.utils import *

class NN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model_info = dict()
        self.save_model_kwargs(locals())
    
    def forward(self, x: torch.Tensor):
        pass
    
    def __initializer(self, layer):
        pass
    
    def count_params(self):
        num_params = 0
        for param in self.parameters():
            if len(param.data.shape) == 1:
                num_params += param.data.shape[0]
            elif len(param.data.shape) == 2:
                num_params += param.data.shape[0] * param.data.shape[1]
            elif len(param.data.shape) == 3:
                num_params += param.data.shape[0] * param.data.shape[1] * param.data.shape[2]
        print(f"Number of trainable parameters: {num_params}")
        return num_params
    
    def load_csv(self):
        pass
    
    def setup_dataloader(self,
            input_traindata: pd.DataFrame,
            output_traindata: pd.DataFrame,
            input_validdata: pd.DataFrame,
            output_validdata: pd.DataFrame, batch_size:int=128):
        self.save_model_kwargs(locals())
        input_traindata=df2tensor(input_traindata)
        output_traindata=df2tensor(output_traindata)
        input_validdata = df2tensor(input_validdata)
        output_validdata = df2tensor(output_validdata)
        self.train_dataloader = DataLoader(TensorDataset(input_traindata, output_traindata),
                                           batch_size,
                                           shuffle=True,
                                           num_workers=4,
                                           pin_memory=True,
                                           persistent_workers=True)
        self.valid_dataloader = DataLoader(TensorDataset(input_validdata, output_validdata),
                                           batch_size,
                                           shuffle=True,
                                           num_workers=4,
                                           pin_memory=True,
                                           persistent_workers=True)
        self.model_info['mean_I'] = input_traindata.mean(dim=0).cuda()
        self.model_info['mean_O'] = output_traindata.mean(dim=0).cuda()
        self.model_info['std_I'] = input_traindata.std(dim=0).cuda()
        self.model_info['std_O'] = output_traindata.std(dim=0).cuda()
        self.model_info['min_I'] = input_traindata.min(dim=0).values.cuda()
        self.model_info['min_O'] = output_traindata.min(dim=0).values.cuda()
        self.model_info['max_I'] = input_traindata.max(dim=0).values.cuda()
        self.model_info['max_O'] = output_traindata.max(dim=0).values.cuda()
    
    def setup_optimizer(self):
        self.save_model_kwargs(locals())
    
    def setup_loss_fn(self):
        self.save_model_kwargs(locals())
    
    def fit(self):
        self.save_model_kwargs(locals())
        
        # psuedocodes
        # psuedocodes
        # psuedocodes
        # psuedocodes
        # psuedocodes
        # psuedocodes
        # psuedocodes
        # psuedocodes
        self.setup_dataloader()
        self.setup_optimizer()
        self.setup_loss_fn()
        
        for key, value in self.model_info.items():
            print(f"{key.upper()}: {value}")
        print()
        
        for module in self.children():
            print(module)
        print()
        
        tLoss_O_history = np.empty(epochs)
        vLoss_O_history = np.empty(epochs)
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
                train_I, train_O = trainbatch
                train_I = train_I.cuda()
                train_O = train_O.reshape(-1, self.n_outputsteps * len(O)).cuda()
                
                with torch.no_grad():
                    if self.datascaler == 'z':
                        train_I = (train_I - self.mean_I) / self.std_I
                        train_O = (train_O - self.mean_O.reshape(-1, self.n_outputsteps * len(O))) / self.std_O.reshape(-1,
                                                                                                                        self.n_outputsteps * len(O))
                    elif self.datascaler == 'pm1' or 'datascaler' not in self.__dict__.keys():
                        train_I = 2 * (train_I - self.min_I) / (self.max_I - self.min_I) - 1
                        train_O = 2 * (train_O - self.min_O.reshape(-1, self.n_outputsteps * len(O))) / (
                                self.max_O.reshape(-1, self.n_outputsteps * len(O)) - self.min_O.reshape(-1, self.n_outputsteps * len(O))) - 1
                
                pred_O = self.forward(train_I)
                
                # Loss
                loss_O = self.loss_fn(pred_O, train_O.reshape_as(pred_O))
                
                batch_labels = train_O
                batch_preds = pred_O
                
                tLabel.append(batch_labels.detach().cpu().flatten())
                tPrediction.append(batch_preds.detach().cpu().flatten())
                
                # Regularizer
                regularizer = 0
                if reg_lambda:
                    # for name, param in self.named_parameters():
                    # if 'weight' in name:
                    #     regularizer += torch.sum(torch.square(param))
                    # for name, param in self.ConvBlocks.named_parameters():
                    #     if 'weight' in name:
                    #         regularizer += torch.sum(torch.square(param))
                    for name, param in self.Linear_I.named_parameters():
                        if 'weight' in name:
                            regularizer += torch.sum(torch.square(param))
                    if hasattr(self, 'Linear_H'):
                        for name, param in self.Linear_H.named_parameters():
                            if 'weight' in name:
                                regularizer += torch.sum(torch.square(param))
                    if hasattr(self, 'Linear_O'):
                        for name, param in self.Linear_O.named_parameters():
                            if 'weight' in name:
                                regularizer += torch.sum(torch.square(param))
                    regularizer *= reg_lambda
                
                # Backward
                for param in self.parameters():
                    param.grad = None
                (loss_O + regularizer).backward()
                self.optimizer1.step()
                if multioptim:
                    self.optimizer2.step()
                    self.optimizer3.step()
                batch_end = time.time()
                batch_computation_time = batch_end - batch_start
                epoch_computation_time += batch_computation_time
                # Print
                if (epoch + 1) % print_every == 0 and int(len(self.train_dataloader) / 3) != 0 and (
                        trainbatch_idx + 1) % int(len(self.train_dataloader) / 3) == 0:
                    print(f"Batch {trainbatch_idx + 1}/{len(self.train_dataloader)} Value MSE: {loss_O:.6f}")
            
            tLabel = torch.cat(tLabel, dim=0)
            tPrediction = torch.cat(tPrediction, dim=0)
            
            tLoss_O = MSE(tLabel - tPrediction)
            if self.datascaler == 'z':
                tLoss_O_ = MSE((tLabel - tPrediction) * self.std_O.cpu() + self.mean_O.cpu())
            elif self.datascaler == 'pm1' or 'datascaler' not in self.__dict__.keys():
                tLoss_O_ = MSE((tLabel - tPrediction + 1) * (self.max_O.cpu() - self.min_O.cpu()) + self.min_O.cpu())
            
            # Validation step
            vLabel = []
            vPrediction = []
            self.eval()
            with torch.no_grad():
                for idx_v, validbatch in enumerate(self.valid_dataloader):
                    valid_I, valid_O = validbatch
                    valid_I = valid_I.cuda()
                    valid_O = valid_O.reshape(-1, self.n_outputsteps * len(O)).cuda()
                    
                    with torch.no_grad():
                        if self.datascaler == 'z':
                            valid_I = (valid_I - self.mean_I) / self.std_I
                            valid_O = (valid_O - self.mean_O.reshape(-1, self.n_outputsteps * len(O))) / self.std_O.reshape(-1,
                                                                                                                            self.n_outputsteps * len(O))
                        elif self.datascaler == 'pm1' or 'datascaler' not in self.__dict__.keys():
                            valid_I = 2 * (valid_I - self.min_I) / (self.max_I - self.min_I) - 1
                            valid_O = 2 * (valid_O - self.min_O.reshape(-1, self.n_outputsteps * len(O))) / (
                                    self.max_O.reshape(-1, self.n_outputsteps * len(O)) - self.min_O.reshape(-1, self.n_outputsteps * len(O))) - 1
                    
                    pred_O = self.forward(valid_I)
                    
                    # if idx_v == 0:
                    #     init_input = valid_I
                    #     pred_O = self.forward(init_input)  # batch, seq 10
                    # elif idx_v == 1:
                    #     label_seq = torch.cat([init_input, valid_O], dim=1)
                    #     autoregressive_seq = torch.cat([init_input, pred_O],
                    #                                    dim=1)  # batch, seq 10+3
                    #     pred_O = self.forward(autoregressive_seq)
                    # else:
                    #     label_seq = torch.cat([label_seq, valid_O[:, -1].reshape(1, -1)], dim=1)
                    #     autoregressive_seq = torch.cat([autoregressive_seq,
                    #                                     pred_O[:, -1].reshape(1, -1)], dim=1)
                    
                    batch_labels = valid_O
                    batch_preds = pred_O
                    vLabel.append(batch_labels.detach().cpu())
                    vPrediction.append(batch_preds.detach().cpu())
                self.train()
                
                vLabel = torch.cat(vLabel, dim=0)
                vPrediction = torch.cat(vPrediction, dim=0)
                vLoss_O = MSE(vLabel - vPrediction)
                if self.datascaler == 'z':
                    vLoss_O_ = MSE((vLabel - vPrediction) * self.std_O.cpu().reshape(-1, self.n_outputsteps * len(O)) + self.mean_O.cpu().reshape(-1,
                                                                                                                                                  self.n_outputsteps * len(
                                                                                                                                                          O)))
                elif self.datascaler == 'pm1' or 'datascaler' not in self.__dict__.keys():
                    vLoss_O_ = MSE((vLabel - vPrediction + 1) * (self.max_O.cpu().reshape(-1, self.n_outputsteps * len(O)) - self.min_O.cpu().reshape(
                            -1,
                            self.n_outputsteps * len(O))) + self.min_O.cpu().reshape(-1, self.n_outputsteps * len(O)))
                R2value = r2_score(vLabel, vPrediction, multioutput='raw_values')
                CrossCorr = []
                RRMSE = []
                PeakToPeak = []
                for j in range(vLabel.shape[1]):
                    CrossCorr.append(CrossCorrelate(vLabel[:, j], vPrediction[:, j]))
                    RRMSE.append(RelativeRMSErr(vLabel[:, j], vPrediction[:, j]))
                    PeakToPeak.append(PeakToPeakErr(vLabel[:, j], vPrediction[:, j]))
            
            if valid_measure == 'mse':
                vLoss = MSE(vLabel - vPrediction)
            elif valid_measure == 'rrmse':
                vLoss = RRMSE
            elif valid_measure == 'crosscorr':
                vLoss = 1 - np.mean(CrossCorr)
            elif valid_measure == 'peaktopeak':
                vLoss = PeakToPeak
            
            tLoss_O_history[epoch] = tLoss_O_.item()
            vLoss_O_history[epoch] = vLoss_O_.item()
            if (epoch + 1) % print_every == 0:
                print(f"Epoch {epoch + 1} (done in {epoch_computation_time:.4f} sec")
                print(f"Learning rate={self.optimizer1.param_groups[0]['lr']:.3e}")
                print(f"Regularizer = {regularizer:.6f}")
                print(f"Normalized loss = Train: {tLoss_O.item():.6f}, Valid: {vLoss_O.item():.6f}")
                print(f"Validation measure({valid_measure.upper()} Loss) = {vLoss.item():.6f}")
                if valid_measure == 'crosscorr':
                    print(f"Mean CrossCorr: {np.mean(CrossCorr):.6f}")
                for outidx, r2value in enumerate(R2value):
                    print_text = f"{O[outidx]}: R2={r2value:.4f}, "
                    print_text += f"CrossCorr={CrossCorr[outidx]:.4f}, "
                    print_text += f"RRMSE={RRMSE[outidx]:.4f}%, "
                    print_text += f"PeakToPeak={PeakToPeak[outidx]:.4f}%"
                    print(print_text)
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
                if multioptim:
                    self.optimizer2.param_groups[0]['lr'] /= 2
                    self.optimizer3.param_groups[0]['lr'] /= 2
                new_LR = self.optimizer1.param_groups[0]['lr']
                print(f"LR decayed, {old_LR:.3e} -> {new_LR:.3e}")
                print()
        # End of training epoch
        TrainingEndTime = time.time()
        Hr, Min, Sec = MyPackage.utils.Sec2Time(TrainingEndTime - TrainingStartTime)
        print(f"Training finished in {Hr}hr {Min}min {Sec}sec.")
        self.count_params()
        if save:
            idx_argmin = np.argmin(model_validation_loss)
            print(f"Saving the best model:")
            print(f"Validation loss({valid_measure.upper()}) {model_validation_loss[idx_argmin]:.6f} at Epoch {idx_argmin + 1} ")
            
            self.model_info['training_time'] = (TrainingEndTime - TrainingStartTime)
            self.model_info['O_training_loss_history'] = tLoss_O_history
            self.model_info['O_validation_loss_history'] = vLoss_O_history
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
            
            if model_parameter_group[idx_argmin] is None:
                print(f"WARNING: Saved model parameter is None")
    
    def load_model_info(self, model_info: dict):
        self.model_info = model_info
        import inspect
        list_kwargs = inspect.signature(self.__init__)
        valid_keywords_init = []
        for k, v in list_kwargs.parameters.items():
            valid_keywords_init.append(k)
        init_kwargs = {k: self.model_info[k] for k in valid_keywords_init}
        self.__init__(**init_kwargs)
        self.__dict__.update(model_info)
        
        self.min_I = model_info['min_I']
        self.min_O = model_info['min_O']
        self.max_I = model_info['max_I']
        self.max_O = model_info['max_O']
        self.mean_I = model_info['mean_I']
        self.mean_O = model_info['mean_O']
        self.std_I = model_info['std_I']
        self.std_O = model_info['std_O']
        
        self.load_state_dict(self.model_state_dict)
        self.cuda()
    
    def save_model_kwargs(self, locals):
        self.__dict__.update(locals)
        vars = dict(locals.items())
        if 'self' in vars.keys(): del vars['self']
        if '__class__' in vars.keys(): del vars['__class__']
        self.model_info.update(vars)
        self.model_info = {key: value for key, value in sorted(self.model_info.items())}
    
    def forward_with_normalization(self, x):
        self.eval()
    
    
    def forward_without_normalization(self, x):
        self.eval()

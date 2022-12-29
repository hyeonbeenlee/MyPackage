import torch

def MSE(self, x):
    return torch.mean(torch.square(x))

def MAE(self, x):
    return torch.mean(torch.abs(x))
import torch

def MSE(x):
    return torch.mean(torch.square(x))

def MAE(x):
    return torch.mean(torch.abs(x))
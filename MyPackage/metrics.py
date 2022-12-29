import numpy as np
from scipy.interpolate import Akima1DInterpolator as akispl

def RelativeMSE(exptime, exp, simtime, sim):
    exp_interp = akispl(exptime, exp)
    sim_interp = akispl(simtime, sim)
    epsilon = 1e-10
    new_time = np.arange(simtime[0], simtime[-1] + epsilon, exptime[1] - exptime[0])
    exp = exp_interp(new_time)
    sim = sim_interp(new_time)
    err = np.mean(np.square(exp - sim)) / np.sum(np.square(sim))
    return err

def RelativeRMSErr(self, label, prediction):
    label = label.numpy()
    prediction = prediction.numpy()
    label = label - label.mean()
    prediction = prediction - prediction.mean()
    rms_label = np.sqrt(np.mean(np.square(label)))
    rms_prediction = np.sqrt(np.mean(np.square(prediction)))
    rms_error = np.abs(rms_label - rms_prediction) / rms_label * 100
    return rms_error

def MeanValueErr(exp, sim):
    mean_exp = exp.mean()
    mean_sim = sim.mean()
    err = np.abs(np.abs(mean_exp - mean_sim) / mean_exp) * 100
    return err

def CrossCorrelate(self, label, pred, labeltime=None, predtime=None):
    label = label.numpy()
    pred = pred.numpy()
    
    if label.shape[0]!=pred.shape[1]:
        label_interp = akispl(labeltime, label)
        pred_interp = akispl(predtime, pred)
        epsilon = 1e-10
        new_time = np.arange(predtime[0], predtime[-1] - epsilon, labeltime[1] - labeltime[0])
        label = label_interp(new_time)
        pred = pred_interp(new_time)
    coeff = np.corrcoef(label, pred)
    return np.abs(coeff[0, 1])  # exp and sim

def PeakToPeakErr(self, label, pred):
    label = label.numpy()
    pred = pred.numpy()
    label_err = np.max(label) - np.min(label)
    pred_err = np.max(pred) - np.min(pred)
    err = np.abs(label_err - pred_err) / label_err * 100
    return err


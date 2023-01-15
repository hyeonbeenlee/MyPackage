import numpy as np
import pandas as pd
from scipy.signal import butter, sosfiltfilt

def FiltButterworth(data_1darray, cutoff, timestep, order, mode: str = 'low'):
    f_sampling = 1 / timestep
    nyq = f_sampling * 0.5
    normal_cutoff = cutoff / nyq
    sos = butter(order, normal_cutoff, btype=mode, analog=False, output='sos')
    data_1darray = sosfiltfilt(sos, data_1darray)
    return data_1darray

def FFT(data_1D, timestep):
    n_samples = len(data_1D)
    Freq = np.fft.rfftfreq(n=n_samples, d=timestep)
    Amp = np.abs(np.fft.rfft(data_1D, n=n_samples, norm='forward')) * 2  # Drop imaginary and double amplitude
    return Freq, Amp

def PCA(data_2D: np.array, n_components: int):
    if n_components > data_2D.shape[1]:
        raise UserWarning(f"n_components cannot exceed the number of features ({data_2D.shape[1]}) of data array.")
    # Standardize
    mean = data_2D.mean(axis=0)
    std = data_2D.std(axis=0)
    data = (data_2D - mean) / std
    u, s, vh = np.linalg.svd(data)
    s = np.diag(s)
    u_t = u[:, :n_components]
    s_t = s[:n_components, :n_components]
    vh_t = vh[:n_components, :]
    reduced_data = u_t @ s_t
    restored_data = (u_t @ s_t @ vh_t) * std + mean
    singular_values=s.diagonal()
    return reduced_data, restored_data, singular_values

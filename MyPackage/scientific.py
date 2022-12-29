import numpy as np
from scipy.signal import butter,sosfiltfilt

def FiltButterworth(data_1darray, cutoff, timestep, order, mode: str = 'low'):
    f_sampling = 1 / timestep
    nyq = f_sampling * 0.5
    normal_cutoff = cutoff / nyq
    sos = butter(order, normal_cutoff, btype=mode, analog=False, output='sos')
    data_1darray = sosfiltfilt(sos, data_1darray)
    return data_1darray

def FFT(data_1Darray, timestep):
    n_samples = len(data_1Darray)
    Freq = np.fft.rfftfreq(n=n_samples, d=timestep)
    Amp = np.abs(np.fft.rfft(data_1Darray, n=n_samples, norm='forward')) * 2  # Drop imaginary and double amplitude
    return Freq, Amp
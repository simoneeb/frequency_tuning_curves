
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.ndimage import gaussian_filter
import seaborn as sns
import pickle
from tqdm import tqdm

import matplotlib.colors as colors
import matplotlib.cm as cmx

from numpy.fft import fft, rfft
from scipy.signal import spectrogram,find_peaks
from sklearn.linear_model import LinearRegression


def power_spectrum(signal,time, show = True, nb_pad =0, reg = True):
        
    T = time[-1]
    N = len(time)
    dt = time[1]-time[0]
    
    time = np.arange(0,T+nb_pad*T,dt)
    signal = np.concatenate((signal,np.zeros(nb_pad*N)))

    # if signal.std() > 0 :
    #     signal = (signal - np.mean(signal))/signal.std()
        
    if reg is True:

        ind = [0,55]
        timesh = time[ind[0]:ind[1]]
        timsht = timesh.reshape(-1,1)
        sigsh = signal[ind[0]:ind[1]]
        model = LinearRegression()
        model.fit(timsht, sigsh)
        a = model.coef_[0]
        b = model.intercept_

        regline = a*timesh +b

        signal[ind[0]:ind[1]] = signal[ind[0]:ind[1]]- regline    

  

    xf = fft(signal)                                            # Compute Fourier transform of x
    Sxx = 2 * dt ** 2 / (nb_pad*T + T) * (xf * xf.conj())       # Compute spectrum
    Sxx = Sxx[:int(len(signal) / 2)]                            # Ignore negative frequencies

    df = 1 / (nb_pad*T + T)              # Determine frequency resolution
    fNQ = 1 / dt / 2                     # Determine Nyquist frequency
    faxis = np.arange(0,fNQ,df)          # Construct frequency axis
    faxis = np.arange(0,(N*nb_pad+N)/2)*df               # Construct frequency axis
    
    if Sxx.std() > 0:
        Sxx =(Sxx -Sxx.mean())/Sxx.std()

    # if show is True:
    #     plt.plot(faxis,Sxx.real, label = 'power spectrum')
    #     for i in fs:
    #         plt.axvline(i)
        
    #     plt.xlabel('Frequency [Hz]')              # Label the axes
    #     plt.ylabel('Power [$\mu V^2$/Hz]')
    #     plt.legend()
    if reg is False:
        return faxis, Sxx
    if reg is True:
        return faxis, Sxx, regline, signal





# load data

exp_name = 'MR-0609'
fpdata = f'/user/sebert/home/Documents/Experiments/Spatiotemporal_tuning_curves/Results/{exp_name}/data.pkl'

with open(fpdata, "rb") as handle:   #Pickling
    data = pickle.load(handle)

keys = list(data.keys())[3:]
#keys = keys[:4]

fpdataheat = f'/user/sebert/home/Documents/Experiments/Spatiotemporal_tuning_curves/Results/{exp_name}/dataframe.csv'
frequency_df = pd.read_csv(fpdataheat)


fxs = [1.,2.,4.,6.,10.]
fts = [2.,4.,5.,6.,8.]
contrasts = ['cm10', 'cm05', 'cm02']
grating_types = ['flickering', 'moving']
i = 'nd4'
nb_stimuli = 25


T = 2.0
dt = 0.02
time = np.arange(0,T,dt)
N = len(time)

#keys = ['Unit_0114']

for key in tqdm(keys):

    for grating_type in grating_types:

        for cm in contrasts:

            grat = f'{grating_type}_{cm}'

            spectrums = []

            resps = []
            resps_full = []
            data[key][i][grat]['power_spectrum_old']=[]
            data[key][i][grat]['power_spectrum']=[]
            data[key][i][grat]['F_close_amp']=[]
            data[key][i][grat]['F_close_f']=[]
            data[key][i][grat]['F_close_idx']=[]
            data[key][i][grat]['F_max_amp']=[]
            data[key][i][grat]['F_max_f']=[]
            data[key][i][grat]['F_max_idx']=[]
            data[key][i][grat]['fpeaks']=[]

            
            signals = []

            for nbs in range(nb_stimuli):

                #nbs = 2
                y = int(np.floor(nbs/5))
                x = int(nbs%5)
                ft = fts[x]

                signal = data[key][i][grat]['counts_sorted_stim_alinged'][nbs]
                mean_stim = np.mean(signal)
                #signal  = signal - mean_stim
                
                
                # look for regress line 
                
                # remove regress line from signal 
                
            
                faxis,Sxx = power_spectrum(signal,time,show = False, nb_pad = 10, reg = False)
                faxis_new,Sxx_new,regline,newsig = power_spectrum(signal,time,show = False, nb_pad = 10)
                spectrums.append(Sxx.real)
                resps.append(signal)


                #signals.append(signal)
                
                # detect highest peaks bigger than before 
                end = int(len(Sxx.real))
        
                # detect all peaks 

                # if x == 0:
                #start = 0
                peaks_idxs,peak_heights = find_peaks(Sxx_new.real, height =-0.1)
                #peaks_idxs,peak_heights = find_peaks(Sxx_new.real[:end], height = 0)
                peaks = faxis[peaks_idxs]
                # else : 
                #     start = int(np.nonzero(faxis > fts[x-1])[0][0])
                #     faxis = faxis[start:end]
                #     peaks_idxs,peak_heights = find_peaks(Sxx_new.real[start:end], height = 0)
                #     peaks = faxis[peaks_idxs]




                #get peak closest to stimulus frequency
                difference_array = np.absolute(np.abs(peaks-ft))
                # find the index of minimum element from the array

                try:
                    if difference_array.min() <= 1 :
                        idx = difference_array.argmin()
                    else:
                        idx = np.nan
                except:
                    idx = np.nan

                try:
                    amp_idx = peaks_idxs[idx]
                    amp_ft = peak_heights['peak_heights'][idx]
                    f_ft = faxis[amp_idx]

                except: 
                    amp_idx = np.nan
                    amp_ft = np.nan
                    f_ft = np.nan


                # get biggest peak
                try:
                    idx_max = peaks_idxs[np.argmax(peak_heights['peak_heights'])]
                    f_ft_max = faxis[idx_max]
                    amp_ft_max = np.max(peak_heights['peak_heights'])
                except :
                    idx_max = np.nan
                    f_ft_max = np.nan
                    amp_ft_max = np.nan


                data[key][i][grat]['power_spectrum_old'].append([faxis, Sxx])
                data[key][i][grat]['power_spectrum'].append([faxis_new, Sxx_new])
                data[key][i][grat]['F_close_amp'].append(amp_ft)
                data[key][i][grat]['F_close_f'].append(f_ft)
                data[key][i][grat]['F_close_idx'].append(amp_idx )
                data[key][i][grat]['F_max_amp'].append(amp_ft_max)
                data[key][i][grat]['F_max_f'].append(f_ft_max)
                data[key][i][grat]['F_max_idx'].append(idx_max )
                data[key][i][grat]['fpeaks'].append(peaks)

                # add data to dataframe

                frequency_df.loc[(frequency_df['key'] == key) & (frequency_df['stim_id'] == nbs), 'F1'] = amp_ft
                frequency_df.loc[(frequency_df['key'] == key )& (frequency_df['stim_id'] == nbs), 'zF1']= amp_ft



x = 0
# save new data
with open(fpdata, "wb") as handle:   #Pickling
    pickle.dump(data,handle, protocol=4)


frequency_df.to_csv(fpdataheat)



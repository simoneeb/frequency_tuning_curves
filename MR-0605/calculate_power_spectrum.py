
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
from scipy.signal import spectrogram


def power_spectrum(signal,time, show = True):
    


    T = time[-1]
    N = len(time)
    dt = time[1]-time[0]
    signal = (signal - signal.mean())/signal.std()
    xf = fft(signal)               # Compute Fourier transform of x
    Sxx = 2 * dt ** 2 / T * (xf * xf.conj())       # Compute spectrum
    Sxx = Sxx[:int(len(signal) / 2)]               # Ignore negative frequencies

    df = 1 / T                     # Determine frequency resolution
    fNQ = 1 / dt / 2                      # Determine Nyquist frequency
    faxis = np.arange(0,fNQ,df)           # Construct frequency axis


    Sxx = (Sxx - Sxx.mean())/Sxx.std()
    # if show is True:
    #     plt.plot(faxis,Sxx.real, label = 'power spectrum')
    #     for i in fs:
    #         plt.axvline(i)
        
    #     plt.xlabel('Frequency [Hz]')              # Label the axes
    #     plt.ylabel('Power [$\mu V^2$/Hz]')
        # plt.legend()

    return faxis, Sxx




# load data
fpdata = '/user/sebert/home/Documents/Experiments/Spatiotemporal_tuning_curves/MR-0605/results/data.pkl'
with open(fpdata, "rb") as handle:   #Pickling
    data = pickle.load(handle)


frequency_df = pd.DataFrame(columns = ['fx', 'ft', 'mean_stim','F1' 'zF1', 'key', 'grating','intensity'])


key = 'temp_3'
nb_stimuli = 25
i = 'nd3'
grat = 'flickering'



fxs = [1.,2.,4.,6.,10.]
fts = [2.,4.,5.,6.,8.]

T = 2.0
dt = 0.02
nb_pad = 10

time = np.arange(0,T,dt)
time_pad = np.arange(0,T+nb_pad*T,dt)
N = len(time)



spectrums = []
spectrums_pad = []

resps = []
resps_pad = []

for nbs in range(nb_stimuli):

    row ={}

    y = int(np.floor(nbs/5))
    row['fx'] = float(fxs[y])
    
    x = int(nbs%5)
    row['ft'] = float(fts[x])
    row['v'] = float(fts[x]/fxs[y])

    row['intensity'] = i
    row['grating'] = grat
    row['key'] = key


    signal = data[key][i][grat]['counts_sorted_stim_alinged'][nbs][:-1]
    signal_pad = np.concatenate((signal,np.zeros(nb_pad*N)))

    mean_stim = np.mean(signal)
    faxis,Sxx = power_spectrum(signal,time,show = False)
    faxis_pad,Sxx_pad = power_spectrum(signal_pad,time_pad,show = False)
    spectrums.append(Sxx.real)
    spectrums_pad.append(Sxx_pad.real)
    resps.append(signal)
    resps_pad.append(signal_pad)



    amp_ft = Sxx.real[np.round(faxis) == fts[x]][0]
    mean = np.mean(Sxx.real)
    std = np.std(Sxx.real)
    zamp_ft = (amp_ft - mean)/std


    row['F1'] = amp_ft
    row['zF1'] = zamp_ft
    row['mean_stim'] = mean_stim



    frequency_df = frequency_df.append(row, ignore_index = True)



F1s = []
zF1s = []
meansFFT = []
stdsFFT = []

for fti in fts:
    amp_ft = Sxx.real[faxis == fti]
    mean = np.mean(Sxx.real)
    std = np.std(Sxx.real)
    zamp_ft = (amp_ft - mean)/std
    F1s.append(amp_ft)
    zF1s.append(zamp_ft)
    meansFFT.append(mean)
    stdsFFT.append(std)





# fig,ax = plt.subplots(5,5, sharey = True,figsize = (20,16))

# for nbs in range(nb_stimuli):

#         y = int(np.floor(nbs/5))
#         x = int(nbs%5)

#         fx = float(fxs[y])
#         ft = float(fts[x])

#         ax[x,y].plot(time_pad,resps_pad[nbs])
#         ax[x,y].plot(time,resps[nbs])
#         ax[-1,y].set_xlabel('time [s]')
#         ax[x,0].set_ylabel(f'ft \n {ft} Hz')
#         ax[0,y].set_title(f'fx \n {fx} Hz')

# plt.show()





fig,ax = plt.subplots(5,5, sharex = True, sharey = True, figsize = (20,16))

X = 0
for nbs in range(nb_stimuli):

        y = int(np.floor(nbs/5))
        x = int(nbs%5)

        fx = float(fxs[y])
        ft = float(fts[x])

        if X ==0:
                ax[x,y].plot(faxis,spectrums[nbs], label = 'raw')
                ax[x,y].plot(faxis_pad,spectrums_pad[nbs], label = 'padded')

                
        else:
            ax[x,y].plot(faxis,spectrums[nbs])
            ax[x,y].plot(faxis_pad,spectrums_pad[nbs])

        amp_idx = np.round(faxis) == fts[x]
        amp_ft = np.mean(Sxx.real[amp_idx])


        ax[x,y].axvline(ft, color = 'r', label = f'ft = {ft} Hz')
        # try:
        #     ax[x,y].axvline(faxis[amp_idx], color = 'g', label = f'ft = {ft} Hz')
        # except:
        #     for fdx in faxis[amp_idx]:
        #         ax[x,y].axvline(fdx, color = 'g', label = f'ft = {ft} Hz')
        # ax[x,y].axhline(amp_ft, color = 'g', label = f'ft = {ft} Hz')
        ax[x,y].legend()
        # ax[x,y].set_xlim(1,10)
        print(faxis[spectrums[nbs].argmax()])
        ax[-1,y].set_xlabel('frequency [Hz]')
        ax[x,0].set_ylabel(f'ft \n {ft} Hz')
        ax[0,y].set_title(f'fx \n {fx} Hz')


plt.show()


x = 0

# fig = plt.figure(figsize = (16,20))
# gs = fig.add_gridspec(5,4)
# fig.subplots_adjust(top=0.92,
#                     bottom=0.045,
#                     left=0.11,
#                     right=0.9,
#                     hspace=4.0,
#                     wspace=0.5)



# for fxx,fx in enumerate(fxs):

#     ax = fig.add_subplot(gs[fxx,0])
#     ax2 = fig.add_subplot(gs[fxx,1])

#     mean_ft = frequency_df[frequency_df['fx' ] == fx]['mean_stim']
#     F1_ft = frequency_df[frequency_df['fx' ] == fx]['F1']
#     ft_ft = frequency_df[frequency_df['fx' ] == fx]['ft']

   
#     ax.scatter(ft_ft, mean_ft, label = f'fx = {fx}')
#     ax.plot(ft_ft, mean_ft, label = f'fx = {fx}', linestyle = '--')
#     ax.set_title('mean amplitude ')

#     ax2.scatter(ft_ft, F1_ft, label = f'fx = {fx}')
#     ax2.plot(ft_ft, F1_ft, label = f'fx = {fx}', linestyle = '--')

#     ax2.set_title('power at ft')

#     ax.set_xlabel('ft')
#     ax.set_ylabel('ft')
#     ax2.set_xlabel('ft')
#     ax.legend()
#     ax2.legend()

# plt.show()


# x = 0
import h5py
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.ndimage import gaussian_filter
import seaborn as sns
import pickle
from tqdm import tqdm

import json
import matplotlib.colors as colors
import matplotlib.cm as cmx
import os



exp_name = 'MR-0609'
fp = f'/user/sebert/home/Documents/Experiments/Spatiotemporal_tuning_curves/{exp_name}/results'
# fpsta = f'/user/sebert/home/Documents/Experiments/Spatiotemporal_tuning_curves/{exp_name}/STA/{exp_name}/fitting'

framerate = 60.
stim_length = 35.0
bin_size = 0.001
fontsize = 20

time_PSTH = np.arange(0,stim_length,bin_size)
time_stim = np.arange(0,stim_length,1/framerate)

features = pd.read_csv(f'{fp}/features.csv')
response = h5py.File(f'{fp}/response.hdf5')
keys = list(features['Unnamed: 0'].values)

#load data picke: 
fpdata = f'/user/sebert/home/Documents/Experiments/Spatiotemporal_tuning_curves/Results/{exp_name}/data.pkl'

with open(fpdata, "rb") as handle:   #Pickling
    data = pickle.load(handle)


for key in tqdm(keys): 

    fpkey = f'{fp}/spiketimes/{key}_trials.json'
    # fpkeysta = f'{fpsta}/{key}_fitting.json'



    typ = features['flash_type'][features['Unnamed: 0']== key].values[0]

    try:
        with open(fpkey, 'r', encoding='utf-8') as handle:
            raster = json.load(handle)

        # with open(fpkeysta, 'r', encoding='utf-8') as handle:
        #     stadat = json.load(handle)


        PSTH  = raster['chirp_psth']
        time = np.arange(0,len(raster['chirp_psth']),1)*0.06
        stim = response[key]['chirp_signal']['full_signal'][()]

        fig = plt.figure(figsize = (20,16))

        fig.subplots_adjust(top=0.88,
                            bottom=0.11,
                            left=0.11,
                            right=0.9,
                            hspace=0.2,
                            wspace=1.0
                            )
        gs = fig.add_gridspec(2,4)

        axs = fig.add_subplot(gs[0,1:])
        axs.plot(time_stim,stim)
        axs.set_title('stimulus', loc= 'left')


        ax = fig.add_subplot(gs[1,1:])
        ax.plot(time_PSTH,PSTH)
        ax.set_xlabel('time [s]')
        ax.set_ylabel('Firing Rate [Hz]')
        ax.set_title('PSTH', loc= 'left')


        ax0 = fig.add_subplot(gs[:,0], frameon = False)
        ax0.set_xticks([])
        ax0.set_yticks([])

        bias = np.round(features ['bias_idx'][features['Unnamed: 0']== key].values[0],2)
        ax0.text(0.0,0.9,f'{typ}, bias : {bias}',fontsize = fontsize, fontweight = 'bold')


        data[key]['type'] = typ
        data[key]['bias'] = bias


        if typ == 'ON':
            suidx = np.round(features['on_sust_index'][features['Unnamed: 0']== key].values[0],2)
            cutoff = np.round(features['on_freq_fcut'][features['Unnamed: 0']== key].values[0],2)
            lat = np.round(features['on_latency'][features['Unnamed: 0']== key].values[0],2)

            ax0.text(0.0,0.8,f'sustained index ',fontsize = fontsize)
            ax0.text(0.0,0.75,f'{suidx}',fontsize = fontsize)
            ax0.text(0.0,0.6,f'cutoff frequency ',fontsize = fontsize)
            ax0.text(0.0,0.55,f'{cutoff} Hz',fontsize = fontsize)
            ax0.text(0.0,0.4,f'latency ',fontsize = fontsize)
            ax0.text(0.0,0.35,f'{lat} ms',fontsize = fontsize)

            data[key]['on_sust_index'] = suidx
            data[key]['on_freq_fcut'] = cutoff
            data[key]['on_latency'] = lat

        elif typ == 'OFF':
            suidx = np.round(features['off_sust_index'][features['Unnamed: 0']== key].values[0],2)
            cutoff = np.round(features['off_freq_fcut'][features['Unnamed: 0']== key].values[0],2)
            lat = np.round(features['off_latency'][features['Unnamed: 0']== key].values[0],2)
        
            ax0.text(0.0,0.8,f'sustained index ',fontsize = fontsize)
            ax0.text(0.0,0.75,f'{suidx}',fontsize = fontsize)
            ax0.text(0.0,0.6,f'cutoff frequency ',fontsize = fontsize)
            ax0.text(0.0,0.55,f'{cutoff} Hz',fontsize = fontsize)
            ax0.text(0.0,0.4,f'latency ',fontsize = fontsize)
            ax0.text(0.0,0.35,f'{lat} ms',fontsize = fontsize)

            data[key]['off_sust_index'] = suidx
            data[key]['off_freq_fcut'] = cutoff
            data[key]['off_latency'] = lat


        else:
            suidx_off = np.round(features['off_sust_index'][features['Unnamed: 0']== key].values[0],2)
            cutoff_off = np.round(features['off_freq_fcut'][features['Unnamed: 0']== key].values[0],2)
            lat_off = np.round(features['off_latency'][features['Unnamed: 0']== key].values[0],2)

            suidx_on = np.round(features['on_sust_index'][features['Unnamed: 0']== key].values[0],2)
            cutoff_on = np.round(features['on_freq_fcut'][features['Unnamed: 0']== key].values[0],2)
            lat_on = np.round(features['on_latency'][features['Unnamed: 0']== key].values[0],2)

            ax0.text(0.0,0.8,f'sustained index ',fontsize = fontsize)
            ax0.text(0.0,0.75,f'ON {suidx_on}, OFF {suidx_off}',fontsize = fontsize)
            ax0.text(0.0,0.6,f'cutoff frequency ',fontsize = fontsize)
            ax0.text(0.0,0.55,f'ON {cutoff_on} Hz, OFF {cutoff_off} Hz',fontsize = fontsize)
            ax0.text(0.0,0.4,f'latency',fontsize = fontsize)
            ax0.text(0.0,0.35,f'ON {lat_on} ms , OFF {lat_off} ms',fontsize = fontsize)
        
            data[key]['on_sust_index'] = suidx_on
            data[key]['on_freq_fcut'] = cutoff_on
            data[key]['on_latency'] = lat_on

            data[key]['off_sust_index'] = suidx_off
            data[key]['off_freq_fcut'] = cutoff_off
            data[key]['off_latency'] = lat_off


        if typ == 'ON':
            POL = 'ON'

        if typ == 'OFF':
            POL = 'OFF'

        if typ == 'ON/OFF':
            POL = 'ONOFF'


        fpout = f'/user/sebert/home/Documents/Experiments/Spatiotemporal_tuning_curves/Results/{exp_name}/plots/chirp/{POL}'
        
        if not os.path.isdir(fpout):
            os.mkdir(fpout)

        fig.suptitle(f'{POL}_{key}')
        fig.savefig(f'{fpout}/{POL}_{key}.png')
        plt.close()


    except:
        print(f'{key} json file not found, deleted')
        
        # remove unit from data dict
        del data[key]

        continue
    #plt.show()



with open(fpdata, "wb") as handle:   #Pickling
    pickle.dump(data, handle,protocol=4 )
  


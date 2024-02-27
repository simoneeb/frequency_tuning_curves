import h5py
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.ndimage import gaussian_filter
import seaborn as sns
import pickle
from tqdm import tqdm

import matplotlib.colors as colors
import matplotlib.cm as cmx


# load data 
fpdata = '/user/sebert/home/Documents/Experiments/Spatiotemporal_tuning_curves/Simone/MR-0605/results/data.pkl'
with open(fpdata, "rb") as handle:   #Pickling
    data = pickle.load(handle)

keys = list(data.keys())[3:]

intensities = list(data['stimuli'].keys())
grating_types = list(data['stimuli'][intensities[0]].keys())


fp = f'/user/sebert/home/Documents/Experiments/Spatiotemporal_tuning_curves/Simone/MR-0605/sync/event_list_MR-0605_.csv'
events = pd.read_csv(fp)


# group by stimulus
g = events.groupby('protocol_name')
protocols = list(g.groups.keys())
print(f'protocols : {protocols}')



# group by intensity
events_gratings = g.get_group("gratings")
gi = events_gratings.groupby('nd')
intensities = list(gi.groups.keys())
print(f'intensitites : {intensities}')


sampling_frequency = 20000
dt = 0.06*sampling_frequency

frame_duration = 1/59.9 #s
stim_duration = 60*frame_duration
#stim_duration = 1.0


# in frames
dt = 0.02*sampling_frequency
stim_length = events_gratings['event_duration'].values[0]
time = np.arange(0,stim_length,dt)

# in seconds
dt_sec = 0.02
stim_length_sec = events_gratings['event_duration'].values[0]/sampling_frequency
time_sec = np.arange(0,stim_length_sec,dt_sec)

bin_edges = np.append(time_sec, 2 * time_sec[-1] - time_sec[-2])


stim_length_ot = 2*stim_duration
time_ot = np.arange(0,stim_length_ot,dt_sec)
bin_edges_ot = np.append(time_ot, 2 * time_ot[-1] - time_ot[-2])



nb_stimuli = 25


fxs = [1,2,4,6,10]
fts = [2,4,5,6,8]

fxs = np.array([1,2,4,6,10])
fts = np.array([2,4,5,6,8])

vss = []
for i in range(len(fxs)):
    vs = fts/fxs[i]
    vss.append(vs)




for key in tqdm(keys):
       
    # ========================================================================================================================================================================
    # FlICKERING ALL STIMS
    # ========================================================================================================================================================================

    fpplots = '/user/sebert/home/Documents/Experiments/Spatiotemporal_tuning_curves/Simone/MR-0605/results/plots_all_stims/flickering_grating'

    # resp_type = 'rasters_sorted_stim'
    colors_flicker = ['teal','turquoise', 'cyan']
    grat = grating_types[0]

    
    resp_type = 'counts_sorted_stim_alinged'
    #resp_type = 'counts_sorted_stim'

    fig,ax = plt.subplots(5,5, sharey = True,figsize = (20,16))
    heatdf_flicker = pd.DataFrame(columns = ['fx', 'ft','v' 'max','mean','intensity'])

    for nbs in range(nb_stimuli):

        for ix,i in enumerate(intensities):

            row ={}

            y = int(np.floor(nbs/5))
            row['fx'] = float(fxs[y])
            
            x = int(nbs%5)
            row['ft'] = float(fts[x])
            row['v'] = float(fts[x]/fxs[y])

            row['intensity'] = i

            resp = data[key][i][grat][resp_type][nbs]
            stim = data['stimuli'][i][grat]['stimuli_sorted_stim'][nbs]

            start = data['stimuli'][i][grat]['starts_sorted_stim'][nbs]
            #end = data['stimuli'][i][grat]['ends_sorted_stim'][nbs]
            stim_trig = data['stimuli'][i][grat]['stimuli_sorted_stim_aligned'][nbs]

            idx = int(len(resp)/2)
            
            mean_stim = np.mean(resp[:idx])
            mean_base = np.mean(resp[idx:])
            mean_ratio = mean_stim-mean_base

            data[key][i][grat]['means_stim'] = mean_stim
            data[key][i][grat]['means_base'] = mean_base
            data[key][i][grat]['means_ratio'] = mean_ratio

            row['max'] = float(resp.max())
            row['mean_stim'] = float(mean_stim)
            row['mean_base'] = float(mean_base)
            row['mean_ratio'] = float(mean_ratio)

            heatdf_flicker = heatdf_flicker.append(row, ignore_index=True)
            #ax[x,y].eventplot(resp)
            if nbs == 0:
                ax[x,y].plot((time_ot+0.5*dt_sec) ,resp, color = colors_flicker[ix], label = f'{i}')
                ax[x,y].axvline(time_ot[idx], linestyle = ':', color = 'k', label = 'stimulus_end')

            else:
                ax[x,y].plot((time_ot+0.5*dt_sec) ,resp, color = colors_flicker[ix])
                ax[x,y].axvline(time_ot[idx], linestyle = ':', color = 'k')



    ax[0,0].set_title('fx = 1 c/mm')
    ax[0,1].set_title('fx = 2 c/mm')
    ax[0,2].set_title('fx = 4 c/mm')
    ax[0,3].set_title('fx = 6 c/mm')
    ax[0,4].set_title('fx = 10 c/mm')


    ax[0,0].set_ylabel('ft = 2 1/s')
    ax[1,0].set_ylabel('ft = 4 1/s')
    ax[2,0].set_ylabel('ft = 5 1/s')
    ax[3,0].set_ylabel('ft = 6 1/s')
    ax[4,0].set_ylabel('ft = 8 1/s')

    ax[4,0].set_xlabel('time [s]')
    ax[4,1].set_xlabel('time [s]')
    ax[4,2].set_xlabel('time [s]')
    ax[4,3].set_xlabel('time [s]')
    ax[4,4].set_xlabel('time [s]')


    fig.legend()
    fig.suptitle(f'Firing Rate Responses to {grat} gratings')


    fig.legend()
    fig.savefig(f'{fpplots}/{key}')
    plt.close()


    # ========================================================================================================================================================================
    # MOVING ALL STIMS
    # ========================================================================================================================================================================

    fpplots = '/user/sebert/home/Documents/Experiments/Spatiotemporal_tuning_curves/Simone/MR-0605/results/plots_all_stims/moving_grating'

    # resp_type = 'rasters_sorted_stim'
    colors_moving = ['plum','magenta','darkmagenta']

    grat = grating_types[1]

    fig,ax = plt.subplots(5,5, sharey = True, figsize = (20,16))
    fxs = [1,2,4,6,10]
    fts = [2,4,5,6,8]

    heatdf_moving = pd.DataFrame(columns = ['fx', 'ft', 'max','intensity'])

    for nbs in range(nb_stimuli):

        for ix,i in enumerate(intensities):
            row ={}

            y = int(np.floor(nbs/5))
            row['fx'] = float(fxs[y])
            
            x = int(nbs%5)
            row['ft'] = float(fts[x])
            row['v'] = float(fts[x]/fxs[y])

            row['intensity'] = i

            resp = data[key][i][grat][resp_type][nbs]
            stim = data['stimuli'][i][grat]['stimuli_sorted_stim']
            
            mean_stim = np.mean(resp[:idx])
            mean_base = np.mean(resp[idx:])
            mean_ratio = mean_stim-mean_base

            data[key][i][grat]['means_stim'] = mean_stim
            data[key][i][grat]['means_base'] = mean_base
            data[key][i][grat]['means_ratio'] = mean_ratio

            row['max'] = float(resp.max())
            row['mean_stim'] = float(mean_stim)
            row['mean_base'] = float(mean_base)
            row['mean_ratio'] = float(mean_ratio)

            heatdf_moving = heatdf_moving.append(row, ignore_index=True)
            #ax[x,y].eventplot(resp)
            if nbs == 0:
                ax[x,y].plot((time_ot+0.5*dt_sec),resp, color = colors_moving[ix], label = f'{i}')
            else:
                ax[x,y].plot((time_ot+0.5*dt_sec),resp, color = colors_moving[ix])
            #ax[x,y].plot(stim, color = colors_moving[ix])

    ax[0,0].set_title('fx = 1 c/mm')
    ax[0,1].set_title('fx = 2 c/mm')
    ax[0,2].set_title('fx = 4 c/mm')
    ax[0,3].set_title('fx = 6 c/mm')
    ax[0,4].set_title('fx = 10 c/mm')

    ax[0,0].set_ylabel('ft = 2 1/s')
    ax[1,0].set_ylabel('ft = 4 1/s')
    ax[2,0].set_ylabel('ft = 5 1/s')
    ax[3,0].set_ylabel('ft = 6 1/s')
    ax[4,0].set_ylabel('ft = 8 1/s')

    ax[4,0].set_xlabel('time [s]')
    ax[4,1].set_xlabel('time [s]')
    ax[4,2].set_xlabel('time [s]')
    ax[4,3].set_xlabel('time [s]')
    ax[4,4].set_xlabel('time [s]')

    fig.legend()
    fig.suptitle(f'Firing Rate Responses to {grat} gratings')

    fig.legend()
    fig.savefig(f'{fpplots}/{key}')
    plt.close()


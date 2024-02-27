import h5py
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.ndimage import gaussian_filter
import seaborn as sns
import pickle
from tqdm import tqdm
import os

import matplotlib.colors as colors
import matplotlib.cm as cmx


# load data 
exp_name = 'MR-0609'
fpdata = f'/user/sebert/home/Documents/Experiments/Spatiotemporal_tuning_curves/Results/{exp_name}/data.pkl'
with open(fpdata, "rb") as handle:   #Pickling
    data = pickle.load(handle)


keys = list(data.keys())[3:]

intensities = list(data['stimuli'].keys())
grating_types = list(data['stimuli'][intensities[0]].keys())
contrasts = ['cm10', 'cm05', 'cm02']

fp = f'/user/sebert/home/Documents/Experiments/Spatiotemporal_tuning_curves/{exp_name}/sync/event_list_{exp_name}_.csv'
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

stim_duration = 1.0

# in frames
dt = 0.02*sampling_frequency
stim_length = events_gratings['event_duration'].values[0]
time = np.arange(0,stim_length,dt)

# in seconds
dt_sec = 0.02
stim_length_sec = events_gratings['event_duration'].values[0]/sampling_frequency
time_sec = np.arange(0,stim_length_sec,dt_sec)

bin_edges = np.append(time_sec, 2 * time_sec[-1] - time_sec[-2])


# for one trial
stim_length_ot = 2*stim_duration
time_ot = np.arange(0,stim_length_ot,dt_sec)
bin_edges_ot = np.append(time_ot, 2 * time_ot[-1] - time_ot[-2])


# params
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


    if  data[key]['response'] == 'yes':
        #print(f'{key} yes')

        
        # ========================================================================================================================================================================
        # FlICKERING ALL STIMS
        # ========================================================================================================================================================================

        # resp_type = 'rasters_sorted_stim'
        colors_flicker = ['teal','turquoise', 'cyan']
        grating_type = 'flickering'

        resp_type = 'counts_sorted_stim_alinged'
        #resp_type = 'counts_sorted_stim'

        fig,ax = plt.subplots(5,11, sharey = True,figsize = (20,16))

        i = 'nd4'

        for nbs in range(nb_stimuli):

            for ix,cm in enumerate(contrasts):

                grat = f'{grating_type}_{cm}'

                y = int(np.floor(nbs/5))
                
                x = int(nbs%5)
               

                resp = data[key][i][grat][resp_type][nbs]
                stim = data['stimuli'][i][grat]['stimuli_sorted_stim'][nbs]

                start = data['stimuli'][i][grat]['starts_sorted_stim'][nbs]
                #end = data['stimuli'][i][grat]['ends_sorted_stim'][nbs]
                stim_trig = data['stimuli'][i][grat]['stimuli_sorted_stim_aligned'][nbs]

                idx = int(len(resp)/2)
            

                #ax[x,y].eventplot(resp)
                if nbs == 0 :
                    ax[x,y].plot((time_ot+0.5*dt_sec) ,resp, color = colors_flicker[ix], label = f'{grating_type} {cm}')

                    if ix == 2:
                        ax[x,y].axvline(time_ot[idx], linestyle = ':', color = 'k', label = 'stimulus_end')

                else:
                    ax[x,y].plot((time_ot+0.5*dt_sec) ,resp, color = colors_flicker[ix])
                    if ix == 0:
                        ax[x,y].axvline(time_ot[idx], linestyle = ':', color = 'k')

                ax[x,5].spines['top'].set_visible(False)
                ax[x,5].spines['right'].set_visible(False)
                ax[x,5].spines['bottom'].set_visible(False)
                ax[x,5].spines['left'].set_visible(False)

                ax[x,5].set_xticks([])
                ax[x,5].set_yticks([])



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


        #fig.suptitle(f'Firing Rate Responses to {grat} gratings')
        #fig.text(2000, 2000,f'{grating_type} gratings', ha = 'left')
        #fig.suptitle(f'{grating_type} gratings', ha = 'left')


        # fig.legend()
        # fig.savefig(f'{fpplots}/{key}')
        # plt.close()


        # ========================================================================================================================================================================
        # MOVING ALL STIMS
        # ========================================================================================================================================================================

        
        #fpplots = '/user/sebert/home/Documents/Experiments/Spatiotemporal_tuning_curves/Simone/MR-0605/results/plots_all_stims/moving_grating'

        # resp_type = 'rasters_sorted_stim'
        colors_moving = ['plum','magenta','darkmagenta']

        grating_type = 'moving'

        #fig,ax = plt.subplots(5,5, sharey = True, figsize = (20,16))
        fxs = [1,2,4,6,10]
        fts = [2,4,5,6,8]


        for nbs in range(nb_stimuli):

            for ix,cm in enumerate(contrasts):

                grat = f'{grating_type}_{cm}'

                y = 6+int(np.floor(nbs/5))
                
                x = int(nbs%5)

                resp = data[key][i][grat][resp_type][nbs]
                stim = data['stimuli'][i][grat]['stimuli_sorted_stim']
                
                            #ax[x,y].eventplot(resp)
                if nbs == 0:
                    ax[x,y].plot((time_ot+0.5*dt_sec),resp, color = colors_moving[ix], label = f'{grating_type} {cm}')

                    if ix == 2:
                        ax[x,y].axvline(time_ot[idx], linestyle = ':', color = 'k')
                else:
                    ax[x,y].plot((time_ot+0.5*dt_sec),resp, color = colors_moving[ix])

                    if ix == 0:
                        ax[x,y].axvline(time_ot[idx], linestyle = ':', color = 'k')
                #ax[x,y].plot(stim, color = colors_moving[ix])

        ax[0,6].set_title('fx = 1 c/mm')
        ax[0,7].set_title('fx = 2 c/mm')
        ax[0,8].set_title('fx = 4 c/mm')
        ax[0,9].set_title('fx = 6 c/mm')
        ax[0,10].set_title('fx = 10 c/mm')

        ax[0,6].set_ylabel('ft = 2 1/s')
        ax[1,6].set_ylabel('ft = 4 1/s')
        ax[2,6].set_ylabel('ft = 5 1/s')
        ax[3,6].set_ylabel('ft = 6 1/s')
        ax[4,6].set_ylabel('ft = 8 1/s')

        ax[4,6].set_xlabel('time [s]')
        ax[4,7].set_xlabel('time [s]')
        ax[4,8].set_xlabel('time [s]')
        ax[4,9].set_xlabel('time [s]')
        ax[4,10].set_xlabel('time [s]')




        #fig.suptitle(f'Firing Rate Responses to {grat} gratings')
        #fig.suptitle(f'{grating_type} gratings',ha = 'right')

        fig.legend()

        try:

            if data[key]['type'] == 'ON':
                POL = 'ON'

            if data[key]['type'] == 'OFF':
                POL = 'OFF'

            if data[key]['type'] == 'ON/OFF':
                POL = 'ONOFF'

            fpplots = f'/user/sebert/home/Documents/Experiments/Spatiotemporal_tuning_curves/Results/{exp_name}/plots/gratings/compare_contrasts/{POL}'

            if not os.path.isdir(fpplots):
                os.mkdir(fpplots)


            #plt.show()

            fig.savefig(f'{fpplots}/{key}.png')

        except:
            print(f'{key} not found')
            
        plt.close()
    
    else:
        continue


import h5py
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.ndimage import gaussian_filter
from scipy.stats import skewnorm
import seaborn as sns
import pickle
from tqdm import tqdm
import os

from lmfit.models import SkewedGaussianModel


import matplotlib.colors as colors
import matplotlib.cm as cmx


# load data 
exp_name = 'MR-0609'
fpdata = f'/Users/simone/Documents/Experiments/Spatiotemporal_tuning_curves/Results/{exp_name}/data.pkl'
with open(fpdata, "rb") as handle:   #Pickling
    data = pickle.load(handle)

keys = list(data.keys())[1:-3]

intensities = list(data['stimuli'].keys())
grating_types = list(data['stimuli'][intensities[0]].keys())
contrasts = ['cm10', 'cm05', 'cm02']

fp = f'/Users/simone/Documents/Experiments/Spatiotemporal_tuning_curves/{exp_name}/sync/event_list_{exp_name}_.csv'
events = pd.read_csv(fp)

fpdataheat = f'/Users/simone/Documents/Experiments/Spatiotemporal_tuning_curves/Results/{exp_name}/dataframe.csv'
frequency_df = pd.read_csv(fpdataheat)


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


colors_flicker = ['teal','turquoise', 'cyan']
colors_moving = ['plum','magenta','darkmagenta']

resp_type = 'counts_sorted_stim_alinged'
i = 'nd4'

#keys = ['Unit_0009', 'Unit_0125']
#keys = ['Unit_0114', 'Unit_0125']




def norma(x):
    x = np.asarray(x)

    return (x-np.nanmin(x))/(np.nanmax(x)-np.nanmin(x))

for key in tqdm(keys):
    

    try:
        if data[key]['type'] == 'ON':
            POL = 'ON_with_power'
        if data[key]['type'] == 'OFF':
            POL = 'OFF_with_power'
        if data[key]['type'] == 'ON/OFF':
            POL = 'ONOFF_wth_power'
        print(f' analyzing {key}')


    except:
        print(f'{key} not found')
        continue
    



    if  data[key]['response'] == 'yes':
        
        # ========================================================================================================================================================================
        # FlICKERING ALL STIMS
        # ========================================================================================================================================================================

        # for ix,cm in enumerate(contrasts):
            
        #     #fig,ax = plt.subplots(10,14,sharey = 'row', sharex = 'row', figsize = (20,16))
        #     fig = plt.figure(figsize = (16,20))
        #     # fig.subplots_adjust(top=0.96,
        #     #                     bottom=0.07,
        #     #                     left=0.035,
        #     #                     right=0.9,
        #     #                     hspace=0.89,
        #     #                     wspace=0.8)

        #     gs = fig.add_gridspec(10,5)

        #     for nbs in range(nb_stimuli):

        #         grating_type = 'flickering'

        #         grat = f'{grating_type}_{cm}'
        #         row ={}

        #         # loop over spatial frequencies in rows
        #         y = int(nbs%5)
        #         ft = float(fts[y])

        #         # loop over spatial frequencies in  columns
        #         x = int(np.floor(nbs/5))
        #         fx = fxs[x]


        #         resp = data[key][i][grat][resp_type][nbs]
        #         fax,pow = data[key][i][grat]['power_spectrum'][nbs]
        #         zf1 = data[key][i][grat]['F_close_amp'][nbs]
        #         zf1_idx = data[key][i][grat]['F_close_idx'][nbs]
        #         zf1_f = data[key][i][grat]['F_close_f'][nbs]

        #         zf1_m = data[key][i][grat]['F_max_amp'][nbs]
        #         zf1_idx_m = data[key][i][grat]['F_max_idx'][nbs]
        #         zf1_f_m = data[key][i][grat]['F_max_f'][nbs]
        #         stim = data['stimuli'][i][grat]['stimuli_sorted_stim'][nbs]

        #         start = data['stimuli'][i][grat]['starts_sorted_stim'][nbs]
        #         stim_trig = data['stimuli'][i][grat]['stimuli_sorted_stim_aligned'][nbs]

        #         idx = int(len(resp)/2)

                
        #         mean_stim = np.mean(resp[:idx])
        #         mean_base = np.mean(resp[idx:])
        #         std_base = np.std(resp[idx:])
        #         mean_ratio = mean_stim-mean_base



        #         if nbs == 0 :

        #             ax0 = fig.add_subplot(gs[x*2,y])
        #             ax1 = fig.add_subplot(gs[x*2+1,y])

        #             ax0.plot((time_ot+0.5*dt_sec) ,resp, color = colors_flicker[ix])
        #             ax0.axvline(time_ot[idx], linestyle = ':', color = 'k')

        #             ax1.axvline(ft, linestyle = ':', color = 'k', label = 'stimulus frequency')
        #             try:
        #                 ax1.axvline(fax[zf1_idx], linestyle = ':', color = 'r', label = 'closest peak')
        #             except:
        #                 None

        #             try:
        #                 ax1.axvline(fax[zf1_idx_m], linestyle = ':', color = 'g', label = 'highest local peak')
        #             except:
        #                 None

        #             ax1.plot(fax,pow.real, color = 'k')
        #             #ax1.legend()
        #             ax1.set_xlabel('frequency [Hz]')
        #             ax0.set_xlabel('time [s]')

        #             ax1.set_xlim(0,9)


        #         else:
        #             ax0 = fig.add_subplot(gs[x*2,y], sharey = ax0)
        #             ax1 = fig.add_subplot(gs[x*2+1,y], sharey = ax1)

        #             ax0.plot((time_ot+0.5*dt_sec) ,resp, color = colors_flicker[ix])
        #             ax0.axvline(time_ot[idx], linestyle = ':', color = 'k')
        #             ax1.plot(fax,pow.real, color = 'k')

        #             try:
        #                 ax1.axvline(fax[zf1_idx], linestyle = ':', color = 'r')
        #             except:
        #                 None

        #             try:
        #                 ax1.axvline(fax[zf1_idx_m], linestyle = ':', color = 'g')
        #             except:
        #                 None


        #             ax1.axvline(ft, linestyle = ':', color = 'k') 

        #             ax1.set_xlabel('frequency [Hz]')
        #             ax0.set_xlabel('time [s]')

        #             ax1.set_xlim(0,9)

        #         if x == 0:
        #             ax0.set_title(f'ft = {ft} 1/s')
        #         if y  == 0 :
        #             ax0.set_ylabel(f'fx = {fx} c/mm \n PSTH')
        #             ax1.set_ylabel(f'fx = {fx} c/mm \n power spectrum')

        #         fig.suptitle('PSTH and power spectrum to all flickering stimuli')
        #         fig.legend()


        #         if data[key]['type'] == 'ON':
        #             POL = 'ON_with_power'
        #         if data[key]['type'] == 'OFF':
        #             POL = 'OFF_with_power'
        #         if data[key]['type'] == 'ON/OFF':
        #             POL = 'ONOFF_wth_power'

        #     fpplots = f'/Users/simone/Documents/Experiments/Spatiotemporal_tuning_curves/Results/{exp_name}/plots/gratings/{cm}/{POL}/flickering'

        #     if not os.path.isdir(fpplots):
        #         os.mkdir(fpplots)


        #     #plt.show()

        #     #fig.savefig(f'{fpplots}/{key}_with_power.png')
        #     plt.close()




        # ========================================================================================================================================================================
        # MOVING ALL STIMS
        # ========================================================================================================================================================================
        for ix,cm in enumerate(contrasts):
            
            #fig,ax = plt.subplots(10,14,sharey = 'row', sharex = 'row', figsize = (20,16))
            fig = plt.figure(figsize = (16,20))
            # fig.subplots_adjust(top=0.96,
            #                     bottom=0.07,
            #                     left=0.035,
            #                     right=0.9,
            #                     hspace=0.89,
            #                     wspace=0.8)

            gs = fig.add_gridspec(10,5)

          
            for nbs in range(nb_stimuli):

                grating_type = 'moving'

                grat = f'{grating_type}_{cm}'


                # loop over spatial frequencies in rows
                y = int(nbs%5)
                ft = float(fts[y])

            
                # loop over spatial frequencies in  columns
                x = int(np.floor(nbs/5))
                fx = fxs[x]
            

                resp = data[key][i][grat][resp_type][nbs]
                fax,pow = data[key][i][grat]['power_spectrum'][nbs]
                zf1 = data[key][i][grat]['F_close_amp'][nbs]
                zf1_idx = data[key][i][grat]['F_close_idx'][nbs]
                zf1_f = data[key][i][grat]['F_close_f'][nbs]

                zf1_m = data[key][i][grat]['F_max_amp'][nbs]
                zf1_idx_m = data[key][i][grat]['F_max_idx'][nbs]
                zf1_f_m = data[key][i][grat]['F_max_f'][nbs]

                stim = data['stimuli'][i][grat]['stimuli_sorted_stim']



                idx = int(len(resp)/2)

                
                mean_stim = np.mean(resp[:idx])
                mean_base = np.mean(resp[idx:])
                std_base = np.std(resp[idx:])
                mean_ratio = mean_stim-mean_base


                if nbs == 0:

                    ax2 = fig.add_subplot(gs[x*2,y])
                    ax3 = fig.add_subplot(gs[x*2+1,y])

                    ax2.plot((time_ot+0.5*dt_sec),resp, color = colors_moving[ix])
                    ax2.axvline(time_ot[idx], linestyle = ':', color = 'k', label = 'stimulus_end')
                    ax3.plot(fax,pow.real, color = 'k')
                    try:
                        ax3.axvline(fax[zf1_idx], linestyle = ':', color = 'r', label = 'closest peak')
                    except:
                        None

                    try:
                        ax3.axvline(fax[zf1_idx_m], linestyle = ':', color = 'g', label = 'highest peak')
                    except:
                        None

                    ax3.axvline(ft, linestyle = ':', color = 'k', label = 'stimulus frequency')

                    ax3.set_xlabel('frequency [Hz]')
                    ax2.set_xlabel('time [s]')

                    ax3.set_xlim(0,9)




                else:
                    ax2 = fig.add_subplot(gs[x*2,y], sharey = ax2)
                    ax3 = fig.add_subplot(gs[x*2+1,y], sharey = ax3)

                    ax2.plot((time_ot+0.5*dt_sec),resp, color = colors_moving[ix])
                    ax2.axvline(time_ot[idx], linestyle = ':', color = 'k')
                    ax3.plot(fax,pow.real, color = 'k')
                    ax3.axvline(ft, linestyle = ':', color = 'k')

                    try:
                        ax3.axvline(fax[zf1_idx], linestyle = ':', color = 'r')
                    except:
                        None

                    try:
                        ax3.axvline(fax[zf1_idx_m], linestyle = ':', color = 'g')
                    except:
                        None

                    ax3.set_xlabel('frequency [Hz]')
                    ax2.set_xlabel('time [s]')
                    ax3.set_xlim(0,9)


                if x == 0:
                    ax2.set_title(f'ft = {ft} 1/s')
                if  y == 0:
                    ax2.set_ylabel(f'fx = {fx} c/mm \n PSTH')
                    ax3.set_ylabel(f'fx = {fx} c/mm \n power spectrum')


                fig.suptitle('PSTH and power spectrum to all flickering stimuli')
                fig.legend

                if data[key]['type'] == 'ON':
                    POL = 'ON_with_power'
                if data[key]['type'] == 'OFF':
                    POL = 'OFF_with_power'
                if data[key]['type'] == 'ON/OFF':
                    POL = 'ONOFF_wth_power'

            fpplots = f'/Users/simone/Documents/Experiments/Spatiotemporal_tuning_curves/Results/{exp_name}/plots/gratings/{cm}/{POL}/moving'

            if not os.path.isdir(fpplots):
                os.mkdir(fpplots)


            #plt.show()


            fig.savefig(f'{fpplots}/{key}_with_power.png')
    

            plt.close()

        else:
            continue


# save new data
with open(fpdata, "wb") as handle:   #Pickling
    pickle.dump(data,handle, protocol=4)



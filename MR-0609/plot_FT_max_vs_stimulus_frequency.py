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


fpdataheat = f'/user/sebert/home/Documents/Experiments/Spatiotemporal_tuning_curves/Results/{exp_name}/dataframe.csv'
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

for key in tqdm(keys):



    if  data[key]['response'] == 'yes':
        
        # ========================================================================================================================================================================
        # FlICKERING ALL STIMS
        # ========================================================================================================================================================================



        fig = plt.figure(figsize = (12,12))
        fig.suptitle(f'highest peaks found in the power spect')
        fig.subplots_adjust(top=0.96,
                                bottom=0.07,
                                left=0.035,
                                right=0.9,
                                hspace=0.3,
                                wspace=0.8)

        gs = fig.add_gridspec(3,2)

        for ix,cm in enumerate(contrasts):
            
            #fig,ax = plt.subplots(10,14,sharey = 'row', sharex = 'row', figsize = (20,16))
            

            zf1s = []
            zf1s_mo = []

            axt = fig.add_subplot(gs[ix,0])
            axt.set_title(f'cm = {cm}, flickering')
            axt.set_xlabel('ft')
            axt.set_ylabel('highest peak in FFT')
            axtm = fig.add_subplot(gs[ix,1])
            axtm.set_title(f'cm = {cm}, moving')
            axt.set_xlabel('ft')
            axt.set_ylabel('highest peak in FFT')

            for nbs in range(nb_stimuli):

                grating_type = 'flickering'

                grat = f'{grating_type}_{cm}'
                row ={}


                # loop over spatial frequencies in rows
                y = int(nbs%5)
                ft = float(fts[y])

                # print(nbs)
                # print(f'y = {y}, ft = {ft}')
            
                # loop over spatial frequencies in  columns
                x = int(np.floor(nbs/5))
                fx = fxs[x]



                # print(f'x = {x}, fx = {fx}')
            
                resp = data[key][i][grat][resp_type][nbs]
                fax,pow = data[key][i][grat]['power_spectrum'][nbs]
                zf1 = data[key][i][grat]['F_close_amp'][nbs]
                zf1_idx = data[key][i][grat]['F_close_idx'][nbs]
                zf1_f = data[key][i][grat]['F_close_f'][nbs]

                zf1_m = data[key][i][grat]['F_max_amp'][nbs]
                zf1_idx_m = data[key][i][grat]['F_max_idx'][nbs]
                zf1_f_m = data[key][i][grat]['F_max_f'][nbs]
                stim = data['stimuli'][i][grat]['stimuli_sorted_stim'][nbs]

                start = data['stimuli'][i][grat]['starts_sorted_stim'][nbs]
                #end = data['stimuli'][i][grat]['ends_sorted_stim'][nbs]
                stim_trig = data['stimuli'][i][grat]['stimuli_sorted_stim_aligned'][nbs]

                idx = int(len(resp)/2)

                
                mean_stim = np.mean(resp[:idx])
                mean_base = np.mean(resp[idx:])
                std_base = np.std(resp[idx:])
                mean_ratio = mean_stim-mean_base

                zf1s.append(zf1_f_m)

            
                axt.plot(fts,fts, color = 'k', linestyle = ':' )

                if y == 4 :
                   

                   # plot stimulus frequency against

                   axt.plot(fts,zf1s, label = f'fx = {fx}' )
                   axt.scatter(fts,zf1s )

                   zf1s = []
                   zf1s_m = []
                axt.legend()


        # ========================================================================================================================================================================
        # MOVING ALL STIMS
        # ========================================================================================================================================================================


                grating_type = 'moving'

                grat = f'{grating_type}_{cm}'


                # loop over spatial frequencies in rows
                y = 7+int(nbs%5)
                ft = float(fts[y-7])

            
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
                
                mean_stim = np.mean(resp[:idx])
                mean_base = np.mean(resp[idx:])
                std_base = np.std(resp[idx:])
                mean_ratio = mean_stim-mean_base

            
                zf1s_mo.append(zf1_f_m)


                axtm.plot(fts,fts, color = 'k', linestyle = ':' )

                if y == 4+7 :

                    axtm.plot(fts,zf1s_mo, label = f'fx = {fx}' )
                    axtm.scatter(fts,zf1s_mo )
                    zf1s = []
                    zf1s_m = []
                    axt.legend()
                    zf1s_mo = []
                    zf1s_mo_m = []





            #fig.suptitle(f'Firing Rate Responses to {grat} gratings')
            #fig.suptitle(f'{grating_type} gratings',ha = 'right')


            try:

                if data[key]['type'] == 'ON':
                    POL = 'ON_with_power'
                if data[key]['type'] == 'OFF':
                    POL = 'OFF_with_power'
                if data[key]['type'] == 'ON/OFF':
                    POL = 'ONOFF_wth_power'

                fpplots = f'/user/sebert/home/Documents/Experiments/Spatiotemporal_tuning_curves/Results/{exp_name}/plots/gratings/FFT/highest'

                if not os.path.isdir(fpplots):
                    os.mkdir(fpplots)


                #plt.show()


                fig.savefig(f'{fpplots}/{key}.png')
    

        

            except:
                print(f'{key} not found')
            

            x = 0

            plt.close()

        else:
            continue



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



def temp_to_unit(key):

    unit = int(key.split('_')[-1])+1

    return 'Unit_{:04d}'.format(unit)




exp_name = 'MR-0609'

nb_trials = 10
trials =np.array([1,2,3,4,5,6,7,8,9])
nb_stimuli = 25

sampling_frequency = 20000.0

# load analog
fp = f'/user/sebert/home/Documents/Experiments/Spatiotemporal_tuning_curves/{exp_name}/analog/{exp_name}_analog.h5'
data_voltf = h5py.File(fp, 'r')
data_volt = data_voltf['Data']
data_volt = data_volt['Recording_0']
data_volt = data_volt['AnalogStream']

stim_full = data_volt['Stream_0']['ChannelData'][0]
data_voltf.close()


# load spikes
fp = f'/user/sebert/home/Documents/Experiments/Spatiotemporal_tuning_curves/{exp_name}/sorting/{exp_name}_results_curated.hdf5'
dat = h5py.File(fp, 'r')

keys = list(dat['spiketimes'].keys())
spiketimes = dat['spiketimes'] # spike times in frames



#load events
fp = f'/user/sebert/home/Documents/Experiments/Spatiotemporal_tuning_curves/{exp_name}/sync/event_list_{exp_name}_.csv'
events = pd.read_csv(fp)



# group by stimulus
g = events.groupby('protocol_name')
protocols = list(g.groups.keys())
print(f'protocols : {protocols}')



# group by intensity
# events_gratings = g.get_group("gratings")
# gi = events_gratings.groupby('nd')
# intensities = list(gi.groups.keys())print(f'intensitites : {intensities}')




# group by grating
events_gratings = g.get_group("gratings")
gi = events_gratings.groupby('nd')
intensities = list(gi.groups.keys())
print(f'intensitites : {intensities}')

experiments = list(gi['extra_description'].unique())
print(f'experimeints:')
print(f'{experiments}')
# sort spikes per stimulus 




# time in seconds
dt_sec = 0.02
stim_length_sec = events_gratings['event_duration'].values[0]/sampling_frequency
time_sec = np.arange(0,stim_length_sec,dt_sec)

bin_edges = np.append(time_sec, 2 * time_sec[-1] - time_sec[-2])


# time in frames
dt = dt_sec*sampling_frequency
stim_length = events_gratings['event_duration'].values[0]
time = np.arange(0,stim_length,dt)


sigma = 3

# create data dict
data = {}
data['stimuli']={}




print('getting spikes per trial')
for key_old in tqdm(keys):

    key  = temp_to_unit(key_old)
    data[key] = {}
    spktms = spiketimes[key_old][()].flatten()
    df = pd.DataFrame(spktms,  columns = ['Times'])

    for i in intensities:

        data['stimuli'][i]={}

        data[key][i] = {}
        evts_i= gi.get_group(i)
        ggr = evts_i.groupby("extra_description")
        grating_types = list(ggr.groups.keys())

        for grat in grating_types:
            
            data['stimuli'][i][grat] = {}
            data[key][i][grat] = {}

            evts = ggr.get_group(grat)
            start = evts['start_event'].values
            end = evts['end_event'].values

            idxs = np.array(evts.index)

            nb_frames= evts['n_frames']

            rasters = []
          
            stims = []

            

            for nb in range(nb_trials):
                if nb == 0:
                    continue

                sti = stim_full[start[nb]:end[nb]]

                times = df.query(f"Times>={start[nb]} & Times<={end[nb]}") - start[nb]

                rasters.append(np.asarray(times["Times"])/sampling_frequency)
                stims.append(sti)
             


            spike_times_all_trials = np.concatenate(rasters, axis= 0)
            count, bins = np.histogram(spike_times_all_trials, bins=bin_edges, density=False) 
            count_smooth = gaussian_filter(count,sigma)/10

            data[key][i][grat]['rasters'] = rasters

            data[key][i][grat]['count'] = count

            data['stimuli'][i][grat]['full_stim'] = stims
           




nb_trials = 9
keys = list(data.keys())[1:]



# create time for one stimulus
# frame_duration = 1/59.9 #s
# stim_duration = 60*frame_duration
stim_duration = 1.0
stim_length_ot = 2*stim_duration
time_ot = np.arange(0,stim_length_ot,dt_sec)
bin_edges_ot = np.append(time_ot, 2 * time_ot[-1] - time_ot[-2])

print('getting spikes per stimulus')
for key in tqdm(keys):

    for i in intensities:

        for grat in grating_types:

            rasters_sorted_stim = []
            rasters_sorted_stim_aligned = []
            #rasters_sorted_base = []

            counts_sorted_stim = []
            maxis_sorted_stim = []
            means_sorted_stim = []
            #counts_sorted_base = []

           
            #stims_sorted_base = []


            counts_sorted_stim_aligned = []
            maxis_sorted_stim_aligned= []
            means_sorted_stim_aligned = []

            #counts_sorted_base = []
            #stims_sorted_base = []
            
            rasters = data[key][i][grat]['rasters']
            sti = data['stimuli'][i][grat]['full_stim']

            if key == keys[0]:

                stims_sorted_stim_aligned= []
                start_sorted_stim_aligned = []

                stims_sorted_stim = []
                starts_sorted_stim = []

                

                for nbs in range(nb_stimuli):


                    start_stim  = 2*stim_duration*nbs
                    end_stim = start_stim + 1*stim_duration
                    start_base = end_stim
                    end_base = start_base + 1*stim_duration

                    start_stim_idx  = int(2*stim_duration*nbs*sampling_frequency)
                    end_stim_idx = start_stim_idx + 1*stim_duration*sampling_frequency
                    start_base_idx = end_stim_idx
                    end_base_idx = int(start_base_idx + 1*stim_duration*sampling_frequency)


                    stims_per_stim = []
                    stims_per_stim_aligned = []
                    start_stim_exs = []
                    
                    for nb in range(nb_trials):

                        sti_stim = sti[nb][int(start_stim_idx):int(end_base_idx)]
                        stims_per_stim.append(sti_stim)

                        # extract from photodiode signal
                        dets = np.diff(sti_stim)>250
                        xf = np.where(dets == True)
                        
                        if nbs == 0:
                            ref = xf[0][0]
                            start_stim_ex_idx = 0

                        else:
                            start_stim_ex_idx  = xf[0][0]-ref

                        end_base_ex_idx  = int(start_stim_ex_idx+40000)
                        
                        start_stim_ex  = start_stim_ex_idx/sampling_frequency
                        end_base_ex  = end_base_ex_idx/sampling_frequency

                        start_stim_exs.append(start_stim_ex)
                        
                        sti_stim_aligned = sti_stim[start_stim_ex_idx:end_base_ex_idx]
                        stims_per_stim_aligned.append(sti_stim_aligned)

                    start_sorted_stim_aligned.append(start_stim_exs)
                    stims_sorted_stim.append(stims_per_stim)
                    starts_sorted_stim.append(start_stim)
                    stims_sorted_stim_aligned.append(stims_per_stim_aligned)





            for nbs in range(nb_stimuli):

                rasters_pers_stim = []
                rasters_pers_stim_aligned = []
                #rasters_pers_base = []
                

                #stims_per_base = []
                start_stim = starts_sorted_stim[nbs]
                
                for nb in range(nb_trials):

                    # start_stim_trig  = data['triggers_sorted'][i][grat][nb][nbs][0]
                    # end_base_trig = data['triggers_sorted'][i][grat][nb][nbs][-1]

                    
                    times = pd.DataFrame(rasters[nb], columns = ['Times'])
                    #print(len(times), times["Times"][0],times["Times"][len(times["Times"])-1])
                    times_stim = times.query(f"Times>= {start_stim} & Times<= {end_base}") - start_stim


                    start_aligned = float(start_stim + start_sorted_stim_aligned[nbs][nb])
                    end_aligned = start_aligned + 2.0
                    #print(start_stim, start_aligned,end_aligned)

                    times_stim_aligned = times.query(f"Times>= {start_aligned} & Times<= {end_aligned}") - start_aligned
                    #times_stim_aligned = times_stim.query(f"Times>= {start_stim_aligned[nb]} & Times<= {start_stim_aligned[nb]+2.0}") - start_stim_aligned[nb]
                    #times_base = times.query(f"Times>= {start_base} & Times<= {end_base}") - start_base

                    #sti_base = sti[nb][int(start_stim):int(end_base)]

                    rasters_pers_stim.append(np.asarray(times_stim["Times"]))
                    rasters_pers_stim_aligned.append(np.asarray(times_stim_aligned["Times"]))
                    # if nb ==0 :
                    #     print(np.asarray(times_stim["Times"]))

                    #rasters_pers_base.append(np.asarray(times_base["Times"]))

                    #stims_per_base.append(sti_base)


                spike_times_all_trials_stim = np.concatenate(rasters_pers_stim, axis= 0)
                count_stim, bins = np.histogram(spike_times_all_trials_stim, bins=bin_edges_ot, density=False) 
                count_smooth_stim = gaussian_filter(count_stim,sigma)/10
                maxi = np.max(count_smooth_stim)
                mean = np.mean(count_smooth_stim)

                counts_sorted_stim.append(count_smooth_stim)
                maxis_sorted_stim.append(maxi)
                means_sorted_stim.append(mean)


                spike_times_all_trials_stim_aligned = np.concatenate(rasters_pers_stim_aligned, axis= 0)
                count_stim_aligned, bins = np.histogram(spike_times_all_trials_stim_aligned, bins=bin_edges_ot, density=False) 
                count_smooth_stim_aligned = gaussian_filter(count_stim_aligned,sigma)/nb_trials
                maxi_aligned= np.max(count_smooth_stim_aligned)
                mean_aligned = np.mean(count_smooth_stim_aligned)

                counts_sorted_stim_aligned.append(count_smooth_stim_aligned)
                maxis_sorted_stim.append(maxi_aligned)
                means_sorted_stim.append(mean_aligned)

                #spike_times_all_trials_base = np.concatenate(rasters_pers_base, axis= 0)
                #count_base, bins = np.histogram(spike_times_all_trials_base, bins=bin_edges_ot, density=False) 
                #count_smooth_base = gaussian_filter(count_base,sigma)/10

                #counts_sorted_base.append(count_smooth_base)


                rasters_sorted_stim.append(rasters_pers_stim)
                rasters_sorted_stim_aligned.append(rasters_pers_stim_aligned)
                #rasters_sorted_base.append(rasters_pers_base)

                # stims_per_stim = np.mean(stims_per_stim, axis = 0)
                # stims_per_base = np.mean(stims_per_base,axis = 0)
                if key ==  keys[0]:
                    stims_sorted_stim.append(stims_per_stim)
                    starts_sorted_stim.append(start_sorted_stim_aligned)
                    stims_sorted_stim_aligned.append(stims_per_stim_aligned)
                #stims_sorted_base.append(stims_per_base)


            data[key][i][grat]['rasters_sorted_stim'] = rasters_sorted_stim
            data[key][i][grat]['rasters_sorted_stim_aligned'] = rasters_sorted_stim_aligned
            #data[key][i][grat]['rasters_sorted_base'] = rasters_sorted_base

            data[key][i][grat]['counts_sorted_stim'] = counts_sorted_stim
            data[key][i][grat]['counts_sorted_stim_alinged'] = counts_sorted_stim_aligned
            #data[key][i][grat]['counts_sorted_base'] = counts_sorted_base

            data[key][i][grat]['max_sorted_stim'] = maxis_sorted_stim
            data[key][i][grat]['mean_sorted_stim'] = means_sorted_stim
            data[key][i][grat]['max_sorted_stim_aligned'] = maxis_sorted_stim_aligned
            data[key][i][grat]['mean_sorted_stim_alinged'] = means_sorted_stim_aligned


            #data[key][i][grat]['max_sorted_base'] = np.max(counts_sorted_base)

            if key == keys[0]:
                data['stimuli'][i][grat]['stimuli_sorted_stim'] = stims_sorted_stim
                data['stimuli'][i][grat]['starts_sorted_stim'] = starts_sorted_stim
                data['stimuli'][i][grat]['stimuli_sorted_stim_aligned'] = stims_sorted_stim_aligned
            #data[key][i][grat]['stimuli_sorted_base'] = stims_sorted_base





#save data
fpdata = f'/user/sebert/home/Documents/Experiments/Spatiotemporal_tuning_curves/Results/{exp_name}'

if not os.path.isdir(fpdata):
    os.mkdir(fpdata)

with open(f'{fpdata}/data.pkl', "wb") as handle:   #Pickling
    pickle.dump(data, handle,protocol=4 )
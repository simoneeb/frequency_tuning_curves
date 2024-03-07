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

#load data picke: 
fpdata = f'/user/sebert/home/Documents/Experiments/Spatiotemporal_tuning_curves/Results/{exp_name}/data.pkl'

with open(fpdata, "rb") as handle:   #Pickling
    data = pickle.load(handle)


keys = list(data.keys())[3:]

intensities = list(data['stimuli'].keys())
#grating_types = list(data['stimuli'][intensities[0]].keys())
grating_types = ['flickering', 'moving']
contrasts = ['cm10', 'cm05', 'cm02']

resp_type = 'counts_sorted_stim_alinged'
inte = 'nd4'
nb_stimuli = 25


fxs = [1,2,4,6,10]
fts = [2,4,5,6,8]

fxs = np.array([1,2,4,6,10])
fts = np.array([2,4,5,6,8])

vss = []

for i in range(len(fxs)):
    vs = fts/fxs[i]
    vss.append(vs)




# create dataframe 
# heatdf = pd.DataFrame(columns = ['fx', 'ft','v', 'max','mean','intensity', 'cm', 'grating_type', 'key','respose', 'polarity', 'stim_id'])


dict = {'fx':[],
        'ft':[],
        'v':[],
        'mean_stim':[],
        'mean_base':[],
        'intensity':[],
        'cm':[], 
        'grating_type':[],
        'key':[],
        'polarity':[],
        'stim_id':[]
        }
# get stm and base mean per stimulus 
for key in tqdm(keys):

    mean_tot_stim = 0
    mean_tot_base = 0
    std_tot_base = 0

    for cm in contrasts:

        for grating_type in grating_types:

            for nbs in range(nb_stimuli):

                grat = f'{grating_type}_{cm}'
                #row ={}

                # get response
                resp = data[key][inte][grat][resp_type][nbs]
 
                #print('1')
                # split stim and base
                idx = int(len(resp)/2)
                resp_stim = resp[:idx]
                resp_base = resp[idx:]

                #print('2')

                # mean and std 
                mean_stim = np.mean(resp_stim)
                mean_base = np.mean(resp_base)
                std_base = np.std(resp_base)

                #print('3')

                # append to total mean
                mean_tot_stim = mean_tot_stim + mean_stim
                mean_tot_base = mean_tot_base + mean_base
                std_tot_base = std_tot_base + std_base
                #print('4')


                # save to data dict
                data[key][inte][grat]['means_stim'] = mean_stim
                data[key][inte][grat]['means_base'] = mean_base

                #print('5')

                # create dataframe row 
                y = int(np.floor(nbs/5))
                # row['fx'] = float(fxs[y])
                # row['stim_id'] = nbs
                
                # x = int(nbs%5)
                # row['ft'] = float(fts[x])
                # row['v'] = float(fts[x]/fxs[y])

                # row['intensity'] = i
                # row['cm'] = cm
                # row['grating_type'] = grating_type
                # row['key'] = key

                # row['mean_stim'] = float(mean_stim)
                # row['mean_base'] = float(mean_base)

                # row['polarity'] = data[key]['type']
                dict['fx'].append(float(fxs[y]))
                dict['stim_id'].append(nbs)
                
                x = int(nbs%5)
                dict['ft'].append(float(fts[x]))
                dict['v'].append(float(fts[x]/fxs[y]))

                dict['intensity'].append(i)
                dict['cm'].append(cm)
                dict['grating_type'].append(grating_type)
                dict['key'].append(key)

                dict['mean_stim'].append(float(mean_stim))
                dict['mean_base'].append(float(mean_base))

                dict['polarity'].append(data[key]['type'])
                #print('6')


                # append
                # rowdf = pd.Series(row)
                # heatdf = pd.concat([heatdf, rowdf], ignore_index = True)
                # heatdf = heatdf.append(row, ignore_index = True)
                #print('7')

    heatdf = pd.DataFrame.from_dict(dict)
    mean_tot_stim = mean_tot_stim/(50*3)
    mean_tot_base = mean_tot_base/(50*3)
    std_tot_base = std_tot_base/(50*3)


    # evaluate if response or not for entire responses
    if mean_tot_stim >= mean_tot_base+2*std_tot_base :
        # add to data dict
        data[key]['response'] = 'yes'
        # add to df 
        heatdf.loc[heatdf['key'] == key, 'response']= 'yes'
        print(f'{key} yes')
    else:
        # add to data dict
        data[key]['response'] = 'no'
        # add to df 
        heatdf.loc[heatdf['key'] == key, 'response']= 'no'
        print(f'{key} no')

    #print('8')




# save data dict
with open(fpdata, "wb") as handle:   
    pickle.dump(data, handle,protocol=4 ) 



# save df
fpdataheat = f'/user/sebert/home/Documents/Experiments/Spatiotemporal_tuning_curves/Results/{exp_name}/dataframe.csv'
heatdf.to_csv(fpdataheat)







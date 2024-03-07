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
import os


experiment_name = 'MR-0609'


# load data 
#fpdata = f'/user/sebert/home/Documents/Experiments/Spatiotemporal_tuning_curves/Results/{experiment_name}/data.pkl'
fpdata = f'/Users/simone/Documents/Experiments/Spatiotemporal_tuning_curves/Results/{experiment_name}/data.pkl'

with open(fpdata, "rb") as handle:   #Pickling
    data = pickle.load(handle)

keys = list(data.keys())[1:-3]
#keys = ['Unit_0114', 'Unit_0125']



intensities = list(data['stimuli'].keys())
#grating_types = list(data['stimuli'][intensities[0]].keys())
grating_types = ['flickering', 'moving']
contrasts = ['cm10', 'cm05', 'cm02']



fxs = [1,2,4,6,10]
fts = [2,4,5,6,8]

fxs = np.array([1,2,4,6,10])
fts = np.array([2,4,5,6,8])


colors_flicker = ['teal','turquoise', 'cyan']
colors_high = ['darkmagenta','magenta','orchid']
#colors_close = ['darkgreen', 'seagreen', 'springgreen']
colors_close = ['darkred', 'red', 'tomato']
colors_moving = ['plum','magenta','darkmagenta']


i = 'nd4'


for key in keys:

    try:
        if data[key]['type'] == 'ON':
            POL = 'ON'
        if data[key]['type'] == 'OFF':
            POL = 'OFF'
        if data[key]['type'] == 'ON/OFF':
            POL = 'ONOFF'

        print(f' analyzing {key}')

    except:
        print(f'{key} not found')
        continue
    

    if data[key]['response'] == 'yes':


        for gi,grati in enumerate(grating_types):

            if grati == 'flickering':

                fig = plt.figure(figsize = (24,12))
                fig.subplots_adjust(hspace = 0.46)

                gs = fig.add_gridspec(3,5)

                for x,fx in enumerate(fxs): 

                    # mean
                    ax = fig.add_subplot(gs[0,x])
                    ax.set_xlabel('ft')
                    if x == 0:
                        ax.set_ylabel('mean response')
                    if x == 2:
                        ax.set_title(f'Mean response tuning across contrasts \n {fx} [cy/mm]')
                    else:
                        ax.set_title(f' {fx} [cy/mm]')

                    # highest
                    ax2 = fig.add_subplot(gs[1,x])
                    ax2.set_xlabel('ft')
                    if x == 0:
                        ax2.set_ylabel('amplitude')
                    if x == 2:
                        ax2.set_title(f'higest peak in power spectrum\n {fx} [cy/mm]')
                    else:
                        ax.set_title(f' {fx} [cy/mm]')

                    # closest
                    ax3 = fig.add_subplot(gs[2,x])
                    ax3.set_xlabel('ft')
                    if x == 0:
                        ax3.set_ylabel('amplitude')
                    if x == 2:
                        ax3.set_title(f'power spectrum peak closest to stimulus frequency\n {fx} [cy/mm]')
                    else:
                        ax.set_title(f' {fx} [cy/mm]')
    
            
                    for ic,cm in enumerate(contrasts):

                        grat = f'{grati}_{cm}'

                        # mean
                        mean_amp = data[key][i][grat][f'{fx}']['mean_per_ft']['vals']
                        [xrange,mean_amp_curve] = data[key][i][grat][f'{fx}']['mean_per_ft']['fits']

                        ax.scatter(fts,mean_amp, color = colors_flicker[ic])
                        ax.plot(xrange, mean_amp_curve, linestyle = ':', color = colors_flicker[ic], label = f'{cm}')

                        # high
                        high_amp = data[key][i][grat][f'{fx}']['highest_power_peak_per_ft']['vals']
                        [xrange,high_amp_curve] = data[key][i][grat][f'{fx}']['highest_power_peak_per_ft']['fits']

                        ax2.scatter(fts,high_amp, color = colors_high[ic])
                        ax2.plot(xrange, high_amp_curve, linestyle = ':', color = colors_high[ic], label = f'{cm}')

                        # close
                        close_amp = data[key][i][grat][f'{fx}']['nearest_power_peak_per_ft']['vals']
                        [xrange,close_amp_curve] = data[key][i][grat][f'{fx}']['nearest_power_peak_per_ft']['fits']

                        ax3.scatter(fts,close_amp, color = colors_close[ic])
                        ax3.plot(xrange, close_amp_curve, linestyle = ':', color = colors_close[ic], label = f'{cm}')

                        ax.legend()
                        ax2.legend()
                        ax3.legend()

                fig.suptitle('Tuning curves for different contrast levels of flickering gratings')



                if data[key]['type'] == 'ON':
                    POL = 'ON'
                if data[key]['type'] == 'OFF':
                    POL = 'OFF'
                if data[key]['type'] == 'ON/OFF':
                    POL = 'ONOFF'

                fpplots = f'/Users/simone/Documents/Experiments/Spatiotemporal_tuning_curves/Results/{experiment_name}/plots/tuning_curves/ft_compare_contrast/{POL}/{grati}'

                if not os.path.isdir(fpplots):
                    os.mkdir(fpplots)


                fig.savefig(f'{fpplots}/{key}.png')
    



            if grati == 'moving':

                fig = plt.figure(figsize = (24,12))
                fig.subplots_adjust(hspace = 0.46)

                gs = fig.add_gridspec(3,5)

                for x,fx in enumerate(fxs): 

                    # mean
                    ax = fig.add_subplot(gs[0,x])
                    ax.set_xlabel('ft')
                    if x == 0:
                        ax.set_ylabel('mean response')
                    if x == 2:
                        ax.set_title(f'Mean response tuning across contrasts \n {fx} [cy/mm]')
                    else:
                        ax.set_title(f' {fx} [cy/mm]')

                    # highest
                    ax2 = fig.add_subplot(gs[1,x])
                    ax2.set_xlabel('ft')
                    if x == 0:
                        ax2.set_ylabel('amplitude')
                    if x == 2:
                        ax2.set_title(f'higest peak in power spectrum\n {fx} [cy/mm]')
                    else:
                        ax.set_title(f' {fx} [cy/mm]')

                    # closest
                    ax3 = fig.add_subplot(gs[2,x])
                    ax3.set_xlabel('ft')
                    if x == 0:
                        ax3.set_ylabel('amplitude')
                    if x == 2:
                        ax3.set_title(f'power spectrum peak closest to stimulus frequency\n {fx} [cy/mm]')
                    else:
                        ax.set_title(f' {fx} [cy/mm]')
    
            
                    for ic,cm in enumerate(contrasts):

                        grat = f'{grati}_{cm}'

                        # mean
                        mean_amp = data[key][i][grat][f'{fx}']['mean_per_ft']['vals']
                        [xrange,mean_amp_curve] = data[key][i][grat][f'{fx}']['mean_per_ft']['fits']

                        ax.scatter(fts,mean_amp, color = colors_moving[ic])
                        ax.plot(xrange, mean_amp_curve, linestyle = ':', color = colors_moving[ic], label = f'{cm}')

                        # high
                        high_amp = data[key][i][grat][f'{fx}']['highest_power_peak_per_ft']['vals']
                        [xrange,high_amp_curve] = data[key][i][grat][f'{fx}']['highest_power_peak_per_ft']['fits']

                        ax2.scatter(fts,high_amp, color = colors_high[ic])
                        ax2.plot(xrange, high_amp_curve, linestyle = ':', color = colors_high[ic], label = f'{cm}')

                        # close
                        close_amp = data[key][i][grat][f'{fx}']['nearest_power_peak_per_ft']['vals']
                        [xrange,close_amp_curve] = data[key][i][grat][f'{fx}']['nearest_power_peak_per_ft']['fits']

                        ax3.scatter(fts,close_amp, color = colors_close[ic])
                        ax3.plot(xrange, close_amp_curve, linestyle = ':', color = colors_close[ic], label = f'{cm}')

                        ax.legend()
                        ax2.legend()
                        ax3.legend()

                fig.suptitle('Tuning curves for different contrast levels of moving gratings')

                if data[key]['type'] == 'ON':
                    POL = 'ON'
                if data[key]['type'] == 'OFF':
                    POL = 'OFF'
                if data[key]['type'] == 'ON/OFF':
                    POL = 'ONOFF'

                fpplots = f'/Users/simone/Documents/Experiments/Spatiotemporal_tuning_curves/Results/{experiment_name}/plots/tuning_curves/ft_compare_contrast/{POL}/{grati}'

                if not os.path.isdir(fpplots):
                    os.mkdir(fpplots)


                fig.savefig(f'{fpplots}/{key}.png')


















            plt.close()

        else:
            continue
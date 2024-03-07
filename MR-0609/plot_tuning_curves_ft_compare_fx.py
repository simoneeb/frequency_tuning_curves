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
#keys = ['Unit_0038', 'Unit_0125']



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


        fig = plt.figure(figsize = (24,12))
        fig.subplots_adjust(hspace = 0.46)

        gs = fig.add_gridspec(3,2)

        for gi,grati in enumerate(grating_types):

            if grati == 'flickering':

                # mean
                ax = fig.add_subplot(gs[0,0])
                ax.set_xlabel('ft')
                ax.set_ylabel('mean response')
                ax.set_title(f'Tuning curves for different spatial frequencies flickering gratings \n Mean response ')
                

                # highest
                ax2 = fig.add_subplot(gs[1,0])
                ax2.set_xlabel('ft')
                ax2.set_ylabel('amplitude')
                ax2.set_title(f'higest peak in power spectrum')
            

                # closest
                ax3 = fig.add_subplot(gs[2,0])
                ax3.set_xlabel('ft')
                ax3.set_ylabel('amplitude')
                ax3.set_title(f'power spectrum peak closest to stimulus frequency')


                for x,fx in enumerate(fxs): 

                    grat = f'{grati}_cm10'

                    # mean
                    mean_amp = data[key][i][grat][f'{fx}']['mean_per_ft']['vals']
                    [xrange,mean_amp_curve] = data[key][i][grat][f'{fx}']['mean_per_ft']['fits']

                    l = ax.scatter(fts,mean_amp)
                    ax.plot(xrange, mean_amp_curve, linestyle = ':', label = f'{fx} [cy/mm]', color = l.get_facecolor())

                    # high
                    high_amp = data[key][i][grat][f'{fx}']['highest_power_peak_per_ft']['vals']
                    [xrange,high_amp_curve] = data[key][i][grat][f'{fx}']['highest_power_peak_per_ft']['fits']

                    l2 = ax2.scatter(fts,high_amp)
                    ax2.plot(xrange, high_amp_curve, linestyle = ':', label = f'{fx} [cy/mm]', color = l2.get_facecolor())

                    # close
                    close_amp = data[key][i][grat][f'{fx}']['nearest_power_peak_per_ft']['vals']
                    [xrange,close_amp_curve] = data[key][i][grat][f'{fx}']['nearest_power_peak_per_ft']['fits']

                    l3 = ax3.scatter(fts,close_amp)
                    ax3.plot(xrange, close_amp_curve, linestyle = ':', label = f'{fx} [cy/mm]', color = l3.get_facecolor())

                    ax.legend()
                    ax2.legend()
                    ax3.legend()



            if grati == 'moving':

               # mean
                ax = fig.add_subplot(gs[0,1])
                ax.set_xlabel('ft')
                ax.set_ylabel('mean response')
                ax.set_title(f'Tuning curves for different spatial frequencies moving gratings \n Mean response ')
                

                # highest
                ax2 = fig.add_subplot(gs[1,1])
                ax2.set_xlabel('ft')
                ax2.set_ylabel('amplitude')
                ax2.set_title(f'higest peak in power spectrum')
            

                # closest
                ax3 = fig.add_subplot(gs[2,1])
                ax3.set_xlabel('ft')
                ax3.set_ylabel('amplitude')
                ax3.set_title(f'power spectrum peak closest to stimulus frequency')


                for x,fx in enumerate(fxs): 

                    grat = f'{grati}_cm10'

                    # mean
                    mean_amp = data[key][i][grat][f'{fx}']['mean_per_ft']['vals']
                    [xrange,mean_amp_curve] = data[key][i][grat][f'{fx}']['mean_per_ft']['fits']

                    l = ax.scatter(fts,mean_amp)
                    ax.plot(xrange, mean_amp_curve, linestyle = ':', label = f'{fx} [cy/mm]', color = l.get_facecolor())

                    # high
                    high_amp = data[key][i][grat][f'{fx}']['highest_power_peak_per_ft']['vals']
                    [xrange,high_amp_curve] = data[key][i][grat][f'{fx}']['highest_power_peak_per_ft']['fits']

                    l2 = ax2.scatter(fts,high_amp)
                    ax2.plot(xrange, high_amp_curve, linestyle = ':', label = f'{fx} [cy/mm]', color = l2.get_facecolor())

                    # close
                    close_amp = data[key][i][grat][f'{fx}']['nearest_power_peak_per_ft']['vals']
                    [xrange,close_amp_curve] = data[key][i][grat][f'{fx}']['nearest_power_peak_per_ft']['fits']

                    l3 = ax3.scatter(fts,close_amp)
                    ax3.plot(xrange, close_amp_curve, linestyle = ':', label = f'{fx} [cy/mm]', color = l3.get_facecolor())

                    ax.legend()
                    ax2.legend()
                    ax3.legend()


                if data[key]['type'] == 'ON':
                    POL = 'ON'
                if data[key]['type'] == 'OFF':
                    POL = 'OFF'
                if data[key]['type'] == 'ON/OFF':
                    POL = 'ONOFF'

                fpplots = f'/Users/simone/Documents/Experiments/Spatiotemporal_tuning_curves/Results/{experiment_name}/plots/tuning_curves/ft_compare_fx/{POL}'

                if not os.path.isdir(fpplots):
                    os.mkdir(fpplots)


                fig.savefig(f'{fpplots}/{key}.png')


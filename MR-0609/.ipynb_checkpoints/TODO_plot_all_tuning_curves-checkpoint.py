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


    # resp_type = 'rasters_sorted_stim'
    colors_flicker = ['teal','turquoise', 'cyan']
    grat = grating_types[0]

    
    resp_type = 'counts_sorted_stim_alinged'
    #resp_type = 'counts_sorted_stim'

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
            plt.axvline(time_ot[idx])
            
            mean_stim = np.mean(resp[:idx])
            mean_base = np.mean(resp[idx:])
            mean_ratio = mean_stim/mean_base

            data[key][i][grat]['means_stim'] = mean_stim
            data[key][i][grat]['means_base'] = mean_base
            data[key][i][grat]['means_ratio'] = mean_ratio

            row['max'] = float(resp.max())
            row['mean_stim'] = float(mean_stim)
            row['mean_base'] = float(mean_base)
            row['mean_ratio'] = float(mean_ratio)

            heatdf_flicker = heatdf_flicker.append(row, ignore_index=True)
            


    # ========================================================================================================================================================================
    # MOVING ALL STIMS
    # ========================================================================================================================================================================


    # resp_type = 'rasters_sorted_stim'
    colors_moving = ['plum','magenta','darkmagenta']

    grat = grating_types[1]

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
            mean_ratio = mean_stim/mean_base

            data[key][i][grat]['means_stim'] = mean_stim
            data[key][i][grat]['means_base'] = mean_base
            data[key][i][grat]['means_ratio'] = mean_ratio

            row['max'] = float(resp.max())
            row['mean_stim'] = float(mean_stim)
            row['mean_base'] = float(mean_base)
            row['mean_ratio'] = float(mean_ratio)

            heatdf_moving = heatdf_moving.append(row, ignore_index=True)
           

    # ========================================================================================================================================================================
    #  TUNIG CURVE : FLICKERING BY INTENSITIES
    # ========================================================================================================================================================================

    fpplots = '/user/sebert/home/Documents/Experiments/Spatiotemporal_tuning_curves/Simone/MR-0605/results/tuning_curves_mean_stim/tuning_curves_per_density/flickering_grating'


    grat = grating_types[0]
    target = 'mean_stim'

    grouped = heatdf_flicker.groupby('intensity')

    cmap_temporal = plt.get_cmap('Set1', len(fts))

    cmap_spatial = plt.get_cmap('Set2', len(fxs))

    fig, ax = plt.subplots(3,4, sharex = 'col', figsize = (20,20))

    fig.subplots_adjust(top=0.88,
    bottom=0.11,
    left=0.11,
    right=0.9,
    hspace=0.2,
    wspace=0.7)

    for ix,i in enumerate(intensities):

        heatdf_i = grouped.get_group(i)
        heatplot = heatdf_i.pivot_table(index='fx', columns='ft', values=target)

        sns.heatmap(heatplot, ax = ax[ix,0],cmap = 'YlGnBu', cbar_kws={'label' :f'{target} firing rate'})
        ax[ix,0].set_title(f'{i}', loc = 'left', color = colors_flicker[ix])


        for fxi,fx in enumerate(fxs):
            max_ft = heatdf_i[heatdf_i['fx' ] == fx][target].values
            ft_ft = heatdf_i[heatdf_i['fx' ] == fx]['ft'].values
            v_ft = heatdf_i[heatdf_i['fx' ] == fx]['v'].values

            # fit skewed gaussian
            ftwide = np.arange(ft_ft.min(),ft_ft.max(),0.1)
            # params = model.make_params(amplitude= max_ft.max(), center = ft_ft[max_ft.argmax()], sigma=1, gamma=0)
            # result = model.fit(max_ft, params, x=ft_ft)
            # predicted = model.eval(result.params, x=ftwide)

            ax[ix,2].scatter(ft_ft, max_ft, label = f'fx = {fx}', color = cmap_spatial(fxi))
            #ax[ix,2].plot(ftwide, predicted, linestyle = ':', color = cmap_spatial(fxi)) 

            ax[ix,2].plot(ft_ft, max_ft, linestyle = '--', color = cmap_spatial(fxi))
            ax[-1,2].set_xlabel('ft')

            if ix == 0:
                ax[ix,2].legend(bbox_to_anchor=(1., 1.))
            ax[0,2].set_title('temporal tuning curves')
            #ax[ix,2].set_ylabel(f'{target} firing rate')

            # fit skewed gaussian
            vwide = np.arange(v_ft.min(),v_ft.max(),0.1)
            # params = model.make_params(amplitude= max_ft.max(), center = v_ft[max_ft.argmax()], sigma=1, gamma=0)
            # result = model.fit(max_ft, params, x=v_ft)
            # predicted = model.eval(result.params, x=vwide)

            ax[ix,1].scatter(v_ft, max_ft, label = f'fx = {fx}', color = cmap_spatial(fxi))
            #ax[ix,1].plot(vwide, predicted, linestyle = ':', color = cmap_spatial(fxi)) 

            ax[ix,1].plot(v_ft, max_ft, linestyle = '--', color = cmap_spatial(fxi))
            ax[-1,1].set_xlabel('v')

            if ix == 0:
                ax[ix,1].legend(bbox_to_anchor=(1., 1.))
            ax[0,1].set_title('speed tuning curves')
            ax[ix,1].set_ylabel(f'{target} firing rate')


        for fti,ft in enumerate(fts):
            max_fx = heatdf_i[heatdf_i['ft' ] == ft][target]
            fx_fx = heatdf_i[heatdf_i['ft' ] == ft]['fx']
            ax[ix,3].scatter(fx_fx, max_fx, label = f'ft = {ft}', color = cmap_temporal(fti))
            ax[ix,3].plot(fx_fx, max_fx, linestyle = '--', color = cmap_temporal(fti))
            
            if ix ==0:
                ax[ix,3].legend(bbox_to_anchor=(1., 1.))
            ax[0,3].set_title('spatial tuning curves')
            ax[-1,3].set_xlabel('fx')
            #ax[ix,3].set_ylabel(f'{target} firing rate')

    fig.suptitle(f'{target} firing rate tuning curves to flickering grating, sorted by stimulus density')


    fig.savefig(f'{fpplots}/{key}')
    plt.close()


    # ========================================================================================================================================================================
    #  TUNIG CURVE : FLICKERING BY FREQIENCIES
    # ========================================================================================================================================================================

    fpplots = '/user/sebert/home/Documents/Experiments/Spatiotemporal_tuning_curves/Simone/MR-0605/results/tuning_curves_mean_stim//tuning_curves_per_frequency/flickering_grating'


    fig = plt.figure(figsize = (16,20))
    gs = fig.add_gridspec(15,4)
    fig.subplots_adjust(top=0.92,
                        bottom=0.045,
                        left=0.11,
                        right=0.9,
                        hspace=4.0,
                        wspace=0.5)

    axs1 = []
    axs0 = []
    axs2 = []

    for ix,i in enumerate(intensities):
        heatdf_i = grouped.get_group(i)
        heatplot = heatdf_i.pivot_table(index='fx', columns='ft', values=target)

        ax = fig.add_subplot(gs[ix*5:ix*5+5,0])
        sns.heatmap(heatplot, ax = ax,cmap = 'YlGnBu', cbar_kws={'label' :f'{target} firing rate'})

        ax.set_title(f'{i}', loc = 'left', color = colors_flicker[ix])

        for fxx,fx in enumerate(fxs):
            max_ft = heatdf_i[heatdf_i['fx' ] == fx][target]
            ft_ft = heatdf_i[heatdf_i['fx' ] == fx]['ft']
            v_ft = heatdf_i[heatdf_i['fx' ] == fx]['v']

            if ix == 0 :
                axs1.append(fig.add_subplot(gs[fxx*3:fxx*3+3,2]))
                axs0.append(fig.add_subplot(gs[fxx*3:fxx*3+3,1]))


            if fxx == 0:
                axs1[fxx].scatter(ft_ft, max_ft, label = f'{i}', color = colors_flicker[ix])


            else:
                axs1[fxx].scatter(ft_ft, max_ft, color = colors_flicker[ix])

            axs1[fxx].plot(ft_ft, max_ft, linestyle = '--', color = colors_flicker[ix])
            axs1[fxx].set_title(f'fx = {fx}', loc = 'left')

            axs1[0].set_title('temporal tuning curves')
            #axs1[2].set_ylabel(f'{target} firing rate')

            axs0[fxx].scatter(v_ft, max_ft, color = colors_flicker[ix])
            axs0[fxx].plot(v_ft, max_ft, linestyle = '--', color = colors_flicker[ix])
            axs0[fxx].set_title(f'fx = {fx}', loc = 'left')

            axs0[0].set_title('speed tuning curves')

        for ftx,ft in enumerate(fts):

            if ix == 0:
                axs2.append(fig.add_subplot(gs[ftx*3 :ftx*3+3,3]))

            max_fx = heatdf_i[heatdf_i['ft' ] == ft][target]
            fx_fx = heatdf_i[heatdf_i['ft' ] == ft]['fx']

            axs2[ftx].scatter(fx_fx, max_fx, color = colors_flicker[ix])
            axs2[ftx].plot(fx_fx, max_fx, linestyle = '--', color = colors_flicker[ix])
            axs2[0].set_title('spatial tuning curves')
            #axs2[ftx].set_ylabel(f'{target} firing rate')
            axs2[ftx].set_title(f'ft = {ft}', loc = 'left')

        axs1[-1].set_xlabel('ft')
        axs0[-1].set_xlabel('v')
        axs0[2].set_ylabel(f'{target} firing rate')
        axs2[-1].set_xlabel('fx')


    fig.suptitle(f'{target} firing rate tuning curves to flickering grating, sorted by temporal or spatial frequency')
    fig.legend()

    fig.savefig(f'{fpplots}/{key}')
    plt.close()



    # ========================================================================================================================================================================
    #  TUNIG CURVE : MOVING BY INTENSITIES
    # ========================================================================================================================================================================

    fpplots = '/user/sebert/home/Documents/Experiments/Spatiotemporal_tuning_curves/Simone/MR-0605/results/tuning_curves_mean_stim/tuning_curves_per_density/moving_grating'

    grat = grating_types[1]

    grouped = heatdf_moving.groupby('intensity')

    fig, ax = plt.subplots(3,4, sharex = 'col', figsize = (20,20))


    fig.subplots_adjust(top=0.88,
    bottom=0.11,
    left=0.11,
    right=0.9,
    hspace=0.2,
    wspace=0.7)


    for ix,i in enumerate(intensities):

        heatdf_i = grouped.get_group(i)
        heatplot = heatdf_i.pivot_table(index='fx', columns='ft', values=target)

        sns.heatmap(heatplot, ax = ax[ix,0],cmap = 'RdPu', cbar_kws={'label' :f'{target} firing rate'})
        ax[ix,0].set_title(f'{i}', loc = 'left', color = colors_moving[ix])

        for fxi,fx in enumerate(fxs):
            max_ft = heatdf_i[heatdf_i['fx' ] == fx][target]
            ft_ft = heatdf_i[heatdf_i['fx' ] == fx]['ft']
            v_ft = heatdf_i[heatdf_i['fx' ] == fx]['v']

            ax[ix,2].scatter(ft_ft, max_ft, label = f'fx = {fx}', color = cmap_spatial(fxi))
            ax[ix,2].plot(ft_ft, max_ft, linestyle = '--', color = cmap_spatial(fxi))
            ax[-1,2].set_xlabel('ft')

            if ix ==0 :
                ax[ix,2].legend(bbox_to_anchor=(1., 1.))
            ax[0,2].set_title('temporal tuning curves')
            ax[ix,2].set_ylabel(f'{target} firing rate')


            
            ax[ix,1].scatter(v_ft, max_ft, label = f'fx = {fx}', color = cmap_spatial(fxi))
            ax[ix,1].plot(v_ft, max_ft, linestyle = '--', color = cmap_spatial(fxi))
            ax[-1,1].set_xlabel('v')

            if ix == 0 :
                ax[ix,1].legend(bbox_to_anchor=(1., 1.))
            ax[0,1].set_title('speed tuning curves')
            ax[ix,1].set_ylabel(f'{target} firing rate')


        for fti,ft in enumerate(fts):
            max_fx = heatdf_i[heatdf_i['ft' ] == ft][target]
            fx_fx = heatdf_i[heatdf_i['ft' ] == ft]['fx']
            ax[ix,3].scatter(fx_fx, max_fx, label = f'ft = {ft}', color = cmap_temporal(fti))
            ax[ix,3].plot(fx_fx, max_fx, linestyle = '--', color = cmap_temporal(fti))
            if ix == 0:
                ax[ix,3].legend(bbox_to_anchor=(1., 1.))
            ax[0,3].set_title('spatial tuning curves')
            ax[-1,3].set_xlabel('fx')
            ax[ix,3].set_ylabel(f'{target} firing rate')


    fig.suptitle(f'{target} firing rate tuning curves to moving grating, sorted by light density')
    fig.savefig(f'{fpplots}/{key}')
    plt.close()


    # ========================================================================================================================================================================
    #  TNUNG CURVE : MOVING BY FREQUENCIES
    # ========================================================================================================================================================================


    fpplots = '/user/sebert/home/Documents/Experiments/Spatiotemporal_tuning_curves/Simone/MR-0605/results/tuning_curves_mean_stim/tuning_curves_per_frequency/moving_grating'


    fig = plt.figure(figsize = (16,20))
    gs = fig.add_gridspec(15,4)
    fig.subplots_adjust(top=0.92,
                        bottom=0.045,
                        left=0.11,
                        right=0.9,
                        hspace=1.5,
                        wspace=0.2)

    axs1 = []
    axs0 = []
    axs2 = []

    for ix,i in enumerate(intensities):
        heatdf_i = grouped.get_group(i)
        heatplot = heatdf_i.pivot_table(index='fx', columns='ft', values=target)

        ax = fig.add_subplot(gs[ix*5:ix*5+5,0])
        sns.heatmap(heatplot, ax = ax,cmap = 'RdPu', cbar_kws={'label' :f'{target} firing rate'})

        ax.set_title(f'{i}', loc = 'left', color = colors_moving[ix])

        for fxx,fx in enumerate(fxs):
            max_ft = heatdf_i[heatdf_i['fx' ] == fx][target]
            ft_ft = heatdf_i[heatdf_i['fx' ] == fx]['ft']
            v_ft = heatdf_i[heatdf_i['fx' ] == fx]['v']

            if ix == 0 :
                axs1.append(fig.add_subplot(gs[fxx*3:fxx*3+3,2]))
                axs0.append(fig.add_subplot(gs[fxx*3:fxx*3+3,1]))


            if fxx == 0:
                axs1[fxx].scatter(ft_ft, max_ft, label = f'{i}', color = colors_moving[ix])


            else:
                axs1[fxx].scatter(ft_ft, max_ft, color = colors_moving[ix])

            axs1[fxx].plot(ft_ft, max_ft, linestyle = '--', color = colors_moving[ix])
            axs1[fxx].set_title(f'fx = {fx}', loc = 'left')

            axs1[0].set_title('temporal tuning curves')
            #axs1[2].set_ylabel(f'{target} firing rate')

            axs0[fxx].scatter(v_ft, max_ft, color = colors_moving[ix])
            axs0[fxx].plot(v_ft, max_ft, linestyle = '--', color = colors_moving[ix])
            axs0[fxx].set_title(f'fx = {fx}', loc = 'left')

            axs0[0].set_title('speed tuning curves')

        for ftx,ft in enumerate(fts):

            if ix == 0:
                axs2.append(fig.add_subplot(gs[ftx*3 :ftx*3+3,3]))

            max_fx = heatdf_i[heatdf_i['ft' ] == ft][target]
            fx_fx = heatdf_i[heatdf_i['ft' ] == ft]['fx']

            axs2[ftx].scatter(fx_fx, max_fx, color = colors_moving[ix])
            axs2[ftx].plot(fx_fx, max_fx, linestyle = '--', color = colors_moving[ix])
            axs2[0].set_title('spatial tuning curves')
            #axs2[ftx].set_ylabel(f'{target} firing rate')
            axs2[ftx].set_title(f'ft = {ft}', loc = 'left')


        axs1[-1].set_xlabel('ft')
        axs0[-1].set_xlabel('v')
        axs0[2].set_ylabel(f'{target} firing rate')
        axs2[-1].set_xlabel('fx')



    fig.legend()
    fig.suptitle(f'{target} firing rate tuning curves to moving grating, sorted by temporal or spatial frequency')

    fig.savefig(f'{fpplots}/{key}')
    plt.close()

#import h5py
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import imageio as iio
import os
from PIL import Image

import math
import scipy.signal as signal


cm = 0.2
img_mean = 128
stim_name = f'moving_gratings_cm_{cm}'
fp = f'/user/sebert/home/Documents/Experiments/Spatiotemporal_tuning_curves/stimuli/{stim_name}'


if not os.path.exists(fp):
    os.makedirs(fp)

# make images with spatial gratings with different spatial frequencies 
pix = 400
pix_mm = 200
pxs = np.arange(0,pix,1)
pix_width = 0.005 # mm


sp_frqs = np.array([1,2,4,6,10])
sp_frqs_pix = sp_frqs*pix_width

# define temporal flicker 
frqs = [2,4,5,6,8] # Hz
framerate = 60 #Hz
dt = 1/framerate
duration = 1
time  = np.arange(0,duration,dt)


#plt.show() 
#plt.savefig(f'{fp}/fxs.png')
#plt.close()



#save images to this directiry 
fpf = f'{fp}/frames'


if not os.path.exists(fpf):
    os.makedirs(fpf)


def create_grating(sf, phase = 0, ori = 90, wave = 'sin', imsize = 400, mm_per_pix = 0.005):

    """
    :param sf: spatial frequency (in pixels)
    :param ori: wave orientation (in degrees, [0-360])
    :param phase: wave phase (in degrees, [0-360])
    :param wave: type of wave ('sqr' or 'sin')
    :param imsize: image size (integer)
    :return: numpy array of shape (imsize, imsize)
    """
    # # Get x and y coordinates
    x, y = np.meshgrid(np.arange(imsize), np.arange(imsize))

    # # Get the appropriate gradient
    # gradient = np.sin(ori * math.pi / 180) * x - np.cos(ori * math.pi / 180) * y
    # Plug gradient into wave function
    if wave == 'sin':
        grating = np.sin((2 * math.pi *sf * x)+ (phase * math.pi) / 180)
    elif wave == 'sqr':
        grating = signal.square((2 * math.pi * x) / sf + (phase * math.pi) / 180)
    else:
        raise NotImplementedError

    return grating



# save stimulus informations


event_list = pd.DataFrame(columns = ['start_event',
                                'end_event',
                                'start_next_event',
                                'event_duration',
                                'start_event_sec',
                                'end_start_sec',
                                'start_next_event_sec',
                                'event_duration_sec',
                                'protocol_name',
                                'extra_description',
                                'repetition_name'])




imgi = 0
repetitions = 1

#fig,ax = plt.subplots(2,len(sp_frqs), sharex = True, figsize = (16,8))


for REP in range(repetitions): 

    fig,ax = plt.subplots(5,5, figsize = (16,16))
    for x,fx in enumerate(sp_frqs_pix):

        ax[0,x].set_title(f'fx = {sp_frqs_pix[x]}')

        # plt.imshow(grating)
        # plt.show()
        #print(f'fx : {fx}')

        for ti,ft in enumerate(frqs):



            # calculate speed 
            v = ft/sp_frqs[x]
            v = v/pix_width

            ax[ti,0].set_ylabel(f'ft = {ft} Hz \n v = {v*pix_width} mm/s')


            # print(f'fx pix {fx}')
            # print(f'fx mm {sp_frqs[x]}')
            # print(f'ft {ft}')
            # print(f'v {v}')
            # fp =f'/user/sebert/home/Documents/Experiments/Spatiotemporal_tuning_curves/stimuli/flickering_gratings/fx_{fx}/ft_{ft}'


            frames = []
            #print(f'ft : {ft}')

            event_info = {}
            event_info['start_event'] = imgi
            event_info['start_next_event'] = imgi + 120
            event_info['event_duration'] = 60
            event_info['protocol_name'] = stim_name
            event_info['extra_description'] = f'fx_{sp_frqs[x]}_ft_{ft}_v_{v*pix_width}'
            event_info['repetition_name'] = REP



            for i,t in enumerate(time):

                # new_frame = np.zeros((pix,pix))
    
                # for row in range(pix):
                #     new_frame[row,:] = ((np.sin((2*np.pi*fx)/pix_mm* pxs - np.round((v*t)/pix_width,0)) +1 ) /2) * 255

                #ax[ti,x].scatter(t,new_frame[200,200])
                pf = np.round(v*t,0)
                new_frame = create_grating(sf = fx, phase = pf)
                new_frame = (new_frame*img_mean*cm) + img_mean



                L_max = new_frame.max()
                L_min = new_frame.min()

                Cm = (L_max - L_min)/(L_max + L_min)
                print(f'Michaelson Contrast : {Cm}')
                #new_frame = ((new_frame+1)/2)*255
#                 if i%10 == 0 :
#                     ax[ti,x].plot(pxs, ((np.sin((2*np.pi*fx)/pix_mm* pxs - np.round((v*t)/pix_width,0)) +1 ) /2) * 255)
                ax[ti,x].scatter(t,new_frame[200,200], color ='k') #TODO color according to image intensity
# x = 1
                colorchannels = np.zeros((pix,pix,3))
                colorchannels[:,:,1] = new_frame
                colorchannels[:,:,2] = new_frame
                frames.append(new_frame)

                img = Image.fromarray(colorchannels.astype(np.uint8))

                img.save(f'{fpf}/img{imgi:05}.png')
                imgi = imgi +1
            
            event_info['end_event'] = imgi


            # add black intervals of 1 s between stimuli
            colorchannels = np.zeros((pix,pix,3))
            colorchannels[::,::,1] = 0
            colorchannels[::,::,2] = 0

            for i,t in enumerate(time):

                img = Image.fromarray(colorchannels.astype(np.uint8))
                img.save(f'{fpf}/img{imgi:05}.png')

                imgi = imgi+1

            event_list = event_list.append(event_info, ignore_index = True)


        ax[-1,0].set_xlabel(f'time [s]')



event_list.to_csv(f'{fp}/event_list_{stim_name}.csv')
x = 0
# # save new stimulis


fig.savefig(f'{fp}/wave_one_pixel.png')




os.system(f"ffmpeg -r {framerate} -i {fp}/frames/img%05d.png -vcodec mpeg4 -y {fp}/video_{stim_name}.mp4")

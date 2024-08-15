#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 04 15:56:17 2021

@author: burakgur
"""

#%% This code performs online RF mapping and generation of centered stimuli to be used with PyStim  

# %% Importing packages
import os
import numpy as np
import matplotlib.pyplot as plt
# import h5py
import cPickle
import seaborn as sns
import pandas as pd
from warnings import warn

os.chdir('/Volumes/Backup Plus/Post-Doc/_SiliesLab/Manuscripts/2023_Lum_Gain/Data_code/code/python2p7/pre-processing')

from batch_analysis_functions import copyStimXmlFilesPyStim
from core_functions import motionCorrection
import analysis_core as aCore
import PyROI
plt.switch_backend('Qt5Agg') # Working with Visual Studio code, interactive plotting for ROI selection
plt.style.use('default')
#%% Set directories
# Main directory
initial_dir = '/Volumes/SILIESLAB/BurakG/PhD_Archive/Data/2p_data/Ultima/2p_onlineRF'

raw_data_dir = os.path.join(initial_dir,'raw_to_map') # Directory where fly folders containing the raw T series are located 
aligned_data_dir = os.path.join(initial_dir,'aligned_to_map') # Directory for saving the aligned data
save_output_dir = os.path.join(initial_dir,'selected_ROI_data') # Directory for saving the outputs including the stimuli to be used 
#%% Current experiment
current_exp_ID = '230901bg_fly1'
current_t_series ='TSeries-09012023-1344-002'
Genotype = 'Tm9Rec_lexAopGC6f'
Age = '3'
Sex = 'f'

#%%
roi_save_dir = os.path.join(save_output_dir, current_exp_ID,current_t_series)

if not os.path.exists(os.path.dirname(roi_save_dir)):
    os.mkdir(os.path.dirname(roi_save_dir))

if not os.path.exists(roi_save_dir):
    os.mkdir(roi_save_dir)

#%% Motion correction
answer = raw_input("Motion alignment? (y/n)")
if answer =='y':
    print('Performing motion alignment...\n')
    experiment_path = os.path.join(raw_data_dir, current_exp_ID)
    t_series_path = os.path.join(experiment_path, current_t_series)
    t_series_files = t_series_path + '/' + '*.tif'
    output_path = os.path.join(aligned_data_dir, current_exp_ID)

    # Align T-series
    
    if not os.path.exists(os.path.join(aligned_data_dir, current_exp_ID)):
        os.mkdir(os.path.join(aligned_data_dir, current_exp_ID))
    unique_t_series = motionCorrection(current_t_series, output_path, t_series_files,
                                        maxDisplacement=[40, 40],
                                        granularity='plane',
                                        exportFrames=True)
    # os.mkdir(os.path.join(aligned_data_dir, current_exp_ID,current_t_series))
    t_s_path = os.path.join(experiment_path, current_t_series)
    copyStimXmlFilesPyStim(output_path, t_s_path, current_t_series)

    print('Motion alignment successfully done')

#%%
plt.switch_backend('Qt5Agg') # Working with Visual Studio code, interactive plotting for ROI selection
plt.style.use('default')
plt.close('all')
# Analysis parameters
analysis_type = 'WN'
# ROI selection/extraction parameters
extraction_type = 'manual' # 'SIMA-STICA' 'transfer' 'manual'
transfer_data_name = '200818bg_fly3-TSeries-08182020-0830-006_manual.pickle'

#%% Get the stimulus and imaging information
current_movie_ID = current_exp_ID + '-' + current_t_series # A unique ID for this might will be useful
dataDir = os.path.join(aligned_data_dir, current_exp_ID, current_t_series) 

(time_series, stim_info,imaging_info) = aCore.preProcessMovie(dataDir)

experiment_conditions = \
    {'Genotype' : Genotype, 'Age': Age, 'Sex' : Sex,
     'FlyID' : current_exp_ID, 'MovieID': current_movie_ID}

figure_save_dir = os.path.join(dataDir, 'Results')
if not os.path.exists(figure_save_dir):
    os.mkdir(figure_save_dir)
    

#%% Define analysis/extraction parameters and run region selection
#   generate ROI objects.

# Organizing extraction parameters
extraction_params = \
    aCore.organizeExtractionParams(extraction_type,
                               current_t_series=current_t_series,
                               current_exp_ID=current_exp_ID,
                               alignedDataDir=aligned_data_dir,
                               use_other_series_roiExtraction = False,
                               use_avg_data_for_roi_extract = False,
                               roiExtraction_tseries=0,
                               transfer_data_n = transfer_data_name,
                               transfer_data_store_dir = save_output_dir,
                               transfer_type = None,
                               imaging_info=imaging_info,
                               experiment_conditions=experiment_conditions)
        
    
analysis_params = {'deltaF_method': 'mean',
                   'analysis_type': analysis_type} 


#%% Select/extract ROIs
mean_image = time_series.mean(0)
(cat_masks, cat_names, roi_masks, all_rois_image, rois,
threshold_dict) = \
    aCore.selectROIs(extraction_params,image_to_select=mean_image)

# A mask needed in SIMA STICA to exclude ROIs based on regions
cat_bool = np.zeros(shape=np.shape(mean_image))
for idx, cat_name in enumerate(cat_names):
    if cat_name.lower() == 'bg':
        bg_mask = cat_masks[idx]
        continue
    elif cat_name.lower() =='otsu':
        otsu_mask = cat_masks[idx]
        continue
    cat_bool[cat_masks[cat_names.index(cat_name)]] = 1
    
# Generate ROI instances if not generated before
if rois == None:
    del rois
    rois = PyROI.generateROIs(roi_masks, cat_masks, cat_names, mean_image,
                                experiment_info = experiment_conditions, 
                                imaging_info =imaging_info)
else:
    # Add the source image - this will be used to plot the masks on top
    map(lambda roi: roi.setSourceImage(mean_image), rois)

# We can store the parameters inside the objects for further use
for roi in rois:
    roi.extraction_params = extraction_params
    roi.stim_info = stim_info 
    if extraction_type == 'transfer': # Update transferred ROIs
        roi.experiment_info = experiment_conditions
        roi.imaging_info = imaging_info
        for param in analysis_params.keys():
            roi.analysis_params[param] = analysis_params[param]
    else:
        roi.analysis_params= analysis_params


# %% 
# BG subtraction
time_series = np.transpose(np.subtract(np.transpose(time_series),
                                       time_series[:,bg_mask].mean(axis=1)))
print('\n Background subtraction done...')
# Stimulus 
stim = stim_info['meta']['epoch_infos']['epoch_2']['noise_texture']
# ROI raw signals
for iROI, roi in enumerate(rois):
    roi.raw_trace = time_series[:,roi.mask].mean(axis=1)
    roi.wn_stim = stim

#%% White noise analysis and selecting the best RF ROI

rois = PyROI.reverseCorrelation(rois,stim_up_rate=20)
print('RFs computed...')

final_rois = rois

for roi in final_rois:
    max_t = np.where(np.abs(roi.sta)==np.abs(roi.sta).max())[0][0]
    curr_sta = roi.sta[max_t,:,:]
    roi.max_t_sta = curr_sta
    roi.max_RF = curr_sta
    roi.RF_max_val = np.abs(roi.sta).max()
    roi.RF_quality = np.abs(roi.sta).max()/(roi.max_RF.mean()+(3*roi.max_RF.std()))

    # Fit Gaussian
    if np.abs(curr_sta).max() != curr_sta.max(): # Data should be transformed if the peak is negative
        data_to_fit = curr_sta*-1
    else:
        data_to_fit = curr_sta

    gauss_params, success, fit, r_squared = PyROI.fitTwoDgaussian(data_to_fit)

    (height, x, y, width_x, width_y) = gauss_params
    roi.rf_fit = fit
    roi.rf_fit_rsq = r_squared
    # center_coords = np.array(np.where(np.abs(curr_sta)==np.abs(curr_sta).max())).astype(float) # center as max
    center_coords = np.array([x,y]) # center as the center of the fitted gaussian

    wn_dim = roi.stim_info['meta']['epoch_infos']['epoch_2']['x_width']
    center_coords *= wn_dim # Find in degrees
    center_coords += wn_dim/2.0 # Find the center of the square

    # Transform array coordinates to screen coordinates in degrees
    # From fly perspective, up and right are minus coordinates (0 is center) 
    # but projection is flipped in X so in the array left and up are minus, 
    # right and down are plus

    center_coords[1] -= int(roi.stim_info['meta']['proj_params']['sizeX'])/2
    
    center_coords[0] = int(roi.stim_info['meta']['proj_params']['sizeX'])/2 - center_coords[0]
    center_coords[0] *= -1 
    roi.center_coords = center_coords



#%% Select the best ROI
rf_qualities = np.array([roi.RF_quality for roi in final_rois])
coords = np.array([roi.center_coords for roi in final_rois])

# We need 15 degrees from each side of the screen in order to present 30 degrees of gratings centered in the RF
center_roi_indices = np.where(np.sum(np.abs(coords)<15,axis=1)==2)[0]
if not center_roi_indices.any():
    chosen_roi = final_rois[0]
    warn('ROI RF centers are not suitable for stimulation!!!')
    no_suitable_roi = True
else:
    chosen_roi_idx = center_roi_indices[rf_qualities[center_roi_indices].argmax()]
    chosen_roi = final_rois[chosen_roi_idx]
    no_suitable_roi = False

fig, axes = plt.subplots(ncols=1, nrows=2, figsize=(10, 10))
plt.sca(axes[0])
chosen_roi.showRoiMask(cmap='Dark2')

sns.heatmap(chosen_roi.max_RF, ax=axes[1],cmap='coolwarm', cbar=True, center=0)
axes[1].contour(chosen_roi.rf_fit,alpha=.7,cmap='coolwarm')
axes[1].axis('off')
axes[1].set_title('X center: {x} - Y center: {y}'.format(x=int(chosen_roi.center_coords[1]),
    y=int(chosen_roi.center_coords[0])))



f_name = 'Selected_ROI' 
os.chdir(roi_save_dir)
fig.savefig('%s.png'% f_name, bbox_inches='tight',transparent=False)

print('Best ROI selected...')
#%%
# Make an image of masks
mask_image = PyROI.getMasksImage(final_rois)
aCore.plotAllMasks(mask_image, mean_image,len(final_rois),current_movie_ID,
    save_fig = True, save_dir = roi_save_dir)

# Plot and save the RF
PyROI.plotRFs_WN(final_rois, fig_save_dir=roi_save_dir,number = len(final_rois),
    f_w = 6)
PyROI.plotRFs_WN(final_rois, fig_save_dir=roi_save_dir,number = len(final_rois),
    f_w = 6,fit_plot=True)

#%% Save data
save_dict={'final_rois':final_rois}
save_name = os.path.join(roi_save_dir,'{ID}_onlineRF.pickle'.format(ID=current_movie_ID))
saveVar = open(save_name, "wb")
cPickle.dump(save_dict, saveVar, protocol=2) # Protocol 2 (and below) is compatible with Python 2.7 downstream analysis
saveVar.close()
print('\n\n%s saved...\n\n' % save_name)      

#%% Generate .txt files for PyStim stimulation
# x_coord = int(chosen_roi.center_coords[1])
# y_coord = int(chosen_roi.center_coords[0])


no_suitable_roi = False
x_coord = 0
y_coord = -2

if no_suitable_roi:
    warn('No stimulus input file is generated. RFs not suitable.')
else:
    os.chdir(roi_save_dir)
    stim_temp_dir = os.path.join(initial_dir,'stimSamples')
    # Following stimuli are centered drifting grating with constant contrast and luminance with annuli of changing diameter and luminance
    # Read the stimulus template:
    
    stim_name = 'stim-input_circCentGrat_5degGrating_3s_circ4Lum9Size_staticGratingBG__30degLambda_X0Y0_222s.txt'
    stim_temp_path = os.path.join(stim_temp_dir,stim_name)

    # Read the file into a pandas DataFrame
    stim_df = pd.read_csv(stim_temp_path,sep='\t', index_col=0)
    stim_df.loc['param9'] = x_coord
    stim_df.loc['param10'] = y_coord
    stim_df.loc['param12'] = x_coord
    stim_df.loc['param13'] = y_coord

    stim_save_name = 'stim-input_circCentGrat_5degGrating_3s_circ4Lum9Size_staticGratingBG__30degLambda_X{x}Y{y}_222s.txt'.format(x=x_coord,
            y= y_coord)
    output_file_path = os.path.join(roi_save_dir, stim_save_name)
    stim_df.to_csv(output_file_path, sep='\t')


    # Next stim
    # Read the stimulus template:
    stim_name = 'stim-input_centerGrating_5-10-15-20-25-30deg_x0_y0_4s_5lum_5dps_5degrees_240s.txt'
    stim_temp_path = os.path.join(stim_temp_dir,stim_name)

    # Read the file into a pandas DataFrame
    stim_df = pd.read_csv(stim_temp_path,sep='\t', index_col=0)
    stim_df.loc['param9'] = x_coord
    stim_df.loc['param10'] = y_coord

    stim_save_name = 'stim-input_centerGrating_5-10-15-20-25-30deg_x{x}_y{y}_4s_5lum_5dps_5degrees_240s.txt'.format(x=x_coord,
            y= y_coord)
    output_file_path = os.path.join(roi_save_dir, stim_save_name)
    stim_df.to_csv(output_file_path, sep='\t')


    print("Stimuli generated for x={x}, y={y}.".format(x=x_coord,y=y_coord))

#%% These are for presenting centered drifting gratings with changing diameters
    # degs = [5, 10, 15, 20,25,30]
    # for deg in degs:
    #     fname = "stim-input_centerGrating_{deg}deg_x{x}_y{y}_5lum_30dps_30degrees_40s.txt".format(x=int(chosen_roi.center_coords[1]),
    #         y= int(chosen_roi.center_coords[0]), deg= str(deg).replace('.','_'))
    #     f=open(fname, "a+")
    #     f.write('total_epoch_num\t6\n')
    #     f.write('randomization_condition\t1\n')
    #     f.write('epoch_nums\t1\t2\t3\t4\t5\t6\n')
    #     f.write('stim_type\tfff-v1\t{s}\t{s}\t{s}\t{s}\t{s}\n'.format(s='centered-gratings-v1'))
    #     f.write('param1\t{s}\t{s}\t{s}\t{s}\t{s}\t{s}\n'.format(s='4'))
    #     f.write('param2\t0.2625\t{s}\t{s}\t{s}\t{s}\t{s}\n'.format(s='sin'))
    #     f.write('param3\t\t{s}\t{s}\t{s}\t{s}\t{s}\n'.format(s='30'))
    #     f.write('param4\t\t{s}\t{s}\t{s}\t{s}\t{s}\n'.format(s='30'))

    #     f.write('param5\t\t{s}\t{s}\t{s}\t{s}\t{s}\n'.format(s='90'))
    #     f.write('param6\t\t{s}\t{s}\t{s}\t{s}\t{s}\n'.format(s='1'))
    #     f.write('param7\t\t0.0625\t0.125\t0.25\t0.375\t0.5\n')
    #     f.write('param8\t\t{s}\t{s}\t{s}\t{s}\t{s}\n'.format(s=str(deg)))
    #     f.write('param9\t\t{s}\t{s}\t{s}\t{s}\t{s}\n'.format(s=str(int(chosen_roi.center_coords[1]))))
    #     f.write('param10\t\t{s}\t{s}\t{s}\t{s}\t{s}\n'.format(s=str(int(chosen_roi.center_coords[0]))))
    #     f.write('param11\t\t{s}\t{s}\t{s}\t{s}\t{s}\n'.format(s=0.2625))
    #     f.close()


    #     # Generate uniform luminance -100 weber contrast flashes stimulus - darkBG
    #     fname = "stim-input_centerCircles_darkBG_{deg}deg_x{x}_y{y}_6lum_-1WeberC_30s.txt".format(x=int(chosen_roi.center_coords[1]),
    #         y= int(chosen_roi.center_coords[0]), deg= str(deg).replace('.','_'))
    #     f=open(fname, "a+")
    #     f.write('total_epoch_num\t7\n')
    #     f.write('randomization_condition\t1\n')
    #     f.write('epoch_nums\t1\t2\t3\t4\t5\t6\t7\n')
    #     f.write('stim_type\tfff-v1\t{s}\t{s}\t{s}\t{s}\t{s}\t{s}\n'.format(s='centered-circle-v1'))
    #     f.write('param1\t2\t{s}\t{s}\t{s}\t{s}\t{s}\t{s}\n'.format(s='3'))
    #     f.write('param2\t0\t0.0625\t0.125\t0.25\t0.5\t0.75\t1\n')

    #     f.write('param3\t\t{s}\t{s}\t{s}\t{s}\t{s}\t{s}\n'.format(s='-1'))
    #     f.write('param4\t\t{s}\t{s}\t{s}\t{s}\t{s}\t{s}\n'.format(s='1'))
    #     f.write('param5\t\t{s}\t{s}\t{s}\t{s}\t{s}\t{s}\n'.format(s=str(deg)))
    #     f.write('param6\t\t{s}\t{s}\t{s}\t{s}\t{s}\t{s}\n'.format(s=str(int(chosen_roi.center_coords[1]))))
    #     f.write('param7\t\t{s}\t{s}\t{s}\t{s}\t{s}\t{s}\n'.format(s=str(int(chosen_roi.center_coords[0]))))
    #     f.write('param8\t\t{s}\t{s}\t{s}\t{s}\t{s}\t{s}\n'.format(s='0'))
    #     f.close()


    #     # Generate uniform luminance -100 weber contrast flashes stimulus - bright BG
    #     fname = "stim-input_centerCircles_brightBG_{deg}deg_x{x}_y{y}_6lum_-1WeberC_30s.txt".format(x=int(chosen_roi.center_coords[1]),
    #         y= int(chosen_roi.center_coords[0]), deg= str(deg).replace('.','_'))
    #     f=open(fname, "a+")
    #     f.write('total_epoch_num\t7\n')
    #     f.write('randomization_condition\t1\n')
    #     f.write('epoch_nums\t1\t2\t3\t4\t5\t6\t7\n')
    #     f.write('stim_type\tfff-v1\t{s}\t{s}\t{s}\t{s}\t{s}\t{s}\n'.format(s='centered-circle-v1'))
    #     f.write('param1\t2\t{s}\t{s}\t{s}\t{s}\t{s}\t{s}\n'.format(s='3'))
    #     f.write('param2\t1\t0.0625\t0.125\t0.25\t0.5\t0.75\t1\n')

    #     f.write('param3\t\t{s}\t{s}\t{s}\t{s}\t{s}\t{s}\n'.format(s='-1'))
    #     f.write('param4\t\t{s}\t{s}\t{s}\t{s}\t{s}\t{s}\n'.format(s='1'))
    #     f.write('param5\t\t{s}\t{s}\t{s}\t{s}\t{s}\t{s}\n'.format(s=str(deg)))
    #     f.write('param6\t\t{s}\t{s}\t{s}\t{s}\t{s}\t{s}\n'.format(s=str(int(chosen_roi.center_coords[1]))))
    #     f.write('param7\t\t{s}\t{s}\t{s}\t{s}\t{s}\t{s}\n'.format(s=str(int(chosen_roi.center_coords[0]))))
    #     f.write('param8\t\t{s}\t{s}\t{s}\t{s}\t{s}\t{s}\n'.format(s='1'))
    #     f.close()


# %%

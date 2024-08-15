#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 13 13:52:56 2020

@author: burakgur
"""
#%% 2nd step of pre-processing: This code performs preprocessing of motion corrected datasets
# %% Importing packages
import os
import copy
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
os.chdir('/Volumes/Backup Plus/Post-Doc/_SiliesLab/Manuscripts/2023_Lum_Gain/Data_code/code/python2p7/pre-processing')
import ROI_mod
from core_functions import saveWorkspace
import process_mov_core as pmc

plt.switch_backend('Qt5Agg') # wrking with Visual Studio code, interactive plotting for ROI selection
plt.style.use('default')

# %% Setting the directories -> Change to your directories!!
initialDirectory = '/Volumes/HD-SP1/Burak_data/Python_data' # Main directory
alignedDataDir = os.path.join(initialDirectory, 'selected_experiments') # The directory where aligned dataset are located 
stimInputDir = os.path.join(initialDirectory, 'stimulus_types_cPP') # Need stimulus input files for extracting stimulus parameters
saveOutputDir = os.path.join(initialDirectory, 'analyzed_data/...') # Where to save the processed data 
summary_save_dir = os.path.join(alignedDataDir, '_summaries') # Summary plots for inspection

# %% Parameters to adjust
plt.close('all')

# Experimental parameters
current_exp_ID = '220125bg_fly2' # Folder
current_t_series ='TSeries-01252022-1014-007' #T series
Genotype = 'R64G09-Recomb-Homo' # Genotype
Age = '1-3' # Age range
Sex = 'f' # 'm' 'f'

# Stimulus types
analysis_type = 'lum_con_gratings'
# 'lum_con_gratings' # Drifting sinusoidal gratings 5 luminance and 5 contrast
# 'luminance_edges_OFF' # Drifting OFF edges
# 'luminance_gratings' # Drifting sinusoidal gratings constant contrast, changing luminances
# '5sFFF_analyze_save' # Periodic full field flashes

# ROI selection/extraction parameters (Manual is used for everything except the T4/T5 recordings where SIMA-STICA is used)
extraction_type = 'manual'  # 'SIMA-STICA' used for T4 and T5 recordings, 'transfer' 'manual'
transfer_type = 'minimal' # 'minimal' 'predefined' => #
transfer_data_name = '200908bg_fly7-TSeries-09082020-0924-007_manual.pickle'

# Not used in the manuscript
use_avg_data_for_roi_extract = False
use_other_series_roiExtraction = False # ROI selection video
roiExtraction_tseries = ''

# Plot related
plot_roi_summ = False


#%% Get the stimulus and imaging information
dataDir = os.path.join(alignedDataDir, current_exp_ID, current_t_series)

(time_series, stimulus_information,imaging_information) = \
    pmc.pre_processing_movie (dataDir,stimInputDir)
mean_image = time_series.mean(0)
current_movie_ID = current_exp_ID + '-' + current_t_series
figure_save_dir = os.path.join(dataDir, 'Results')
if not os.path.exists(figure_save_dir):
    os.mkdir(figure_save_dir)
       
# Generate an average dataset for epoch visualization if desired
(wholeTraces_allTrials_video, respTraces_allTrials, 
     baselineTraces_allTrials) = \
        pmc.separate_trials_video(time_series,stimulus_information,
                                  imaging_information['frame_rate'])
pmc.generate_avg_movie(dataDir, stimulus_information,
                       wholeTraces_allTrials_video)

experiment_conditions = \
    {'Genotype' : Genotype, 'Age': Age, 'Sex' : Sex,
     'FlyID' : current_exp_ID, 'MovieID': current_movie_ID}
    

#%% Define analysis/extraction parameters and run region selection
#   generate ROI objects.

# Organizing extraction parameters
if transfer_type == 'predefined':
    transfer_type = analysis_type
    
extraction_params = \
    pmc.organize_extraction_params(extraction_type,
                               current_t_series=current_t_series,
                               current_exp_ID=current_exp_ID,
                               alignedDataDir=alignedDataDir,
                               stimInputDir=stimInputDir,
                               use_other_series_roiExtraction = use_other_series_roiExtraction,
                               use_avg_data_for_roi_extract = use_avg_data_for_roi_extract,
                               roiExtraction_tseries=roiExtraction_tseries,
                               transfer_data_n = transfer_data_name,
                               transfer_data_store_dir = saveOutputDir,
                               transfer_type = transfer_type,
                               imaging_information=imaging_information,
                               experiment_conditions=experiment_conditions)
        
    
analysis_params = {'deltaF_method': 'mean',
                   'analysis_type': analysis_type} 


# Select/extract ROIs
(cat_masks, cat_names, roi_masks, all_rois_image, rois,
threshold_dict) = \
    pmc.run_ROI_selection(extraction_params,image_to_select=mean_image)

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
    
# Generate ROI_bg instances
if rois == None:
    del rois
    rois = ROI_mod.generate_ROI_instances(roi_masks, cat_masks, cat_names,
                                          mean_image, 
                                          experiment_info = experiment_conditions, 
                                          imaging_info =imaging_information)
    
# We can store the parameters inside the objects for further use
for roi in rois:
    roi.extraction_params = extraction_params
    if extraction_type == 'transfer': # Update transferred ROIs
        roi.experiment_info = experiment_conditions
        roi.imaging_info = imaging_information
        for param in analysis_params.keys():
            roi.analysis_params[param] = analysis_params[param]
    else:
        roi.analysis_params= analysis_params


# %% 
# BG subtraction
time_series = np.transpose(np.subtract(np.transpose(time_series),
                                       time_series[:,bg_mask].mean(axis=1)))
print('\n Background subtraction done...')
    
# ROI trial separated responses
(wholeTraces_allTrials_ROIs, respTraces_allTrials_ROIs,
 baselineTraces_allTrials_ROIs) = \
    pmc.separate_trials_ROI_v4(time_series,rois,stimulus_information,
                               imaging_information['frame_rate'],
                               df_method = analysis_params['deltaF_method'])

# Append relevant information and calculate some parameters
map(lambda roi: roi.appendStimInfo(stimulus_information), rois)
map(lambda roi: roi.findMaxResponse_all_epochs(), rois)
map(lambda roi: roi.setSourceImage(mean_image), rois)

# Calculate SNR and reliability
(_, respTraces_SNR, baseTraces_SNR) = \
    pmc.separate_trials_ROI_v4(time_series,rois,stimulus_information,
                               imaging_information['frame_rate'],
                               df_method = analysis_params['deltaF_method'],
                               df_use=False)
# SNR and reliability
if stimulus_information['random'] == 2:
    epoch_to_exclude = None
    baseTraces_SNR = respTraces_SNR.copy()
    # baseTraces_SNR[]
elif stimulus_information['random'] == 0:
    epoch_to_exclude = stimulus_information['baseline_epoch']
else:
    epoch_to_exclude = None

[SNR_rois, corr_rois] = pmc.calculate_SNR_Corr(baseTraces_SNR,
                                               respTraces_SNR,rois,
                                               epoch_to_exclude=None)
    


# Thresholding
if threshold_dict is None:
    print('No threshold used, all ROIs will be retained')
    thresholded_rois = rois
else:
    print('Thresholding ROIs')
    thresholded_rois = ROI_mod.threshold_ROIs(rois, threshold_dict)




# Exclude and separate overlapping clusters
if extraction_params['type'] == 'SIMA-STICA':
    # Otsu mask for excluding background clusters
    roi_1d_max_size_pixel = \
        extraction_params['cluster_max_1d_size_micron'] / imaging_information['pixel_size']
    roi_1d_min_size_pixel = \
        extraction_params['cluster_min_1d_size_micron'] / imaging_information['pixel_size']
    final_rois, final_roi_image = pmc.refine_rois(thresholded_rois, cat_bool, 
                                                  extraction_params,
                                                  roi_1d_max_size_pixel,
                                                  roi_1d_min_size_pixel,
                                                  use_otsu=True,
                                                  mean_image=mean_image,
                                                  otsu_mask=otsu_mask)

else:
    final_rois = thresholded_rois
    final_roi_image = ROI_mod.get_masks_image(final_rois)
    
# Plotting ROIs and properties
# pmc.plot_roi_masks(final_roi_image,mean_image,len(final_rois),
#                    current_movie_ID,save_fig=True,
#                    save_dir=figure_save_dir,alpha=0.4)


# Run desired analyses for different types
final_rois = pmc.run_analysis(analysis_params,final_rois,experiment_conditions,
                              imaging_information,summary_save_dir,
                              save_fig=True,fig_save_dir = figure_save_dir,
                              exp_ID=('%s_%s' % (current_movie_ID,
                                                 extraction_params['type'])))

# Make figures for experiment summary
images = []
(properties, colormaps, vminmax, data_to_extract) = \
    pmc.select_properties_plot(final_rois , analysis_params['analysis_type'])
for prop in properties:
    images.append(ROI_mod.generate_colorMasks_properties(final_rois, prop))
pmc.plot_roi_properties(images, properties, colormaps, mean_image,
                        vminmax,current_movie_ID, imaging_information['depth'],
                        save_fig=True, save_dir=figure_save_dir,figsize=(8, 6),
                        alpha=0.5)
final_roi_data = ROI_mod.data_to_list(final_rois, data_to_extract)
rois_df = pd.DataFrame.from_dict(final_roi_data)

pmc.plot_df_dataset(rois_df,data_to_extract,
                    exp_ID=('%s_%s' % (current_movie_ID,
                                       extraction_params['type'])),
                    save_fig=True, save_dir=figure_save_dir)
# plt.close('all')
  
# %% PART 4: Save data
os.chdir('/Users/burakgur/Documents/GitHub/python_lab/2p_calcium_imaging')
varDict = locals()
pckl_save_name = ('%s_%s' % (current_movie_ID, extraction_params['type']))
saveWorkspace(saveOutputDir,pckl_save_name, varDict, 
              varFile='data_save_vars.txt',extension='.pickle')

print('\n\n%s saved...\n\n' % pckl_save_name)

#%% Plot ROI summarries
if plot_roi_summ:
    if analysis_type == 'gratings_transfer_rois_save' :
        import random
        plt.close('all')
        data_to_extract = ['DSI', 'BF', 'SNR', 'reliability', 'uniq_id','CSI',
                               'PD', 'exp_ID', 'stim_name']
        
        
        roi_figure_save_dir = os.path.join(figure_save_dir, 'ROI_summaries')
        if not os.path.exists(roi_figure_save_dir):
            os.mkdir(roi_figure_save_dir)
        copy_rois = copy.deepcopy(final_rois)
        random.shuffle(copy_rois)
        roi_d = ROI_mod.data_to_list(copy_rois, data_to_extract)
        rois_df = pd.DataFrame.from_dict(roi_d)
        for n,roi in enumerate(copy_rois):
            if n>40:
                break
            fig = ROI_mod.make_ROI_tuning_summary(rois_df, roi,cmap='coolwarm')
            save_name = '%s_ROI_summary_%d' % ('a', roi.uniq_id)
            os.chdir(roi_figure_save_dir)
            fig.savefig('%s.png' % save_name,bbox_inches='tight',
                               transparent=False,dpi=300)
                
            plt.close('all')
    elif (analysis_type == 'STF_1'):
        data_to_extract = ['reliability', 'uniq_id','CSI']
        
        roi_figure_save_dir = os.path.join(figure_save_dir, 'ROI_summaries')
        if not os.path.exists(roi_figure_save_dir):
            os.mkdir(roi_figure_save_dir)
        
        roi_d = ROI_mod.data_to_list(final_rois, data_to_extract)
        rois_df = pd.DataFrame.from_dict(roi_d)
        for n,roi in enumerate(final_rois):
            if n>40:
                break
            fig = ROI_mod.plot_stf_map(roi,rois_df)
            save_name = '%s_ROI_STF_%d' % (roi.analysis_params['roi_sel_type'], roi.uniq_id)
            os.chdir(roi_figure_save_dir)
            fig.savefig('%s.png' % save_name,bbox_inches='tight',
                               transparent=False,dpi=300)
                
            plt.close('all')
        
        
    
    
    
    
    
    
    
    
    
    
    
    
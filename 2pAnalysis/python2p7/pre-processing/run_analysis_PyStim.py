# -*- coding: utf-8 -*-
"""
Created on Nov 27 2020

Calcium imaging analysis using PyStim outputs.

@author: burakgur
"""
#%% 2nd step of pre-processing: This code performs preprocessing of motion corrected datasets

# %% Importing packages
import os
import numpy as np
import matplotlib.pyplot as plt
import cPickle

os.chdir('/Volumes/Backup Plus/Post-Doc/_SiliesLab/Manuscripts/2023_Lum_Gain/Data_code/code/python2p7/pre-processing')

import analysis_core as aCore 
import PyROI

plt.switch_backend('Qt5Agg') # Working with Visual Studio code, interactive plotting for ROI selection
plt.style.use('default')

# %% Setting the directories -> Change to your directories!!
initialDirectory = '/Volumes/HD-SP1/Burak_data/Python_data' # Main directory
alignedDataDir = os.path.join(initialDirectory,'selected_experiments') # The directory where aligned dataset are located 
stimInputDir = os.path.join(initialDirectory, 'stimulus_types_cPP') # Need stimulus input files for extracting stimulus parameters
saveOutputDir = os.path.join(initialDirectory, 'analyzed_data/...') # Where to save the processed data 
summary_save_dir = os.path.join(alignedDataDir, '_summaries') # Summary plots for inspection
#%% Initialization
# Experimental parameters
current_exp_ID = '220125bg_fly2' # Folder
current_t_series ='TSeries-01252022-1014-007' #T series
Genotype = 'R64G09-Recomb-Homo' # Genotype
Age = '1-3' # Age range
Sex = 'f' # 'm' 'f'

# Stimulus types
analysis_type = 'luminance_gratings'

# Current stimulus battery
# 'luminance_gratings' # Drifting sinusoidal gratings constant contrast, changing luminances
# 'centered_gratings_expanding_bgCircle' # centered drifting gratings with annuli of different luminance and diameter
# 'centered_gratings_lum_size' # centered drifting gratings with changing diameter
# luminance_edges_OFF # Drifting OFF edges
# 'lum_con_gratings' # Drifting sinusoidal gratings 5 luminance and 5 contrast

# This part is used for matching the centered grating. For each fly, ROI IDs can be changed
add_single_ROI_to_save_name = 0
center_grating_roiID = 1 # This will just be used for centered gratings to keep the ROI identities accross stimuli
# This should change for ROIs within a single fly but doesn't need to change between flies since flyIDs will differentiate them

# ROI selection/extraction parameters (Manual is used for everything except the T4/T5 recordings where SIMA-STICA is used)
extraction_type = 'manual'  # 'SIMA-STICA' used for T4 and T5 recordings, 'transfer' 'manual'
transfer_type = 'minimal' # 'minimal' 'predefined'
transfer_data_name = '200908bg_fly7-TSeries-09082020-0924-007_manual.pickle'

# Not used in the manuscript
use_avg_data_for_roi_extract = False
use_other_series_roiExtraction = False # ROI selection video
roiExtraction_tseries = 'TSeries-02132020-1511-002'
rf_identity_keep = 0
rf_image_id = '210416bg_fly1-TSeries-04162021-0948-002.pickle'

#%% Get the stimulus and imaging information
current_movie_ID = current_exp_ID + '-' + current_t_series # A unique ID for this might will be useful
dataDir = os.path.join(alignedDataDir, current_exp_ID, current_t_series) 

(time_series, stim_info,imaging_info) = aCore.preProcessMovie(dataDir)

experiment_conditions = \
    {'Genotype' : Genotype, 'Age': Age, 'Sex' : Sex,
     'FlyID' : current_exp_ID, 'MovieID': current_movie_ID}


if add_single_ROI_to_save_name:
    figure_save_dir = os.path.join(dataDir, 'Results_singleROI')
else:
    figure_save_dir = os.path.join(dataDir, 'Results')
if not os.path.exists(figure_save_dir):
    os.mkdir(figure_save_dir)
#%% Organize
# Organizing extraction parameters
if transfer_type == 'predefined':
    transfer_type = analysis_type
    
extraction_params = \
    aCore.organizeExtractionParams(extraction_type,
                               current_t_series=current_t_series,
                               current_exp_ID=current_exp_ID,
                               alignedDataDir=alignedDataDir,
                               use_other_series_roiExtraction = use_other_series_roiExtraction,
                               use_avg_data_for_roi_extract = use_avg_data_for_roi_extract,
                               roiExtraction_tseries=roiExtraction_tseries,
                               transfer_data_n = transfer_data_name,
                               transfer_data_store_dir = saveOutputDir,
                               transfer_type = transfer_type,
                               imaging_info=imaging_info,

                               experiment_conditions=experiment_conditions)
        
    
analysis_params = {'deltaF_method': 'mean',
                   'analysis_type': analysis_type} 
if add_single_ROI_to_save_name:
    analysis_params['center_grating_roiID'] = center_grating_roiID
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

    if rf_identity_keep: # For keeping which RF data to use
        roi.rf_image_id = rf_image_id


#%% Processing ROI time traces
# BG subtraction
time_series = np.transpose(np.subtract(np.transpose(time_series),
                                       time_series[:,bg_mask].mean(axis=1)))
print('\nBackground subtraction done...')

# ROI trial separated responses
rois = PyROI.getTimeTraces(rois,time_series)
rois = PyROI.averageTrials(rois)

# Run desired analyses for different stimulus types
final_rois = aCore.analyzeTraces(rois,analysis_params,
    save_fig=True,fig_save_dir = figure_save_dir,summary_save_d=summary_save_d)

# Make an image of masks
mask_image = PyROI.getMasksImage(final_rois)
aCore.plotAllMasks(mask_image, mean_image,len(final_rois),current_movie_ID,
    save_fig = True, save_dir = figure_save_dir)
PyROI.plotMasksROInums(final_rois,mean_image,save_fig = True, 
    save_dir = figure_save_dir,save_id = current_movie_ID)

# Plot and save the trace figure 
PyROI.plotAllTraces(rois,fig_save_dir=figure_save_dir)
#%% Save data
save_dict={'final_rois':final_rois}
if add_single_ROI_to_save_name:
    save_name = os.path.join(saveOutputDir,'{ID}_singleROI.pickle'.format(ID=current_movie_ID))
else:
    save_name = os.path.join(saveOutputDir,'{ID}.pickle'.format(ID=current_movie_ID))
saveVar = open(save_name, "wb")
cPickle.dump(save_dict, saveVar, protocol=2) # Protocol 2 (and below) is compatible with Python 2.7 downstream analysis
saveVar.close()
print('\n\n%s saved...\n\n' % save_name)
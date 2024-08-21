#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""

@author: burakgur
"""

#%%
import cPickle
import pandas as pd
import numpy as np
import warnings
import matplotlib
import matplotlib.pyplot as plt
import os
from scipy.stats import linregress
import seaborn as sns

#change to code directory
os.chdir('.../Gur-etal-2024/2pAnalysis/python2p7/common')

import post_analysis_core as pac

# plt.switch_backend('Qt5Agg') # Working with Visual Studio code, interactive plotting for ROI selection
plt.style.use('default')
#%% Directories for loading data and saving figures (ADJUST THEM TO YOUR PATHS)
initialDirectory = 'data_path'
all_data_dir = os.path.join(initialDirectory, 'raw_data')
results_save_dir = os.path.join(initialDirectory,'Figure4/plots')


#%% Luminance values of the experimental setup (ADJUST THE PATHS)
# Values from the stimulation paradigm
measurements_f = '.../Gur-etal-2024/2pAnalysis/luminance_measurements/210716_Ultima_Luminances_ONOFFpaper.xlsx'
measurement_df = pd.read_excel(measurements_f,header=0)
res = linregress(measurement_df['file_lum'], measurement_df['measured'])
# %% Load datasets and desired variables
exp_folder = 'SpatialPooling_Tm1_Tm9/changingGratingSize'
data_dir = os.path.join(all_data_dir,exp_folder)
datasets_to_load = os.listdir(data_dir)

                    
all_rois = []
roi_data = {}
# Initialize variables

for idataset, dataset in enumerate(datasets_to_load):
    if not(dataset.split('.')[-1] =='pickle'):
        warnings.warn('Skipping non pickle file: {f}\n'.format(f=dataset))
        continue
    load_path = os.path.join(data_dir, dataset)
    load_path = open(load_path, 'rb')
    try:
        workspace = cPickle.load(load_path)
    except ImportError:
        print('Unable to import skipping: \n{f} '.format(f=load_path))
        continue
    curr_rois = workspace['final_rois']
    

    # The roi is going to have a gratings_roi_id
    if  not(('centerGrating' in curr_rois[0].stim_info['meta']['stim_name']) or (('singleROI' in dataset) and ("stim-input_gratings-100p-5lum-40s" in curr_rois[0].stim_info['meta']['stim_name']))):
        continue
    print(curr_rois[0].stim_info['meta']['stim_name'])
    roi = curr_rois[0]
    # Filter out a very noisy ROI:
    if roi.experiment_info['FlyID'] in ['210311bg_fly1','210317bg_fly2']: # Tm9 Tm1very noisy ROI
        continue


    # There will always be a single ROI
    
    if ("stim-input_gratings-100p-5lum-40s" in roi.stim_info['meta']['stim_name']):
        roi.gratings_roi_id = '{flyID}-{roi_id}'.format(flyID=roi.experiment_info['FlyID'],
                                                                roi_id=1)
    geno = roi.experiment_info['Genotype'][:3]
    print(geno)
    
    # Storing in a dictionary
    if not(roi_data.has_key(roi.gratings_roi_id)):
        roi_data[roi.gratings_roi_id] = {}
        roi_data[roi.gratings_roi_id]['responses'] = np.empty((7,5,))
        roi_data[roi.gratings_roi_id]['responses'][:] = np.nan

        roi_data[roi.gratings_roi_id]['responses_norm'] = np.empty((7,5,))
        roi_data[roi.gratings_roi_id]['responses_norm'][:] = np.nan

        roi_data[roi.gratings_roi_id]['geno'] = geno
 
    if ("centerGrating" in roi.stim_info['meta']['stim_name']):
        curr_idx = int((roi.stim_info['meta']['epoch_infos']['epoch_2']['diameter_deg']-5)/5)
    elif ("stim-input_gratings-100p-5lum-40s" in roi.stim_info['meta']['stim_name']):
        curr_idx = 6
    else:
        warnings.warn('{d} do not contain relevant stimulus'.format(d=dataset))

    roi_data[roi.gratings_roi_id]['responses'][curr_idx,:] = roi.sorted_power_at_sineFreq
    roi_data[roi.gratings_roi_id]['responses_norm'][curr_idx,:] = roi.sorted_power_at_sineFreq/roi.sorted_power_at_sineFreq.max()
    
    all_rois.append(roi)

    
    
    print('{ds} successfully loaded\n'.format(ds=dataset))

unique_genos = np.unique([data['geno'] for data in roi_data.values()])
#%% Colors
_, colors = pac.run_matplotlib_params()

c_dict = {k:colors[k] for k in colors if k in unique_genos}

grating_sizes ={0:'5',1:'10',2:'15',3:'20',4:'25',5:'30',6:'full'}

cmap = matplotlib.cm.get_cmap('copper')
norm = matplotlib.colors.Normalize(vmin=0, 
                                    vmax=35)
norm_l = matplotlib.colors.Normalize(vmin=0, 
                                    vmax=0.6)

 #%% Single ROI plots

for idx, roi_d in enumerate(roi_data.values()):
    neuron_save_dir = os.path.join(results_save_dir,roi_d['geno'])
    if not os.path.exists(neuron_save_dir):
        os.mkdir(neuron_save_dir)
    

    fig = plt.figure(figsize=(12, 4))
    grid = plt.GridSpec(1, 2, wspace=0.3, hspace=1)
    
    ax1=plt.subplot(grid[0,0])
    ax2=plt.subplot(grid[0,1])

    # Absolute responses
    for epoch in grating_sizes:
        resp = roi_d['responses'][epoch,:]
        ax1.plot(all_rois[0].sorted_luminances,resp,'o-',
            label=grating_sizes[epoch],color=cmap(norm((epoch*5)+5)))
    ax1.set_title('{g}-ROI {i}'.format(g=roi_d['geno'],i=idx))
    ax1.legend()
    ax1.set_ylim([0,np.nanmax(roi_d['responses'])+0.1])
    ax1.set_xlabel('luminances')
    ax1.set_ylabel('response')

    # Normalized responses
    for epoch in grating_sizes:
        resp = roi_d['responses'][epoch,:]
        ax2.plot(all_rois[0].sorted_luminances,resp/resp.max(),'o-',
            label=grating_sizes[epoch],color=cmap(norm((epoch*5)+5)))
    ax2.set_title('{g}-ROI {i} normalized'.format(g=roi_d['geno'],i=idx))
    ax2.legend()
    ax2.set_ylim([0,1.1])
    ax2.set_xlabel('luminances')
    ax2.set_xlabel('normalized response')

    save_n = '{g}_centerGratings_roi_{idx}'.format(g=roi_d['geno'],idx=idx)
    os.chdir(neuron_save_dir)
    fig.savefig('%s.pdf' % save_n, bbox_inches='tight')
    plt.close('all')

    
#%% Single ROI conc traces
for idx, roi in enumerate(all_rois):
    if ("stim-input_gratings-100p-5lum-40s" in roi.stim_info['meta']['stim_name']):
        continue
    geno = roi.experiment_info['Genotype'][:3]
    neuron_save_dir = os.path.join(results_save_dir,geno,'traces')
    if not os.path.exists(neuron_save_dir):
        os.mkdir(neuron_save_dir)
    fig = plt.figure(figsize=(16, 3))

    deg = roi.stim_info['meta']['epoch_infos']['epoch_2']['diameter_deg']

    luminances = roi.epoch_luminances.values()

    plt.plot(np.concatenate(np.array(roi.int_whole_trace.values())[np.argsort(luminances),20:100]),
        color = c_dict[geno],
        label = deg)

    save_name = '{geno}_{roiID}_{deg}deg'.format(geno=geno,roiID=roi.gratings_roi_id,deg=deg)
    plt.title('{geno}_{roiID}_{deg}deg'.format(geno=geno,roiID=roi.gratings_roi_id,deg=deg))
    os.chdir(neuron_save_dir)
    plt.savefig('%s.pdf' % save_name, bbox_inches='tight')
    plt.close('all')

#%% geno summarized
for geno in unique_genos:

    fig = plt.figure(figsize=(12, 4))
    grid = plt.GridSpec(1, 2, wspace=0.3, hspace=1)
    
    ax1=plt.subplot(grid[0,0])
    ax2=plt.subplot(grid[0,1])

    resps = np.array([data['responses'] for data in roi_data.values() if data['geno'] == geno])
    resps_m = np.nanmean(resps,axis=0)
    resps_std = np.nanstd(resps,axis=0)
    
    norm_resps = np.array([data['responses_norm'] for data in roi_data.values() if data['geno'] == geno])
    norm_resps_m = np.nanmean(norm_resps,axis=0)
    norm_resps_std = np.nanstd(norm_resps,axis=0)

    # Absolute responses
    for epoch in grating_sizes:
        resp = resps_m[epoch,:]
        error = resps_std[epoch,:]/np.sqrt(resps.shape[0])
        ax1.errorbar(all_rois[0].sorted_luminances,resp,error,fmt='-o',alpha=.8,
            label=grating_sizes[epoch],color=cmap(norm((epoch*5)+5)))
    ax1.set_title('{g}: {n} ROIs'.format(g=geno,n=resps.shape[0]))
    ax1.legend(loc=4)
    ax1.set_ylim([0,np.nanmax(resps_m)+0.05])
    ax1.set_xlabel('luminances')
    ax1.set_ylabel('response')

    # Normalized responses
    for epoch in grating_sizes:
        resp = norm_resps_m[epoch,:]
        error = norm_resps_std[epoch,:]/np.sqrt(resps.shape[0])
        ax2.errorbar(all_rois[0].sorted_luminances,resp,error,fmt='-o',alpha=.8,
            label=grating_sizes[epoch],color=cmap(norm((epoch*5)+5)))
    ax2.set_title('{g}: {n} ROIs - normalized'.format(g=geno,n=resps.shape[0]))
    ax2.legend(loc=4)
    ax2.set_ylim([0,1.1])
    ax2.set_xlabel('luminances')
    ax2.set_ylabel('response')

    save_n = '{g}_centerGratings'.format(g=geno)
    os.chdir(results_save_dir)
    fig.savefig('%s.pdf' % save_n, bbox_inches='tight',dpi=300)
    plt.close('all')
#%% ANOVA
from scipy.stats import f_oneway
for geno in unique_genos:
    resps = np.array([data['responses'] for data in roi_data.values() if data['geno'] == geno])
    resps_m = np.nanmean(resps,axis=0)
    resps_std = np.nanstd(resps,axis=0)
    
    norm_resps = np.array([data['responses_norm'] for data in roi_data.values() if data['geno'] == geno])
    norm_resps_m = np.nanmean(norm_resps,axis=0)
    norm_resps_std = np.nanstd(norm_resps,axis=0)

    # Absolute responses
    for epoch in grating_sizes:
        resp = resps_m[epoch,:]
        error = resps_std[epoch,:]/np.sqrt(resps.shape[0])
        # print(geno)
        # print(grating_sizes[epoch])
        # print(resp)
        # print(error)
        anova_array = resps[:,epoch,:]
        good_rows = ~np.isnan(np.array(anova_array))[:,0]
        anova_array = anova_array[good_rows,:]
        # #anova
        F, p = f_oneway(anova_array[:,0], anova_array[:,1], anova_array[:,2],anova_array[:,3],anova_array[:,4])

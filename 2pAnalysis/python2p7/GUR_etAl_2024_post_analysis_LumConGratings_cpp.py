#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  25 2022

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
from sklearn import preprocessing
import seaborn as sns


#change to code directory
os.chdir('.../Gur-etal-2024/2pAnalysis/python2p7/common')

import ROI_mod
import post_analysis_core as pac
#%% Directories for loading data and saving figures (ADJUST THEM TO YOUR PATHS)
initialDirectory = 'data_path'
all_data_dir = os.path.join(initialDirectory, 'raw_data')

results_save_dir = os.path.join(initialDirectory, 'Figure2','plots','5cont_5lum')
#%% Luminance values of the experimental setup (ADJUST THE PATHS)
# Values from the stimulation paradigm
measurements_f = '.../Gur-etal-2024/2pAnalysis/luminance_measurements/200622_Investigator_Luminances_LumGainPaper.xlsx'
measurement_df = pd.read_excel(measurements_f,header=0)
res = linregress(measurement_df['file_lum'], measurement_df['measured'])
# %% Load datasets and desired variables
# Raw data folder: "L2_L3_Tm1_Tm9_luminanceContrast"
# Use only for L1, L2, L3, Tm1, Tm9
exp_folder = 'L2_L3_Tm1_Tm9_luminanceContrast' 

data_dir = os.path.join(all_data_dir,exp_folder)
datasets_to_load = os.listdir(data_dir)

                    
properties = ['SNR','Reliab','depth','slope']
combined_df = pd.DataFrame(columns=properties)
all_rois = []
all_traces=[]
tunings = []
baselines = []
baseline_power = []
z_tunings = []
# Initialize variables


for idataset, dataset in enumerate(datasets_to_load):

    # Loading the datasets
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
    except EOFError:
        print('Unable to import skipping - unknown problem: \n{f} '.format(f=load_path))
        continue
    curr_rois = workspace['final_rois']

    # Check if the stimulus is the one we need for analysis
    if (not('1Dir_meanBG_DriftingSine_1Hz_5c_5lum_200s' in curr_rois[0].stim_name)):
        continue

    # Thresholding
    # Reliability thresholding
    curr_rois = ROI_mod.analyze_gratings_general(curr_rois)
    # 
    curr_rois = ROI_mod.threshold_ROIs(curr_rois, {'reliability':0.6})


    if len(curr_rois) <= 1:
        warnings.warn('{d} contains 1 or no ROIs, skipping'.format(d=dataset))
        continue
    
    
    if curr_rois[0].experiment_info['Genotype'] == 'R64G09-Recomb-Homo':
        geno = 'T45'
    else:
        geno = curr_rois[0].experiment_info['Genotype'][:3]

    for roi in curr_rois:
        roi_lums = np.array(roi.luminances)
        candelas = res.intercept + res.slope*roi_lums
        luminances_photon = pac.convert_cd_to_photons(cdVal=candelas,wavelength=475)
        diff_luminances = luminances_photon
        roi.luminances = luminances_photon

    all_rois.append(curr_rois)
    data_to_extract = ['SNR','reliability','category']
    roi_data = ROI_mod.data_to_list(curr_rois, data_to_extract)
    
    print(geno)
    print(curr_rois[0].stim_name)
    
    df_c = {}
    df_c['depth'] = list(map(lambda roi : roi.imaging_info['depth'], curr_rois))
    # df_c['CS'] = roi_data['CS']
    df_c['SNR'] = roi_data['SNR']
    df_c['category'] = roi_data['category']
    df_c['reliability'] = roi_data['reliability']
    df_c['flyID'] = np.tile(curr_rois[0].experiment_info['FlyID'],len(curr_rois))
    df_c['Geno'] = np.tile(geno,len(curr_rois))
    df_c['uniq_id'] = np.array(map(lambda roi : roi.uniq_id, curr_rois)) 
    df = pd.DataFrame.from_dict(df_c) 
    rois_df = pd.DataFrame.from_dict(df)
    combined_df = combined_df.append(rois_df, ignore_index=True, sort=False)
    print('{ds} successfully loaded\n'.format(ds=dataset))

le = preprocessing.LabelEncoder()
le.fit(combined_df['flyID'])
combined_df['flyIDNum'] = le.transform(combined_df['flyID'])

all_rois=np.concatenate(all_rois)
#%%
_, colors = pac.run_matplotlib_params()

c_dict = {k:colors[k] for k in colors if k in combined_df['Geno'].unique()}

#%% Summary figure per genotype

for geno in np.unique(combined_df['Geno']):
    geno_color = c_dict[geno]
    if geno == "T45":
        cat = "Layer B"
        curr_neuron_mask = ((combined_df['category'] == cat)  & (combined_df['Geno']==geno))
        fig_addition = cat
    else:
        curr_neuron_mask = (combined_df['Geno']==geno)
        fig_addition = ''

    curr_rois = all_rois[curr_neuron_mask]
    plt.close('all')

    fig = plt.figure(figsize=(4, 4))
    grid = plt.GridSpec(1, 1, wspace=0.3, hspace=1)
    ax1=plt.subplot(grid[0,0])

    all_powers = []
    all_luminances = []
    all_contrasts = []
    all_fly_means = []

    
    for contrast in np.unique(curr_rois[0].contrasts):
        curr_tunings = []

        for roi in curr_rois:
            luminances = np.array(roi.luminances)[contrast==np.array(roi.contrasts)]
            powers = np.array(roi.power_at_hz/roi.power_at_hz.max())[contrast==np.array(roi.contrasts)]
            # powers = np.array(roi.power_at_hz)[contrast==np.array(roi.contrasts)]
            curr_tunings.append(list(powers[np.argsort(luminances)]))

        # Luminances
        diff_luminances = np.sort(luminances)
      
    
        a=pac.compute_over_samples_groups(data = curr_tunings, 
                                    group_ids= np.array(combined_df[curr_neuron_mask]['flyIDNum']), 
                                    error ='SEM',
                                    experiment_ids = np.array(combined_df[curr_neuron_mask]['Geno']))

        cmap = matplotlib.cm.get_cmap('gist_gray')
        norm = matplotlib.colors.Normalize(vmin=0, 
                                        vmax=1.2)

        label = 'c: {c}'.format(c = contrast)
        curr_fly_means = np.array(a['experiment_ids'][geno]['over_samples_means'])
        all_mean_data = a['experiment_ids'][geno]['over_groups_mean']
        all_yerr = a['experiment_ids'][geno]['over_groups_error']
        ax1.errorbar(diff_luminances,all_mean_data,all_yerr,
                    fmt='-s',alpha=1,color=cmap(norm(contrast)),label=label)
        
        all_powers = np.concatenate((all_powers,all_mean_data))
        if len(all_fly_means) == 0:
                all_fly_means = curr_fly_means[:,:, np.newaxis]
        else:
            all_fly_means = np.concatenate((all_fly_means,curr_fly_means[:,:, np.newaxis]), axis=2)
        all_luminances = np.concatenate((all_luminances,diff_luminances))
        all_contrasts = np.concatenate((all_contrasts,np.repeat(contrast,5)))
    # Saving figure
    title = '{g} n: {f}({ROI})'.format(g=geno,
                                       f=len(a['experiment_ids'][geno]['over_samples_means']),
                                       ROI=len(a['experiment_ids'][geno]['all_samples']))
    ax1.set_ylim((0,ax1.get_ylim()[1]))
    ax1.set_title(title)
    ax1.legend()
    ax1.set_xlabel("luminance")
    ax1.set_xscale('log')
    ax1.set_xlim((10000,ax1.get_xlim()[1]))

    save_name = '_Norm_Sine_1Hz_5lum_5c_{geno}_{figadd}_AT' .format(geno=geno,figadd=fig_addition)
    os.chdir(results_save_dir)
    plt.savefig('%s.pdf' % save_name, bbox_inches='tight',dpi=300)

    # Heatmap figure
    heatmap_dict = {}
    heatmap_dict['Luminance'] = all_luminances
    heatmap_dict['Contrast'] = all_contrasts
    heatmap_dict['Response'] = all_powers
    
    from scipy.ndimage.filters import gaussian_filter

    import scipy.ndimage

    df_heatmap = pd.DataFrame.from_dict(heatmap_dict)
    heatmap_img = np.array(df_heatmap["Response"]).reshape(5,5)
    

    cl_map = df_heatmap.pivot(index='Contrast',columns='Luminance')
    title = '{g} n: {f}({ROI})'.format(g=geno,
                                       f=len(a['experiment_ids'][geno]['over_samples_means']),
                                       ROI=len(a['experiment_ids'][geno]['all_samples']))

    fig = plt.figure(figsize = (5,5))
        
    ax=sns.heatmap(cl_map, cmap='coolwarm',center=0,
                    xticklabels=np.array(cl_map.columns.levels[1]).astype(float),
                    yticklabels=np.array(cl_map.index),
                    cbar_kws={'label': 'Response'})
    ax.invert_yaxis()


    plt.title('CL map')
    plt.xlabel('Luminance')
    plt.ylabel('Contrast')

    save_name = 'Summary_Norm_CL_{geno}_{figadd}_AT' .format(geno=geno,figadd=fig_addition)
    os.chdir(results_save_dir)
    fig.savefig('%s.pdf' % save_name, bbox_inches='tight',dpi=300)

    # Image
    fig = plt.figure(figsize = (5,5))
    lc_image = gaussian_filter(scipy.ndimage.zoom(heatmap_img, 5, order=1),sigma=10)
    lc_range = lc_image.max() - lc_image.min()
    plt.imshow(lc_image,cmap="magma",vmin=0)
    plt.colorbar().set_label("Power")
    contour_levels = np.linspace(lc_image.min()+lc_range/10,lc_image.max()+lc_range/10,9)
    plt.contour(lc_image,levels=contour_levels, cmap='Greys')
    plt.colorbar().set_label("Contour height")
    plt.title('CL map')
    plt.xlabel('Luminance')
    plt.ylabel('Contrast')
    
    ax = plt.gca()

    ax.invert_yaxis()
    save_name = 'Summary_Norm_CL_Image_{geno}_{figadd}_AT' .format(geno=geno,figadd=fig_addition)
    os.chdir(results_save_dir)
    fig.savefig('%s.pdf' % save_name, bbox_inches='tight')

    data_to_save = {'all_fly_means': all_fly_means, 
                'luminances': diff_luminances,
                'contrast': np.unique(curr_rois[0].contrasts),
                'genotype': geno}


    # Save for further analysis (ANOVA etc.)
    # Specify the file path where you want to save the pickle file
    file_path = os.path.join(results_save_dir, f"{geno}_AT_ConLum.pickle") # change f string to python 2.7 format if necessary

    # Open the file in binary write mode
    with open(file_path, 'wb') as file:
        cPickle.dump(data_to_save, file)
#%% Summary figure per genotype contrast responses

for geno in np.unique(combined_df['Geno']):
    geno_color = c_dict[geno]
    curr_neuron_mask = (combined_df['Geno']==geno)
    curr_rois = all_rois[curr_neuron_mask]
    plt.close('all')

    fig = plt.figure(figsize=(4, 4))
    grid = plt.GridSpec(1, 1, wspace=0.3, hspace=1)
    ax1=plt.subplot(grid[0,0])
    for lum in np.unique(curr_rois[0].luminances):
        # lum = int(lum)
        curr_tunings = []

        for roi in curr_rois:
            contrasts = np.array(roi.contrasts)[lum==np.array(curr_rois[0].luminances)]
            powers = np.array(roi.power_at_hz)[lum==np.array(curr_rois[0].luminances)]
            curr_tunings.append(list(powers[np.argsort(contrasts)]))

        diff_contrasts = np.sort(contrasts)
    
        a=pac.compute_over_samples_groups(data = curr_tunings, 
                                    group_ids= np.array(combined_df[curr_neuron_mask]['flyIDNum']), 
                                    error ='SEM',
                                    experiment_ids = np.array(combined_df[curr_neuron_mask]['Geno']))

        cmap = matplotlib.cm.get_cmap('gist_gray')
        norm = matplotlib.colors.Normalize(vmin=0, 
                                        vmax=curr_rois[0].luminances.max()*1.5)

        label = 'l: {c}'.format(c = int(lum))
        all_mean_data = a['experiment_ids'][geno]['over_groups_mean']
        all_yerr = a['experiment_ids'][geno]['over_groups_error']
        ax1.errorbar(diff_contrasts,all_mean_data,all_yerr,
                    fmt='-s',alpha=1,color=cmap(norm(lum)),label=label)
        
    # Saving figure
    title = '{g} n: {f}({ROI})'.format(g=geno,
                                       f=len(a['experiment_ids'][geno]['over_samples_means']),
                                       ROI=len(a['experiment_ids'][geno]['all_samples']))
    ax1.set_ylim((0,ax1.get_ylim()[1]))
    ax1.set_xlim((0,1.2))
    ax1.set_title(title)
    ax1.legend()
    ax1.set_xlabel("contrast")

    save_name = '_Sine_1Hz_5lum_5c_{geno}_contrast'.format(geno=geno)
    os.chdir(results_save_dir)
    plt.savefig('%s.png' % save_name, bbox_inches='tight',dpi=300)
# %%

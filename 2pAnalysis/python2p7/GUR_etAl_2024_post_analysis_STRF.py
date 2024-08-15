#%% Following is used for analyzing the spatio-temporal receptive fields via cross correlation after white noise stimulation
# Analysis of glutamate signals coming to Tm1, Tm9 dendrites and calcium signals at the axon terminals

#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 19 13:39:02 2020

@author: burakgur
"""
#%%
import cPickle
import os
import numpy as np
import matplotlib.pyplot as plt
import warnings
import post_analysis_core as pac
import ROI_mod
import seaborn as sns

#change to code directory
os.chdir('/Volumes/Backup Plus/Post-Doc/_SiliesLab/Manuscripts/2023_Lum_Gain/Data_code/code/python2p7/common')

#%%
#%% Directories for loading data and saving figures (ADJUST THEM TO YOUR PATHS)
initialDirectory = '/Volumes/Backup Plus/Post-Doc/_SiliesLab/Manuscripts/2023_Lum_Gain/Data_code'
# initialDirectory = '/Volumes/HD-SP1/Burak_data/Python_data'
all_data_dir = os.path.join(initialDirectory, 'raw_data')
results_save_dir = os.path.join(initialDirectory,'FigureS4/plots')


# %% Load datasets and desired variables
# exp_folder = 'Tm1iGluSnFR' # Change thresholds below, according to manuscript methods part Threshold 0.003
exp_folder = 'Tm9iGluSnFR' # Change thresholds below, according to manuscript methods part Threshold 0.005
# exp_folder = 'L2-L3-TmX-edges-1Hzsine-11steps' # Change thresholds below, according to manuscript methods part Threshold 0.005



data_dir = os.path.join(all_data_dir,exp_folder)
datasets_to_load = os.listdir(data_dir)
# Initialize variables
final_rois_all =np.array([])
flyIDs = []
flash_resps = []
genotypes = []

for idataset, dataset in enumerate(datasets_to_load):
    if not(dataset.split('.')[-1] =='pickle'):
        warnings.warn('Skipping non pickle file: {f}\n'.format(f=dataset))
        continue
    load_path = os.path.join(data_dir, dataset)
    load_path = open(load_path, 'rb')
    workspace = cPickle.load(load_path)
    curr_rois = workspace['final_rois']
    
    if (not('El_50ms' in curr_rois[0].stim_name)):
        continue
    # if not(curr_rois[0].experiment_info['Genotype'][:2] == 'L2'):
        # continue
    print(curr_rois[0].experiment_info['Genotype'])
    print(curr_rois[0].stim_name)
    
    for roi in curr_rois: # Only for Tm1iGluSnFR experiments
        roi.sta = np.squeeze(roi.sta)
    final_rois_all = np.append(final_rois_all,np.array(curr_rois))
    geno = curr_rois[0].experiment_info['Genotype']
   
    for roi in curr_rois:
        flyIDs.append(roi.experiment_info['FlyID'])
        genotypes.append(geno)
    print('{ds} successfully loaded\n'.format(ds=dataset))

genotypes = np.array(genotypes)  


    
        
    
    
 #%% Plot all STRFs
threshold = 0.008
for idx,genotype in enumerate(np.unique(genotypes)):
    mask = genotypes == genotype
    mask = genotypes == genotype
    curr_rois = final_rois_all[mask]
    curr_filters = np.array(map(lambda roi : \
                                       roi.sta.T[np.where(roi.sta.T==roi.sta.max())[0]],
                     curr_rois))
        
    filter_masks =np.abs(curr_filters).max(axis=2)>threshold
    
    curr_fly_ids = np.array(flyIDs)[mask]
    curr_fly_filtered = curr_fly_ids[np.array( filter_masks[:,0])]
    curr_rois_filtered = curr_rois[np.array( filter_masks[:,0])]
    
    fig1= ROI_mod.plot_STRFs(curr_rois_filtered, f_w=10,
                             number=len(curr_rois_filtered),cmap='coolwarm')
    fig1.suptitle(genotype)
    f1_n = 'All_STRFs_{t}_{g}'.format(t=threshold,g=genotype)
    os.chdir(results_save_dir)
    fig1.savefig('%s.png'% f1_n, bbox_inches='tight',
                transparent=False,dpi=300)
    
#%% Avg STRFs OFF NEURONS

threshold = 0.005 # Change threshold here
_, colors_d = pac.run_matplotlib_params()
colors = [colors_d['green2'],colors_d['magenta'],colors_d['green1']]
for genotype in np.unique(genotypes):
    plt.close('all')
    fig1 = plt.figure(figsize=(14, 3))
    grid = plt.GridSpec(1, 4, wspace=0.4, hspace=0.6)
    
    ax = plt.subplot(grid[0,0])
    mask = genotypes == genotype
    curr_rois = final_rois_all[mask]
    curr_filters = np.array(map(lambda roi : \
                                       roi.sta.T[np.where(roi.sta.T==roi.sta.min ())[0]],
                     curr_rois))
        
    filter_masks =np.abs(curr_filters).max(axis=2)>threshold

    # Below for Tm1iGluSnFR
    # filter_masks =(np.abs(curr_filters).max(axis=2)/np.mean(curr_filters,axis=2))>10
    # filter_masks2 =np.abs(curr_filters).max(axis=2)>threshold
    # filter_masks = filter_masks *filter_masks2
    
    curr_fly_ids = np.array(flyIDs)[mask]
    curr_fly_filtered = curr_fly_ids[np.array( filter_masks[:,0])]
    curr_rois_filtered = curr_rois[np.array( filter_masks[:,0])]
    curr_filters_filtered = curr_filters[filter_masks]
    
    # Change here if cell is OFF or ON (change the roi.sta.min or max)
    mean_strf = np.mean(np.array(map(lambda roi : \
                 np.roll(roi.sta.T, roi.sta.T.shape[0]/2- \
                         int(np.where(roi.sta.T==roi.sta.min())[0]), axis=0),
                     curr_rois_filtered)),axis=0)
        
    sns.heatmap(mean_strf, cmap='coolwarm', ax=ax,center=0,cbar=True)
    ax.axis('off')
    ax.set_title(genotype,fontsize='xx-small') 
    
    ax2 = plt.subplot(grid[0,1])
      
    mean = np.mean(curr_filters_filtered,axis=0).T
    # mean = (mean-mean.min())/(mean.max() - mean.min())
    # mean = (mean)/(mean.max())
    error = np.std(curr_filters_filtered,axis=0).T / np.sqrt(curr_filters_filtered.shape[0])
    
    label = "{l}, {f} {ROI}".format(l=genotype,
                                    f=len(np.unique(curr_fly_filtered)),
                                    ROI = curr_filters_filtered.shape[0])
    
    ub = mean + error
    lb = mean - error
    t_trace = np.linspace(-len(mean),0,len(mean))*50/1000
    ax2.plot(t_trace,mean,alpha=.8,lw=3,color='k',
             label=label)
    ax2.fill_between(t_trace, ub[:], lb[:], alpha=.4,color='k')
    ax2.legend()
    ax2.set_xlabel('Time(s)')
    
    max_idx = np.argmax(np.abs(mean)) 
    ax3 = plt.subplot(grid[0,2])
    
    mean = mean_strf[:,max_idx]
    # mean = (mean-mean.min())/(mean.max() - mean.min())
    # mean = (mean)/(mean.max())
    # error = np.std(np.array(map(lambda roi : \
    #              np.roll(roi.sta.T, roi.sta.T.shape[0]/2- \
    #                      int(np.where(roi.sta.T==roi.sta.max())[0]), axis=0),
    #                  curr_rois_filtered)),axis=0)[:,-1]
        
    label = "{l}, {f} {ROI}".format(l=genotype,
                                    f=len(np.unique(curr_fly_filtered)),
                                    ROI = curr_filters_filtered.shape[0])
    
    # ub = mean + error
    # lb = mean - error
    t_trace = np.linspace(-30,30,len(mean))
    ax3.plot(t_trace,mean,alpha=.8,lw=3,color='r',
             label=label)
    
    ax3.legend()
    ax3.set_xlim([-30,30])
    ax3.set_xlabel('Space(degrees)')
    ax3.set_title('Spatial filter at max')


    ax4 = plt.subplot(grid[0,3])
    screen_w = 60

    

    all_filters = np.array(map(lambda roi : \
                 np.roll(roi.sta.T, roi.sta.T.shape[0]/2- \
                         int(np.where(roi.sta.T==roi.sta.min())[0]), axis=0),
                     curr_rois_filtered))
    fwhm = []
    rsq = []
    fit_traces = []
    coeffs = []
    for filter in all_filters:
        space_max_i = int(np.where(filter==filter.min())[1])
        curr_filter = filter[:,space_max_i]
        # curr_filter = curr_filter *-1
        # curr_filter = curr_filter - curr_filter.min()
        screen_coords = np.linspace(0, screen_w, num=screen_w*2, endpoint=True)

        filter_t_v = np.linspace(0, screen_w, num=len(curr_filter), endpoint=True)
        
        filter_i = np.interp(screen_coords, filter_t_v, 
                                  curr_filter)
        
        try:
            fit_trace, r_squared, coeff = ROI_mod.fit_1d_gauss(screen_coords, np.abs(filter_i))
        except RuntimeError:
            
            continue

        
        
        coeffs.append(coeff)
        fwhm.append(2.355 * coeff[2]) 
        rsq.append(r_squared)
        fit_traces.append(fit_trace)
    rsq = np.array(rsq)
    fwhm = np.array(fwhm)
    #fwhm = fwhm[fwhm>0]
    pac.bar_bg(np.abs(fwhm[rsq>0.6]), 1, color='k', scat_s =7,ax=ax4, yerr=None, 
            errtype='SEM',alpha = .6,width=0.8,label=geno)
    ax4.set_ylim([0,25])
    f1_n = 'Mean_STRFs_together_{t}threshold_{g}'.format(t=threshold,g=genotype)
    os.chdir(results_save_dir)
    fig1.savefig('%s.pdf'% f1_n, bbox_inches='tight',
                transparent=False,dpi=300)

#%% Following analyzes and saves the data from two photon experiments done in Göttingen in the Investigator microscope
# This code is used for  analyzing the responses to edges with 100% contrast and changing background luminance
# The stimuli were presented with the software based on c++ 
# Analysis of only T4 T5 GCaMP6f recordings
"""
Created on Tue Mar  3 13:16:23 2020

@author: burakgur
"""
#%%
import cPickle
import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import os
from scipy.stats import linregress

#change to code directory
os.chdir('/Volumes/Backup Plus/Post-Doc/_SiliesLab/Manuscripts/2023_Lum_Gain/Data_code/code/python2p7/common')

import ROI_mod
import post_analysis_core as pac

#%% Directories for loading data and saving figures (ADJUST THEM TO YOUR PATHS)
initialDirectory = '/Volumes/Backup Plus/Post-Doc/_SiliesLab/Manuscripts/2023_Lum_Gain/Data_code'
# initialDirectory = '/Volumes/HD-SP1/Burak_data/Python_data'
all_data_dir = os.path.join(initialDirectory, 'raw_data')
allData = os.path.join(all_data_dir, 'T4_T5_luminanceGain',
                             'luminance_edges') # Raw data folder: "T4_T5_luminanceGain/luminance_edges"

#%%
# Plotting parameters
colors, _ = pac.run_matplotlib_params()
color = colors[5]
color_pair = colors[2:4]
roi_plots=False

#%% Luminances
# Values from the stimulation paradigm
measurements_f = '/Users/burakgur/Documents/GitHub/python_lab/2p_calcium_imaging/200622_Investigator_Luminances_LumGainPaper.xlsx'
measurement_df = pd.read_excel(measurements_f,header=0)
res = linregress(measurement_df['file_lum'], measurement_df['measured'])
# %% Load datasets and desired variables
exp_t = 'lumedge'
datasets_to_load = os.listdir(allData)

                    
properties = ['BF', 'PD', 'SNR','Reliab','depth','DSI','slope']
combined_df = pd.DataFrame(columns=properties)
all_rois = []
all_traces=[]
tunings = []
# Initialize variables
flyNum = 0
for dataset in datasets_to_load:
    if not(".pickle" in dataset):
        print('Skipping non pickle file: {d}'.format(d=dataset))
        continue
    load_path = os.path.join(allData, dataset)
    load_path = open(load_path, 'rb')
    workspace = cPickle.load(load_path)
    curr_rois = workspace['final_rois']
    
    if not('Dark_BG_2DOFFedge_20dps_X_to_0__X1to0p03_6vals_84s' in curr_rois[0].stim_name):
        
        continue
    curr_stim_type = 'OFF'
    # Reliability thresholding
    curr_rois = ROI_mod.analyze_luminance_edges(curr_rois)
    curr_rois = ROI_mod.threshold_ROIs(curr_rois, {'reliability':0.4})
    
    
        
    geno = curr_rois[0].experiment_info['Genotype']
    if 'L3Sil_R64G09' in geno:
        geno = 'L3_Kir'
    elif 'Contr' in geno:
        geno = 'T5_Control'
    elif curr_rois[0].experiment_info['Genotype'] == 'R64G09-Recomb-Homo':
        geno = 'T45'
    elif (curr_rois[0].experiment_info['Genotype'][:2] == "L2") or (curr_rois[0].experiment_info['Genotype'][:2] == "L3") or (curr_rois[0].experiment_info['Genotype'][:2] == "L1"):
        continue
    print(curr_rois[0].experiment_info['Genotype'])
    print(curr_rois[0].stim_name)
    
    for roi in curr_rois:
        roi_lums = roi.luminances
        candelas = res.intercept + res.slope*roi_lums
        luminances_photon = pac.convert_cd_to_photons(cdVal=candelas,wavelength=475)
        diff_luminances = luminances_photon


        curr_tunings = roi.edge_resps/roi.edge_resps.max()
        roi.slope = linregress(diff_luminances, np.transpose(curr_tunings))[0]

    all_rois.append(curr_rois)
    data_to_extract = ['SNR','reliability','slope','DSI','category']
    roi_data = ROI_mod.data_to_list(curr_rois, data_to_extract)
    
    
    tunings.append(np.squeeze\
                   (list(map(lambda roi: roi.edge_resps,curr_rois))))
    
    all_traces.append(np.array\
                      (map(lambda roi : roi.max_aligned_traces[:,:49],
                                   curr_rois)))
    
    
    depths = list(map(lambda roi : roi.imaging_info['depth'], curr_rois))

    df_c = {}
    df_c['depth'] = depths
    if "RF_map_norm" in curr_rois[0].__dict__.keys():
        df_c['RF_map_center'] = list(map(lambda roi : (roi.RF_map_norm>0.95).astype(int)
                                             , curr_rois))
        df_c['RF_map_bool'] = np.tile(True,len(curr_rois))
        screen = np.zeros(np.shape(curr_rois[0].RF_map))
        screen[np.isnan(curr_rois[0].RF_map)] = -0.1
        
        
        for roi in curr_rois:
            curr_map = (roi.RF_map_norm>0.95).astype(int)
    
            x1,x2 = ndimage.measurements.center_of_mass(curr_map)
            s1,s2 = ndimage.measurements.center_of_mass(np.ones(shape=screen.shape))
            roi.distance_to_center = np.sqrt(np.square(x1-s1) + np.square(x2-s2))
        df_c['RF_distance_to_center'] = list(map(lambda roi : roi.distance_to_center, 
                                             curr_rois))
        print('RFs found')
    else:
        df_c['RF_map_center'] = np.tile(None,len(curr_rois))
        df_c['RF_map_bool'] = np.tile(False,len(curr_rois))
        df_c['RF_distance_to_center'] = np.tile(np.nan,len(curr_rois))
            
    df_c['slope'] = roi_data['slope']
    df_c['SNR'] = roi_data['SNR']
    df_c['Reliab'] = roi_data['reliability']
    df_c['DSI'] = roi_data['DSI']
    df_c['category'] = roi_data['category']
    df_c['Geno'] = np.tile(geno,len(curr_rois))
    df_c['flyID'] = np.tile(curr_rois[0].experiment_info['FlyID'],len(curr_rois))
    df_c['flyNum'] = np.tile(flyNum,len(curr_rois))
    flyNum = flyNum +1
    df = pd.DataFrame.from_dict(df_c) 
    rois_df = pd.DataFrame.from_dict(df)
    combined_df = combined_df.append(rois_df, ignore_index=True, sort=False)
    print('{ds} successfully loaded\n'.format(ds=dataset))

all_traces = np.concatenate(all_traces)
tunings = np.concatenate(tunings)
all_rois=np.concatenate(all_rois)
#%%
_, colors = pac.run_matplotlib_params()

c_dict = {k:colors[k] for k in colors if k in combined_df['Geno'].unique()}


# %%  Summary

plt.close('all')

for idx, geno in enumerate(np.unique(combined_df['Geno'])):
    if geno == "T45":
        cat = "Layer A"
        cat2 = "Layer B"
        curr_neuron_mask = (((combined_df['category'] == cat) | (combined_df['category'] == cat2))   & (combined_df['Geno']==geno))
        # curr_neuron_mask = ((combined_df['category'] == cat)  & (combined_df['Geno']==geno))

        fig_addition = cat + cat2
        # fig_addition = cat
    else:
        curr_neuron_mask = (combined_df['Geno']==geno)
        fig_addition = ''
    geno_color = c_dict[geno]
    # neuron_save_dir = os.path.join(results_save_dir,geno,'luminance_edges')
   
            
    curr_df = combined_df[curr_neuron_mask]
    curr_rois = all_rois[curr_neuron_mask]
    fig = plt.figure(figsize=(16, 4))
    grid = plt.GridSpec(1, 4, wspace=0.3, hspace=0.6)
    
    ax1=plt.subplot(grid[0,0])
    ax2=plt.subplot(grid[0,1])
    ax3 = plt.subplot(grid[0,2])
    ax4 = plt.subplot(grid[0,3])
    if 'ON' in all_rois[curr_neuron_mask][0].stim_name:
        curr_stim_type ='ON'
    elif 'OFF' in all_rois[curr_neuron_mask][0].stim_name:
        curr_stim_type ='OFF'
        
     
    cmap = mpl.cm.get_cmap('inferno')
    norm = mpl.colors.Normalize(vmin=0, 
                                       vmax=np.max(diff_luminances))
    
    sensitivities = tunings[curr_neuron_mask]
    properties = ['Luminance', 'Response']
    senst_df = pd.DataFrame(columns=properties)
    colors_lum = []
    for idx_lum, luminance in enumerate(diff_luminances):
        curr_color = cmap(norm(luminance))
        colors_lum.append(curr_color)
        curr_traces = all_traces[curr_neuron_mask,idx_lum,:]
        
        a=pac.compute_over_samples_groups(data = curr_traces, 
                                group_ids= np.array(combined_df[curr_neuron_mask]['flyNum']), 
                                error ='SEM',
                                experiment_ids = np.array(combined_df[curr_neuron_mask]['Geno']))

        label = '{g} n: {f}({ROI})'.format(g=geno,
                                           f=len(a['experiment_ids'][geno]['over_samples_means']),
                                           ROI=len(a['experiment_ids'][geno]['all_samples']))
        
        
        mean_r = a['experiment_ids'][geno]['over_groups_mean'][:]
        err = a['experiment_ids'][geno]['over_groups_error'][:]
        
        ub = mean_r + err
        lb = mean_r - err
        x = np.linspace(0,len(mean_r),len(mean_r))
        ax1.plot(x,mean_r,'-',lw=2,color=cmap(norm(luminance)),alpha=.7,
                 label=luminance)
        ax1.fill_between(x, ub, lb,color=cmap(norm(luminance)), alpha=.3)
        ax1.legend()
        
        curr_sensitivities=sensitivities[:,idx]
        curr_luminances = np.ones(curr_sensitivities.shape) * luminance
        df = pd.DataFrame.from_dict({'Luminance':curr_luminances,
                                     'Response':curr_sensitivities}) 
        rois_df = pd.DataFrame.from_dict(df)
        senst_df = senst_df.append(rois_df, ignore_index=True, sort=False)
    ax1.set_ylabel('$\Delta F/F$')
    ax1.set_title('Aligned mean responses {l}'.format(l=label) )   
    
    
    tuning_curves = tunings[curr_neuron_mask]
    # tuning_curves = tuning_curves/tuning_curves.max(axis=1).reshape(len(tuning_curves),1)
    a=pac.compute_over_samples_groups(data = tuning_curves, 
                                group_ids= np.array(combined_df[curr_neuron_mask]['flyNum']), 
                                error ='SEM',
                                experiment_ids = np.array(combined_df[curr_neuron_mask]['Geno']))

    label = '{g} n: {f}({ROI})'.format(g=geno,
                                       f=len(a['experiment_ids'][geno]['over_samples_means']),
                                       ROI=len(a['experiment_ids'][geno]['all_samples']))
    all_mean_data = a['experiment_ids'][geno]['over_groups_mean']
    all_yerr = a['experiment_ids'][geno]['over_groups_error']
    ax2.errorbar(diff_luminances,all_mean_data,all_yerr,
                 fmt='-s',alpha=1,color=geno_color,label=label)
    ax2.set_ylim((0,ax2.get_ylim()[1]))
    # ax2.set_ylim((0,1))
    ax2.set_xscale('log')
    ax2.set_xlim((10000,ax2.get_xlim()[1]))
    ax2.set_title('Step responses') 

    stim_frames = all_rois[curr_neuron_mask][-1].stim_info['output_data'][:,7]  # Frame information
    stim_vals =  all_rois[curr_neuron_mask][-1].stim_info['output_data'][:,3] # Stimulus value
    uniq_frame_id = np.unique(stim_frames,return_index=True)[1]
    stim_vals = stim_vals[uniq_frame_id]

    
    a=pac.compute_over_samples_groups(data = curr_df['slope'], 
                                group_ids= np.array(combined_df[curr_neuron_mask]['flyNum']), 
                                error ='SEM',
                                experiment_ids = np.array(combined_df[curr_neuron_mask]['Geno']))
    all_fly_data = np.array(a['experiment_ids'][geno]['over_samples_means'])
    all_yerr = a['experiment_ids'][geno]['over_groups_error']
    
    pac.bar_bg(all_fly_data, 1, color=c_dict[geno], scat_s =7,ax=ax3, yerr=None, 
            errtype='SEM',alpha = .6,width=0.8,label=geno)
    ax3.set_ylim([0,0.000004])
    ax3.legend()


    ax3.set_title('Luminance sensitivity')   
    ax3.set_xlabel('Slope')
    # Saving figure
    save_name = '_Edge_summary_100p_{st}_{geno}_{addition}'.format(
                                                   st=curr_stim_type,
                                                   geno=geno,
                                                   addition = fig_addition)
    os.chdir(summary_save_dir)
    plt.savefig('%s.pdf' % save_name, bbox_inches='tight',dpi=300)
    #%% Save data
    output_dir = os.path.join(initialDirectory, 'Figure1','processed_data')
    data_dict = {"tunings": tuning_curves,
                "genotypes":curr_df['Geno'],
                "experimentID":curr_df['flyNum'],
                "luminances": luminances_photon,
                "df": curr_df,
                "rois":curr_rois,
                "slopes": curr_df['slope']
                }


    save_name = '{s}-edges-real_lum_vals_rel0p4'.format(s= geno, )
    saveVar = open(os.path.join(output_dir, "{s}.pickle".format(s=save_name)), "wb")
    cPickle.dump(data_dict, saveVar, protocol=-1)
    saveVar.close()
    
    
        

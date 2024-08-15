#%%
import cPickle
import pandas as pd
import numpy as np
import warnings
import matplotlib.pyplot as plt
import os
from scipy.stats import linregress

os.chdir('/Users/burakgur/Documents/GitHub/python_lab/2p_calcium_imaging')

import PyROI
import post_analysis_core as pac

#%% Directories for loading data and saving figures (ADJUST THEM TO YOUR PATHS)
initialDirectory = '/Volumes/Backup Plus/Post-Doc/_SiliesLab/Manuscripts/2023_Lum_Gain/Data_code'
# initialDirectory = '/Volumes/HD-SP1/Burak_data/Python_data'
all_data_dir = os.path.join(initialDirectory, 'raw_data')
results_save_dir = os.path.join(initialDirectory,'Figure4/plots')

#%% Luminance values of the experimental setup (ADJUST THE PATHS)
# Values from the stimulation paradigm
measurements_f = '/Volumes/Backup Plus/Post-Doc/_SiliesLab/Manuscripts/2023_Lum_Gain/Data_code/code/luminance_measurements/210716_Ultima_Luminances_ONOFFpaper.xlsx'
measurement_df = pd.read_excel(measurements_f,header=0)
res = linregress(measurement_df['file_lum'], measurement_df['measured'])
#%%
exp_folder = 'SpatialPooling_Tm1_Tm9/annulusStim'
data_dir = os.path.join(all_data_dir,exp_folder)
datasets_to_load = os.listdir(data_dir)

                    
properties = ['Reliab','depth']
combined_df = pd.DataFrame(columns=properties)
all_rois = []
all_traces=[]
tunings = []
baselines = []
baseline_power = []
z_tunings = []


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
    except EOFError: 
        print('Unable to import skipping: \n{f} '.format(f=load_path))
        continue
    curr_rois = workspace['final_rois']
    
    
    if not('stim-input_circCentGrat_5degGrating_3s_circ4Lum9Size_staticGratingBG__30degLambda' in curr_rois[0].stim_info['meta']['stim_name']) :
        continue
    
    # Thresholding
    # Reliability thresholding
    curr_rois = PyROI.calculateReliability(curr_rois)
    # max power threshold is 0.015 for Tm1 iGluSnfr
    curr_rois = PyROI.threshold_ROIs(curr_rois, {'reliability':0.5})

    if not(curr_rois):
        continue

    geno = curr_rois[0].experiment_info['Genotype']
    print(geno)
    print(curr_rois[0].stim_info['meta']['stim_name'])

    curr_rois = PyROI.analyzeCenterGratwithBGcircle(curr_rois)

    all_rois.append(curr_rois)
    
    data_to_extract = ['reliability','category']
    roi_data = PyROI.data_to_list(curr_rois, data_to_extract)

    df_c = {}
    df_c['depth'] = list(map(lambda roi : roi.imaging_info['depth'], curr_rois))
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

luminances = np.array(all_rois[0].BGcircle_luminance.values())
candelas = res.intercept + res.slope*luminances
luminances_photon = pac.convert_cd_to_photons(cdVal=candelas,wavelength=475)
real_lums = luminances_photon

# %% 1 HZ Power analysis
curr_geno = "Tm1_GC6f"
# Tm9Rec_lexAopGC6f Tm1_GC6f
geno_rois = all_rois[(combined_df['Geno'] ==curr_geno)]
luminances = np.array(all_rois[0].BGcircle_luminance.values())
diameters = np.array(all_rois[0].BGcircle_diameter.values())


all_powers = np.zeros(shape=(len(np.unique(luminances)), len(np.unique(diameters)), len(geno_rois) ))

for roi_id, roi in enumerate(geno_rois):
    luminances = np.array(roi.BGcircle_luminance.values())
    diameters = np.array(roi.BGcircle_diameter.values())
    powers = np.array(roi.power_at_sineFreq.values())
    for idx, luminance in enumerate(np.unique(np.sort(luminances))):

        if luminance == 0.25: #for power at mean
            curr_pow = powers[luminance==luminances]
            all_powers[:,0,roi_id] = curr_pow
            continue
        curr_diams = diameters[luminance==luminances]
        curr_pows = powers[luminance==luminances]

        sorted_pow = curr_pows[np.argsort(curr_diams)]
        all_powers[idx,1:,roi_id] = sorted_pow

        
#%%  Power vs luminance in different sizes
mean_powers = all_powers.mean(axis=2)
error_powers = all_powers.std(axis=2)/np.sqrt(len(geno_rois))



for idx, diam in enumerate(np.unique(np.sort(diameters))):
    if diam == 5: #for power at mean
            continue
    curr_lums = real_lums[diam==diameters]
    plt.errorbar(np.sort(curr_lums),mean_powers[[0,1,3,4],idx],error_powers[[0,1,3,4],idx],fmt='-o',color=[diam/diameters.max(), 0, 0],label='d:{s}'.format(s=str(diam)))
# plt.axhline(y=powers[0.25==luminances], color='k', linestyle='--', label='lum:0.25')
plt.legend()
plt.title('{s}, {d} ROIs'.format(s=curr_geno,d= len(geno_rois)))
plt.xlabel('luminance')
plt.ylabel('Power at 1 Hz')
plt.xscale('log')
ax = plt.gca()
# plt.xlim((10000,ax.get_xlim()[1]))

fig1 = plt.gcf()#
f1_n = 'CircCenter_DataSummary_Luminance%s' % (curr_geno)
os.chdir(results_save_dir)
fig1.savefig('%s.pdf'% f1_n, bbox_inches='tight',
            transparent=False)
#%% Power vs size
mean_powers = all_powers.mean(axis=2)
error_powers = all_powers.std(axis=2)/np.sqrt(len(geno_rois))
for idx, luminance in enumerate(np.unique(np.sort(luminances))):
    if luminance == 0.25: #for power at mean
            continue
    curr_diams = diameters[luminance==luminances]
    curr_diams = np.insert(curr_diams,0,5)
    plt.errorbar(np.sort(curr_diams),mean_powers[idx,:],error_powers[idx,:],color=[2*luminance, 0, 0],label='lum:{s}'.format(s=str(luminance)))
# plt.axhline(y=powers[0.25==luminances], color='k', linestyle='--', label='lum:0.25')
plt.legend()
plt.title('{s}, {d} ROIs'.format(s=curr_geno,d= len(geno_rois)))
plt.xlabel('BG circle size')
plt.ylabel('Power at 1 Hz')
fig1 = plt.gcf()#
f1_n = 'CircCenter_DataSummary_%s' % (curr_geno)
os.chdir(results_save_dir)
fig1.savefig('%s.pdf'% f1_n, bbox_inches='tight',
            transparent=False)

#%% Traces
for roi_id, roi in enumerate(geno_rois):
    dim1 = 15
    dim2 = 10
    fig2, axs = plt.subplots(nrows=len(np.unique(luminances)), ncols=len(np.unique(diameters))-1, figsize=(dim1, dim2),sharey=True,sharex=True)

    # Adjust the spacing between subplots
    plt.subplots_adjust(wspace=0.8, hspace=0.2)

    uniq_lums = np.unique(luminances)
    mask = uniq_lums != 0.25  
    uniq_lums = uniq_lums[mask] 

    uniq_diams = np.unique(diameters)
    uniq_diams = uniq_diams[1:] #get rid of 5deg
    epochs = np.array(roi.int_whole_trace.keys())

    f_epoch = epochs[(diameters == 5)]
    t_axis = np.linspace(0,9,len(roi.int_whole_trace[f_epoch[0]]))

    axs[0, 0].plot(t_axis,roi.int_whole_trace[f_epoch[0]])

    axs[0, 0].axvline(x=3, color=[1,0,0,0.4], linestyle='-')
    axs[0, 0].axvline(x=6, color=[1,0,0,0.4], linestyle='-')

    for i_lum, lum in enumerate(uniq_lums):
        for i_diam,diam in enumerate(uniq_diams):
            curr_epoch = epochs[(lum == luminances) & (diameters == diam)]
            axs[i_lum+1, i_diam].plot(t_axis,roi.int_whole_trace[curr_epoch[0]])
            
            axs[i_lum+1, i_diam].set_title('l:{s}\nd:{d}'.format(s=lum,d =diam))
            
            axs[i_lum+1, i_diam].axvline(x=3, color=[1,0,0,0.4], linestyle='-')
            axs[i_lum+1, i_diam].axvline(x=6, color=[1,0,0,0.4], linestyle='-')

            if i_lum+1 == len(uniq_lums):
                axs[i_lum+1, i_diam].set_xlabel('Time (s)')
    axs[i_lum+1, i_diam].set_xlim([0,10])

    f2_n = '%s_Traces' % (roi.experiment_info['MovieID'])
    os.chdir(os.path.join(results_save_dir,"traces"))
    fig2.savefig('%s-%s.pdf'% (curr_geno,f2_n), bbox_inches='tight',
                transparent=False)
    
    # Powers for each
    fig3 = plt.figure(figsize=(5, 5))
    all_powers = np.zeros(shape=(len(np.unique(luminances)), len(np.unique(diameters)) ))
    luminances = np.array(roi.BGcircle_luminance.values())
    diameters = np.array(roi.BGcircle_diameter.values())
    powers = np.array(roi.power_at_sineFreq.values())
    for idx, luminance in enumerate(np.unique(np.sort(luminances))):

        if luminance == 0.25: #for power at mean
            curr_pow = powers[luminance==luminances]
            all_powers[:,0] = curr_pow
            continue
        curr_diams = diameters[luminance==luminances]
        curr_pows = powers[luminance==luminances]

        sorted_pow = curr_pows[np.argsort(curr_diams)]
        all_powers[idx,1:] = sorted_pow

    for idx, luminance in enumerate(np.unique(np.sort(luminances))):
        if luminance == 0.25: #for power at mean
                continue
        curr_diams = diameters[luminance==luminances]
        curr_diams = np.insert(curr_diams,0,5)
        plt.plot(np.sort(curr_diams),all_powers[idx,:],color=[2*luminance, 0, 0],label='lum:{s}'.format(s=str(luminance)))
    # plt.axhline(y=powers[0.25==luminances], color='k', linestyle='--', label='lum:0.25')
    plt.legend()
    plt.xlabel('BG circle size')
    plt.ylabel('Power at 1 Hz')
    f3_n = '%s_Powers' % (roi.experiment_info['MovieID'])
    os.chdir(os.path.join(results_save_dir,"traces"))
    fig3.savefig('%s-%s.pdf'% (curr_geno,f3_n), bbox_inches='tight',
                transparent=False)

 # %% Analysis of the mean response
curr_geno = "Tm1_GC6f"
# Tm9Rec_lexAopGC6f Tm1_GC6f
geno_rois = all_rois[(combined_df['Geno'] ==curr_geno)]
luminances = np.array(all_rois[0].BGcircle_luminance.values())
diameters = np.array(all_rois[0].BGcircle_diameter.values())


all_powers = np.zeros(shape=(len(np.unique(luminances)), len(np.unique(diameters)), len(geno_rois) ))

for roi_id, roi in enumerate(geno_rois):
    luminances = np.array(roi.BGcircle_luminance.values())
    diameters = np.array(roi.BGcircle_diameter.values())
    powers = np.array(roi.mean_resp.values())
    for idx, luminance in enumerate(np.unique(np.sort(luminances))):

        if luminance == 0.25: #for power at mean
            curr_pow = powers[luminance==luminances]
            all_powers[:,0,roi_id] = curr_pow
            continue
        curr_diams = diameters[luminance==luminances]
        curr_pows = powers[luminance==luminances]

        sorted_pow = curr_pows[np.argsort(curr_diams)]
        all_powers[idx,1:,roi_id] = sorted_pow

        
#%%
mean_powers = all_powers.mean(axis=2)
error_powers = all_powers.std(axis=2)/np.sqrt(len(geno_rois))
for idx, luminance in enumerate(np.unique(np.sort(luminances))):
    if luminance == 0.25: #for power at mean
            continue
    curr_diams = diameters[luminance==luminances]
    curr_diams = np.insert(curr_diams,0,5)
    plt.errorbar(np.sort(curr_diams),mean_powers[idx,:],error_powers[idx,:],color=[2*luminance, 0, 0],label='lum:{s}'.format(s=str(luminance)))
# plt.axhline(y=powers[0.25==luminances], color='k', linestyle='--', label='lum:0.25')
plt.legend()
plt.title('{s}, {d} ROIs'.format(s=curr_geno,d= len(geno_rois)))
plt.xlabel('BG circle size')
plt.ylabel('Mean response')
fig1 = plt.gcf()#
f1_n = 'CircCenter_Mean_resp_DataSummary_%s' % (curr_geno)
os.chdir(results_save_dir)
fig1.savefig('%s.pdf'% f1_n, bbox_inches='tight',
            transparent=False)

# %%

#%% Following analyzes and saves the data from two photon experiments done in Göttingen in the Investigator microscope using the C++ stimulation paradigm
# This code is used for  analyzing the responses to gratings with constant contrast and changing luminance
# The stimuli were presented with the software based on C++ 
# Analysis of only T4-T5 GCaMP6f recordings, many temporal frequencies were used here but in the paper we're only showing 1Hz for consistency with L2-L3 recordings. All temporal frequencies show luminance-invariancy
"""
Created on Wed Apr 29 17:07:02 2020

@author: burakgur
"""

#%%
import cPickle
import pandas as pd
import numpy as np
import os
from scipy.stats import linregress, spearmanr
from sklearn import preprocessing

#change to code directory / update this for yourself
os.chdir('/Volumes/Backup Plus/Post-Doc/_SiliesLab/Manuscripts/2023_Lum_Gain/Data_code/code/python2p7/common')

import ROI_mod
import post_analysis_core as pac

#%% Directories for loading data and saving figures (ADJUST THEM TO YOUR PATHS)
initialDirectory = '/Volumes/Backup Plus/Post-Doc/_SiliesLab/Manuscripts/2023_Lum_Gain/Data_code'
# initialDirectory = '/Volumes/HD-SP1/Burak_data/Python_data'
all_data_dir = os.path.join(initialDirectory, 'raw_data')

dataDir = os.path.join(all_data_dir, 'T4_T5_luminanceGain','luminance_gratings') # Raw data folder: "T4_T5_luminanceGain", change reliability in the following code to 0.4

#%% Luminance values of the experimental setup (ADJUST THE PATHS)
# Values from the stimulation paradigm
measurements_f = '/Volumes/Backup Plus/Post-Doc/_SiliesLab/Manuscripts/2023_Lum_Gain/Data_code/code/luminance_measurements/200622_Investigator_Luminances_LumGainPaper.xlsx'
measurement_df = pd.read_excel(measurements_f,header=0)
res = linregress(measurement_df['file_lum'], measurement_df['measured'])

# %% Load datasets and desired variables
exp_t = 'T4T5' 
datasets_to_load = os.listdir(dataDir)

                    
properties = ['BF', 'PD', 'SNR','Reliab','depth','DSI']
combined_df = pd.DataFrame(columns=properties)
all_rois = []
tunings = []
# Initialize variables
flyNum = 0
for dataset in datasets_to_load:
    if not(".pickle" in dataset):
        print('Skipping non pickle file: {d}'.format(d=dataset))
        continue
    load_path = os.path.join(dataDir, dataset)
    load_path = open(load_path, 'rb')
    workspace = cPickle.load(load_path)
    curr_rois = workspace['final_rois']
    
    if 'lightBG' in curr_rois[0].stim_name:
        curr_stim_type = 'dark background'
        continue
    # if 'darkBG' in curr_rois[0].stim_name:
    #     curr_stim_type = 'bright background'
    #     continue
    
    # Reliability thresholding
    curr_rois = ROI_mod.analyze_luminance_gratings_t45(curr_rois)
    curr_rois = ROI_mod.threshold_ROIs(curr_rois, {'reliability':0.4})

    curr_tunings = np.array(map(lambda roi: roi.power_at_hz, curr_rois))
    tunings.append(curr_tunings)

    for roi in curr_rois:
        freqs = roi.frequencies
        curr_tunings = roi.power_at_hz[freqs==1]/roi.power_at_hz[freqs==1].max()

        roi_luminances = roi.luminances[freqs==1]
        candelas = res.intercept + res.slope*roi_luminances
        luminances_photon = pac.convert_cd_to_photons(cdVal=candelas,wavelength=475)
        diff_luminances = luminances_photon

        roi.slope = linregress(diff_luminances, np.transpose(curr_tunings))[0]
        curr_coeff, pval = spearmanr(curr_tunings,diff_luminances)
        roi.luminance_corr = curr_coeff
        roi.luminance_corr_pval = pval


    all_rois.append(curr_rois)
    data_to_extract = ['SNR','reliability','slope','DSI','CS','luminance_corr']
    roi_data = ROI_mod.data_to_list(curr_rois, data_to_extract)
    depths = list(map(lambda roi : roi.imaging_info['depth'], curr_rois))

    df_c = {}
    df_c['depth'] = depths
    df_c['CS'] = roi_data['CS']
    df_c['slope'] = roi_data['slope']
    df_c['spearman_corr'] = roi_data['luminance_corr']
    df_c['SNR'] = roi_data['SNR']
    df_c['Reliab'] = roi_data['reliability']
    df_c['DSI'] = roi_data['DSI']
    df_c['flyID'] = np.tile(curr_rois[0].experiment_info['FlyID'],len(curr_rois))
    df_c['flyNum'] = np.tile(flyNum,len(curr_rois))
    flyNum = flyNum +1
    df = pd.DataFrame.from_dict(df_c) 
    rois_df = pd.DataFrame.from_dict(df)
    combined_df = combined_df.append(rois_df, ignore_index=True, sort=False)
    print('{ds} successfully loaded\n'.format(ds=dataset))

le = preprocessing.LabelEncoder()
le.fit(combined_df['flyID'])
combined_df['flyIDNum'] = le.transform(combined_df['flyID'])

all_rois=np.concatenate(all_rois)
tunings = np.concatenate(tunings)

#%% Save data
freqs = all_rois[0].frequencies
diff_luminances = all_rois[0].luminances[freqs==1]

output_dir = os.path.join(initialDirectory, 'analyzed_data','GUR_etAl_2024')
data_dict = {"freqs": freqs,
             "all_tunings": tunings,
             "tunings": tunings[:,freqs==1], 
             "genotypes":np.array(np.tile("T4-T5",tunings.shape[0])),
             "experimentID":combined_df['flyIDNum'],
             "luminances": diff_luminances,
             "df": combined_df,
             "rois":all_rois
             }


save_name = 'T45-sineGratings-real_lum_vals-rel0p4'
saveVar = open(os.path.join(output_dir, "{s}.pickle".format(s=save_name)), "wb")
cPickle.dump(data_dict, saveVar, protocol=-1)
saveVar.close()
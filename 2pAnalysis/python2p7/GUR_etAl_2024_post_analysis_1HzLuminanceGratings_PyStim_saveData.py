#%% Following analyzes and saves the data from two photon experiments done in Mainz in the Ultima microscope using the PyStim stimulation paradigm
# This code is used for  analyzing the responses to gratings with constant contrast and changing luminance
# The stimuli were presented with the PyStim software based on PsychoPy
# Analysis of only Tm1 and Tm9 GluClalpha flpSTOP (GCaMP6f recordings) and Tm1 iGluSnFr recording
"""
@author: burakgur
"""

#%%
import cPickle
import pandas as pd
import numpy as np
import warnings
import os
from scipy.stats import linregress
from sklearn import preprocessing
from scipy import stats

#change to code directory
os.chdir('.../Gur-etal-2024/2pAnalysis/python2p7/common')

import PyROI
import post_analysis_core as pac

#%% Directories for loading data and saving figures (ADJUST THEM TO YOUR PATHS)
initialDirectory = 'data_path'
all_data_dir = os.path.join(initialDirectory, 'raw_data')

#%% Luminance values of the experimental setup (ADJUST THE PATHS)
# Values from the stimulation paradigm
measurements_f = '.../Gur-etal-2024/2pAnalysis/luminance_measurements/210716_Ultima_Luminances_ONOFFpaper.xlsx'
measurement_df = pd.read_excel(measurements_f,header=0)
res = linregress(measurement_df['file_lum'], measurement_df['measured'])
# %% Load datasets and desired variables
exp_folder = 'Tm1GluClalphaFLPSTOP' # Raw data folder: "Tm1GluClalphaFLPSTOP", change reliability in the following code to 0.5
# exp_folder = 'Tm9GluClalphaFLPSTOP' # Raw data folder: "Tm9GluClalphaFLPSTOP", change reliability in the following code to 0.5
# exp_folder = 'Tm1iGluSnFR' # Raw data folder: "Tm1iGluSnFR", change reliability in the following code to 0.3

data_dir = os.path.join(all_data_dir,exp_folder)
datasets_to_load = os.listdir(data_dir)

                    
properties = ['Reliab','depth','slope']
combined_df = pd.DataFrame(columns=properties)
all_rois = []
all_traces=[]
tunings = []
baselines = []
baseline_power = []
z_tunings = []
# Initialize variables

polarity_dict={'Mi1' : 'ON','Tm9' : 'OFF','Mi4' : 'ON',
         'L1_' : 'OFF','L2_' : 'OFF',
         'L3_': 'OFF','Tm1': 'OFF',
         'Tm3' : 'ON'}

plot_only_cat = True
cat_dict={
         'L1_' : 'M1',
         'Tm3' : 'M9'}
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
    
    if not('stim-input_gratings-100p-5lum-40s' in curr_rois[0].stim_info['meta']['stim_name']) or \
        ('singleROI' in dataset) :
        continue
    
    # Thresholding
    # Reliability thresholding
    curr_rois = PyROI.analyzeSineGratings(curr_rois)
    curr_rois = PyROI.calculateReliability(curr_rois)
    # max power threshold is 0.015 for Tm1 iGluSnfr
    # rel threshold for Tm1 Tm9 GluCl 0.5 - Tm1 iGluSnfr 0.3
    curr_rois = PyROI.threshold_ROIs(curr_rois, {'reliability':0.5})

    if len(curr_rois) <= 1:
        warnings.warn('{d} contains 1 or no ROIs, skipping'.format(d=dataset))
        continue
    geno = curr_rois[0].experiment_info['Genotype']
    # if "Tm1" in geno: # For Tm9 folder since it has 
        # continue 
    print(geno)
    print(curr_rois[0].stim_info['meta']['stim_name'])

    for roi in curr_rois:
        roi_lums = roi.sorted_luminances
        candelas = res.intercept + res.slope*roi_lums
        luminances_photon = pac.convert_cd_to_photons(cdVal=candelas,wavelength=475)
        roi.sorted_luminances = luminances_photon


        curr_tunings = roi.sorted_power_at_sineFreq/roi.sorted_power_at_sineFreq.max()
        roi.power_lum_slope = linregress(luminances_photon, np.transpose(curr_tunings))[0]

    all_rois.append(curr_rois)
    
    
    # There is just one ROI with 3Hz so discard that one
    
    curr_tuning = np.squeeze\
                   (list(map(lambda roi: roi.sorted_power_at_sineFreq,curr_rois)))
    tunings.append(curr_tuning)
    z_tunings.append(stats.zscore(curr_tuning,axis=1))
    
    curr_base = np.squeeze\
                   (list(map(lambda roi: roi.sorted_mean_resps,curr_rois)))
    baselines.append(curr_base)

    depths = list(map(lambda roi : roi.imaging_info['depth'], curr_rois))
    
    df_c = {}

    data_to_extract = ['reliability','power_lum_slope','category','base_slope']
    roi_data = PyROI.data_to_list(curr_rois, data_to_extract)

    df_c['depth'] = depths
    df_c['slope'] = roi_data['power_lum_slope']
    df_c['base_slope'] = roi_data['base_slope']
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
tunings = np.concatenate(tunings)
z_tunings = np.concatenate(z_tunings)
baselines = np.concatenate(baselines)


#%% Save data
output_dir = os.path.join(initialDirectory, 'Figure6','processed_data')
data_dict = {"tunings": tunings,
             "genotypes":combined_df['Geno'],
             "experimentID":combined_df['flyIDNum'],
             "luminances": luminances_photon,
             "df": combined_df,
             "rois":all_rois
             }


save_name = 'Tm1GluCl-sineGratings-real_lum_vals_rel0p5'
saveVar = open(os.path.join(output_dir, "{s}.pickle".format(s=save_name)), "wb")
cPickle.dump(data_dict, saveVar, protocol=-1)
saveVar.close()

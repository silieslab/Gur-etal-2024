
#%% Following analyzes and saves the data from two photon experiments done in Göttingen in the Investigator microscope
# This code is used for  analyzing the responses to gratings with constant contrast and changing luminance
# The stimuli were presented with the software based on c++ 
# Analysis of only L2-L3-Tm1-Tm2-Tm4-Tm9 GCaMP6f and Tm9 iGluSnFr recordings
#%% Import directories
import cPickle
import pandas as pd
import numpy as np
import warnings
import os
from scipy.stats import linregress, spearmanr
from sklearn import preprocessing
from scipy import stats

#change to code directory
os.chdir('.../Gur-etal-2024/2pAnalysis/python2p7/common')

import ROI_mod
import post_analysis_core as pac


#%% Directories for loading data and saving figures (ADJUST THEM TO YOUR PATHS)
initialDirectory = 'data_path'
all_data_dir = os.path.join(initialDirectory, 'raw_data')

#%% Luminance values of the experimental setup (ADJUST THE PATHS)
# Values from the stimulation paradigm
measurements_f = '.../Gur-etal-2024/2pAnalysis/luminance_measurements/200622_Investigator_Luminances_LumGainPaper.xlsx'
measurement_df = pd.read_excel(measurements_f,header=0)
res = linregress(measurement_df['file_lum'], measurement_df['measured'])
# %% Load datasets and desired variables
exp_folder = 'L2-L3-TmX-edges-1Hzsine' # For Lamina and medulla neurons GCaMP6f
                                    # Raw data folder: L2-L3-TmX-edges-1Hzsine change reliability in the following code to 0.6
# exp_folder = 'Tm9iGluSnFR' # For iGluSnFr Tm9
                                    # Raw data folder: Tm9iGluSnFR change reliability in the following code to 0.3

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

polarity_dict={'Tm9' : 'OFF','L2_' : 'OFF',
         'L3_': 'OFF','Tm1': 'OFF','Tm4' : 'OFF','Tm2' : 'OFF'}

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
    curr_rois = workspace['final_rois']
    # Check if the stimulus is the one we need for analysis
    if (not('1Dir_meanBG_DriftingSine_1Hz100contrast_6luminances_48s' in curr_rois[0].stim_name)):
        continue

    # Thresholding
    # Reliability thresholding
    curr_rois = ROI_mod.analyze_luminance_gratings_1Hz(curr_rois)
    curr_rois = ROI_mod.threshold_ROIs(curr_rois, {'reliability':0.6}) # For Lamina and medulla GCaMP6f
    # curr_rois = ROI_mod.threshold_ROIs(curr_rois, {'reliability':0.3}) # For iGluSnFr Tm9


    if len(curr_rois) <= 1:
        warnings.warn('{d} contains 1 or no ROIs, skipping'.format(d=dataset))
        continue
    
    geno = curr_rois[0].experiment_info['Genotype'][:3]
    # geno = curr_rois[0].experiment_info['Genotype']
    if geno not in polarity_dict.keys():
        continue
    for roi in curr_rois:
        roi_lums = roi.luminances
        candelas = res.intercept + res.slope*roi_lums
        luminances_photon = pac.convert_cd_to_photons(cdVal=candelas,wavelength=475)
        diff_luminances = luminances_photon


        curr_tunings = roi.power_at_hz/roi.power_at_hz.max()
        roi.slope = linregress(diff_luminances, np.transpose(curr_tunings))[0]

        curr_coeff, pval = spearmanr(curr_tunings,diff_luminances)
        roi.luminance_corr = curr_coeff
        roi.luminance_corr_pval = pval

    all_rois.append(curr_rois)
    data_to_extract = ['SNR','reliability','slope','category','base_slope','luminance_corr']
    roi_data = ROI_mod.data_to_list(curr_rois, data_to_extract)
    
    
    curr_tuning = np.squeeze\
                   (list(map(lambda roi: roi.power_at_hz,curr_rois)))
    tunings.append(curr_tuning)
    z_tunings.append(stats.zscore(curr_tuning,axis=1))
    
    curr_base = np.squeeze\
                   (list(map(lambda roi: roi.baselines,curr_rois)))
    baselines.append(curr_base)

    curr_baseP = np.squeeze\
                   (list(map(lambda roi: roi.base_power,curr_rois)))
    baseline_power.append(curr_baseP)
    
    print(curr_rois[0].experiment_info['Genotype'])
    print(curr_rois[0].stim_name)
    depths = list(map(lambda roi : roi.imaging_info['depth'], curr_rois))
    
    df_c = {}
    df_c['depth'] = depths
        
    
    # df_c['CS'] = roi_data['CS']
    df_c['SNR'] = roi_data['SNR']
    df_c['spearman_corr'] = roi_data['luminance_corr']
    df_c['slope'] = roi_data['slope']
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
baseline_power = np.concatenate(baseline_power)

#%% Save data
output_dir = os.path.join(initialDirectory, 'Figure1','processed_data')
data_dict = {"tunings": tunings,
             "genotypes":combined_df['Geno'],
             "experimentID":combined_df['flyIDNum'],
             "luminances": diff_luminances,
             "df": combined_df,
             "rois":all_rois
             }


save_name = 'Lamina-Medulla-Neurons-sineGratings-real_lum_vals_rel0p6'
saveVar = open(os.path.join(output_dir, "{s}.pickle".format(s=save_name)), "wb")
cPickle.dump(data_dict, saveVar, protocol=-1)
saveVar.close()
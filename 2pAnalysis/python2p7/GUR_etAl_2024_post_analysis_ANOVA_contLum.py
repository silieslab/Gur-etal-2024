#%% Following does statistical analysis on 5 contrast 5 luminance moving gratings data.
"""
@author: burakgur
"""
#%% Imports
import sys
import cPickle
import pandas as pd
import numpy as np
import os
import scipy.stats as stats
from scipy.stats import f_oneway
# Change to code folder
os.chdir('.../Gur-etal-2024/2pAnalysis/python2p7/common')

#%%
main_dir = 'data_path'
data_dir = os.path.join(main_dir,'processed_data')
results_save_dir = os.path.join(main_dir,'plots')
#%% Load data
load_path = os.path.join(data_dir, 'Tm9_AT_ConLum.pickle')
with open(load_path, 'rb') as f: 
    dat = cPickle.load(f)
    
all_data = dat['all_fly_means']
luminances = dat['luminances']
contrasts = dat['contrast']
geno = dat['genotype']

#%%
curr_str = 'Tm9_AT'
file_path = os.path.join(results_save_dir,f'lumCon_stats_{curr_str}.txt')
with open(file_path, 'w') as file:
    sys.stdout = file
    for idx, contrast in enumerate(contrasts):
        print('----\n----\n')
        data = all_data[:,:,idx]
        # Normality check
        # Perform Shapiro-Wilk test for each group
        for idx, luminance in enumerate(luminances):
            shap_stat, shap_p = stats.shapiro(data[:,idx])
            print(f'Shapiro test for c:{contrast}, l:{luminance} -> {shap_p}')

        # Check homogeneity of variances
        # Perform Shapiro-Wilk test for each group
        statistic, p_value = stats.levene(data[:,0], data[:,1], data[:,2],data[:,3],data[:,4])
        print(f'\nLevene test for c:{contrast} -> {p_value}')
        
    #anova
    F, p = f_oneway(all_data[:,0,:], all_data[:,1,:], all_data[:,2,:],all_data[:,3,:],all_data[:,4,:])

    print('----\nANOVA\n----\n')
    for idx, pp in enumerate(p):
        print(f'ANOVA for c:{contrasts[idx]} -> {pp}')
sys.stdout = sys.__stdout__
print(f"Output has been saved to: {file_path}")


# %%
#%%
curr_str = 'L3_AT'
for idx, contrast in enumerate(contrasts):
    print('----\n----\n')
    data = all_data[:,:,idx]
    # Normality check
    # Perform Shapiro-Wilk test for each group
    for idx, luminance in enumerate(luminances):
        shap_stat, shap_p = stats.shapiro(data[:,idx])
        print(f'Shapiro test for c:{contrast}, l:{luminance} -> {shap_p}')

    # Check homogeneity of variances
    # Perform Shapiro-Wilk test for each group
    statistic, p_value = stats.levene(data[:,0], data[:,1], data[:,2],data[:,3],data[:,4])
    print(f'\nLevene test for c:{contrast} -> {p_value}')
    
#anova
F, p = f_oneway(all_data[:,0,:], all_data[:,1,:], all_data[:,2,:],all_data[:,3,:],all_data[:,4,:])

print('----\nANOVA\n----\n')
for idx, pp in enumerate(p):
    print(f'ANOVA for c:{contrasts[idx]} -> {pp}') # Adjust the f-string
# %%

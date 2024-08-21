#%% Following analyzes all of the processed data of 1Hz gratings with 100% contrast and 5 luminance
# This code is used for analyzing the processed data of responses to 1Hz gratings with constant contrast (100%) and changing luminance
# Plots, stats
"""
@author: burakgur
"""
#%% Imports
import cPickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn import preprocessing

# Change to code folder
os.chdir('.../Gur-etal-2024/2pAnalysis/python2p7/common')

import post_analysis_core as pac
#%% Directories for loading data and saving figures (ADJUST THEM TO YOUR PATHS)
main_dir = 'data_path'
data_dir = os.path.join(main_dir,'Figure1','processed_data') # Change this to load your data of interest
results_save_dir = os.path.join(main_dir,'Figure1/plots')
#%% Load data

# Lamina medulla data
load_path = os.path.join(data_dir, 'Lamina-Medulla-Neurons-sineGratings-real_lum_vals_rel0p6.pickle')
load_path = open(load_path, 'rb')
lam_med = cPickle.load(load_path)

lam_med_df = lam_med['df'][['slope','flyID','Geno']]
lam_med_tunings = lam_med['tunings']



# Tm9 GluCl
load_path = os.path.join(data_dir, 'Tm9GluCl-sineGratings-real_lum_vals_rel0p5.pickle')
load_path = open(load_path, 'rb')
tm9glucl = cPickle.load(load_path)

tm9glucl_df = tm9glucl['df'][['slope','flyID','Geno']]
tm9glucl_tunings = lam_med['tunings']


# Tm1 GluCl
load_path = os.path.join(data_dir, 'Tm1GluCl-sineGratings-real_lum_vals_rel0p5.pickle')
load_path = open(load_path, 'rb')
tm1glucl = cPickle.load(load_path)

tm1glucl_df = tm1glucl['df'][['slope','flyID','Geno']]
tm1glucl_tunings = lam_med['tunings']


# Tm1 iGlu
load_path = os.path.join(data_dir, 'Tm1iGLu-sineGratings-real_lum_vals_rel0p3.pickle')
load_path = open(load_path, 'rb')
tm1iglu = cPickle.load(load_path)

tm1iglu_df = tm1iglu['df'][['slope','flyID','Geno']]
tm1iglu_tunings = lam_med['tunings']

# T45 data
load_path = os.path.join(data_dir, 'T45-sineGratings-real_lum_vals-rel0p4.pickle')
load_path = open(load_path, 'rb')
t45 = cPickle.load(load_path)
t45['df']['Geno'] = np.array(np.tile("T45",t45['df'].shape[0]))
t45_df = t45['df'][['slope','flyID','Geno']]
t45_tunings = lam_med['tunings']

# Tm9iGlu
load_path = os.path.join(data_dir, 'Tm9iGluSnfr-sineGratings-real_lum_vals_rel0p3.pickle')
load_path = open(load_path, 'rb')
tm9iGlu = cPickle.load(load_path)
tm9iGlu['df']['Geno'] = np.array(np.tile("9Glu",tm9iGlu['df'].shape[0]))
tm9iGlu_df = tm9iGlu['df'][['slope','flyID','Geno']]
tm9iGlu_tunings = lam_med['tunings']

# Dm data
load_path = os.path.join(data_dir, 'Dm12silencing_Tm9_logLum.pickle')
load_path = open(load_path, 'rb')
dm = pd.read_pickle(load_path)

dm_df = dm['df'][['slope','flyID','Geno']]
dm_tunings = dm['tunings']

#%% Recalculate slopes (for log-luminance values)
# Luminance values
input_lum_vals = np.array([0.0625, 0.125,  0.25,   0.375,  0.5])
measurements_f = '.../Gur-etal-2024/2pAnalysis/luminance_measurements/210716_Ultima_Luminances_ONOFFpaper.xlsx' # Enter the path to luminance data
measurement_df = pd.read_excel(measurements_f,header=0)
res = linregress(measurement_df['file_lum'], measurement_df['measured'])
candelas = res.intercept + res.slope*input_lum_vals
luminances_photon = pac.convert_cd_to_photons(cdVal=candelas,wavelength=475)
diff_luminances_Ultima = luminances_photon
        
measurements_f = '.../Gur-etal-2024/2pAnalysis/luminance_measurements/200622_Investigator_Luminances_LumGainPaper.xlsx' # Enter the path to luminance data
measurement_df = pd.read_excel(measurements_f,header=0)
res = linregress(measurement_df['file_lum'], measurement_df['measured'])
candelas = res.intercept + res.slope*input_lum_vals
luminances_photon = pac.convert_cd_to_photons(cdVal=candelas,wavelength=475)
diff_luminances_Investigator = luminances_photon


# Perform linear regression for all variables that contain "_tunings" in log lum scale
for data_name in ["tm9iGlu_tunings", "t45_tunings", "lam_med_tunings", "tm9glucl_tunings", "tm1glucl_tunings", "tm1iglu_tunings"]:
    if data_name in ["tm1iglu_tunings", "tm9glucl_tunings", "tm1glucl_tunings"]:
        diff_luminances = diff_luminances_Ultima
    else:
        diff_luminances = diff_luminances_Investigator
    data_n = data_name[:data_name.rfind("_")]
    
    x = np.log10(diff_luminances)
    curr_tunings = eval(data_name)
    curr_tunings= curr_tunings/curr_tunings.max(axis=1)[:,np.newaxis] # Normalization
    for idx, roi_tuning in enumerate(curr_tunings):
        y = roi_tuning
        
        reg_data = linregress(x, y)
        eval('{data_n}_df'.format(data_n=data_n))['slope'].iloc[idx] =  reg_data.slope # Change to python 2
       
    
#%% Combine data
combined_tuning = np.concatenate((tm9iGlu['tunings'], t45['tunings'],lam_med['tunings'],tm9glucl['tunings'],tm1glucl['tunings'],tm1iglu['tunings']))
combined_df = pd.concat([tm9iGlu_df, t45_df, lam_med_df, tm9glucl_df, tm1glucl_df,tm1iglu_df])

le = preprocessing.LabelEncoder()
le.fit(combined_df['flyID'])
combined_df['flyIDNum'] = le.transform(combined_df['flyID'])

#%% Colors

_, colors = pac.run_matplotlib_params()

c_dict = {k:colors[k] for k in colors if k in combined_df['Geno'].unique()}


#%% Slope stats
import scipy.stats as stats
import statsmodels.stats.multicomp as mc
import statsmodels.api as sm
from statsmodels.formula.api import ols
# import scikit_posthocs as sp
# Per fly data
a=pac.compute_over_samples_groups(data = combined_df['slope'], 
                                group_ids= np.array(combined_df['flyIDNum']), 
                                error ='SEM',
                                experiment_ids = np.array(combined_df['Geno']))

# HERE PICK THE GENOTYPES to compare
# Figure 1: "T45","L3_","L2_"
data1 = np.array(a['experiment_ids']["L2_"]['over_samples_means'])
data2 = np.array(a['experiment_ids']["Tm1"]['over_samples_means'])
data3 = np.array(a['experiment_ids']["Tm1GMR74G01Gal4_iGluSnfr"]['over_samples_means'])


# Normality check
# Perform Shapiro-Wilk test for each group
stat_group1, p_group1 = stats.shapiro(data1)
stat_group2, p_group2 = stats.shapiro(data2)
stat_group3, p_group3 = stats.shapiro(data3)

print("Shapiro-Wilk test results:")
print("Group 1 Shapiro normality p-value: {s}".format(s=p_group1))
print("Group 2 Shapiro normality p-value: {s}".format(s=p_group2))
print("Group 3 Shapiro normality p-value: {s}".format(s=p_group3))

# Check homogeneity of variances
# Perform Shapiro-Wilk test for each group
statistic, p_value = stats.levene(data1, data2, data3)

print("Levene test results:")
print("Levene variance check p-value: {s}".format(s=p_value))

# T test
t_stat, p_value = stats.ttest_ind(data2, data3)
print("T test of data2 and data3 p-value: {s}".format(s=p_value))

# ANOVA
data_all = [data1, data2, data3]
dfs = [pd.DataFrame({k: sample}) for k, sample in enumerate(data_all)]
stats_df = pd.concat(dfs,  ignore_index=True, axis=1)
stats_df.columns = ["L2_","Tm1","Tm1GMR74G01Gal4_iGluSnfr"]

d_melt = pd.melt(stats_df.reset_index(), id_vars=['index'], value_vars= ["L2_","Tm1","Tm1GMR74G01Gal4_iGluSnfr"])
d_melt = d_melt.dropna()
d_melt.columns = ['index', 'geno', 'value']
fvalue, pvalue = stats.f_oneway(data1,data2,data3)

model = ols('value ~ C(geno)', data=d_melt).fit()
anova_table = sm.stats.anova_lm(model, typ=2)
anova_table

comp = mc.MultiComparison(d_melt['value'], d_melt['geno'])
post_hoc_res = comp.tukeyhsd()
post_hoc_res.summary()
post_hoc_res.plot_simultaneous(ylabel= "Geno", xlabel= "Score Difference")


plt.rc('figure', figsize=(12, 7))
plt.text(0.01, 0.05, str(post_hoc_res.summary()), {'fontsize': 10}, fontproperties = 'monospace') # approach improved by OP -> monospace!
plt.axis('off')
plt.tight_layout()



plt.close('all')


#%% Slope bar
fig = plt.figure(figsize=(8, 8))
grid = plt.GridSpec(1, 1, wspace=0.3, hspace=0.3)
ax1=plt.subplot(grid[0,0])
for idx, geno in enumerate(np.unique(d_melt['geno'])):
    # sns.barplot(x='geno',y='value',data=d_melt,
    #                palette=c_dict,ax=ax1,ci="sd")
    pac.bar_bg(d_melt[d_melt['geno']==geno]['value'], idx+1, color=c_dict[geno], scat_s =7,ax=ax1, yerr=None, 
            errtype='SEM',alpha = .6,width=0.8,label=geno)
plt.legend()
save_name = 'Slope_barr_Tm1iGlu'
os.chdir(results_save_dir)
fig.savefig('%s.pdf' % save_name, bbox_inches='tight',dpi=300)

#%% Plot all genotypes together
plt.close('all')
fig = plt.figure(figsize=(5, 5))
grid = plt.GridSpec(1, 1, wspace=0.3, hspace=1)
ax1=plt.subplot(grid[0,0])
for idx, geno in enumerate(np.unique(combined_df['Geno'])):
    geno_color = c_dict[geno]
    curr_neuron_mask = (combined_df['Geno']==geno) 
    if not(geno in  ["L2_","Tm1","Tm1GMR74G01Gal4_iGluSnfr"]):
        continue
    curr_df = combined_df[curr_neuron_mask]
    
    if geno=="T45":
        luminances = np.array([ 12629.9564665 ,  26095.51250893,  53026.6245938 ,  79957.73667867,
       106888.84876353])[::-1]
    else:
        luminances = lam_med['luminances']
    
    curr_tunings = combined_tuning[curr_neuron_mask]
    curr_tunings = curr_tunings/curr_tunings.max(axis=1).reshape(len(curr_tunings),1)
    # curr_tunings = curr_tunings/(curr_tunings[:,0].reshape(len(curr_tunings),1))
    # Normalized
    a=pac.compute_over_samples_groups(data = curr_tunings, 
                                group_ids= np.array(combined_df[curr_neuron_mask]['flyIDNum']), 
                                error ='SEM',
                                experiment_ids = np.array(combined_df[curr_neuron_mask]['Geno']))

    label = '{g} n: {f}({ROI})'.format(g=geno,
                                       f=len(a['experiment_ids'][geno]['over_samples_means']),
                                       ROI=len(a['experiment_ids'][geno]['all_samples']))
    
    
    fly_means = np.array(a['experiment_ids'][geno]['over_samples_means'])
    fly_means /= fly_means.max(axis=1).reshape(len(fly_means),1)
    fly_sem = np.std(fly_means,axis=0)/np.sqrt(fly_means.shape[0])
    fly_mean = np.mean(fly_means,axis=0)
    ax1.errorbar(luminances,fly_mean,fly_sem,
                 fmt='-o',alpha=1,color=geno_color,label=label)

ax1.set_ylim((0,1))
ax1.set_xscale('log')
ax1.legend()
ax1.set_xlim((10000,ax1.get_xlim()[1]))
# Saving figure
save_name = 'Sine_100p_Normalized_Tm1iGlu'.format(geno=geno)
os.chdir(results_save_dir)
fig.savefig('%s.pdf' % save_name, bbox_inches='tight',dpi=300)

